"""
- Teacher: DuETT(TS) + RAD-DINO(CXR) + Distillation Structure + head
- Student: DuETT(TS) + head
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from transformers import AutoModel

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from duett.duett import Model as DuettBase


# Encocder
# DuETT feature extractor
class DuettFeatureExtractor(DuettBase):
    @property
    def d_representation(self) -> int:
        return self.d_embedding * (self.d_time_series_num + 1)

    def encode(self, x):
        xs_static, xs_feats, xs_times, n_timesteps = x
        n_vars = xs_feats.shape[2] // 2

        if self.predict_events:
            event_mask_inds = xs_feats[:, :, n_vars:n_vars * 2] == -1
            event_mask_inds = torch.cat(
                (event_mask_inds,
                 torch.zeros(xs_feats.shape[:2] + (1,), device=xs_feats.device, dtype=torch.bool)),
                dim=2)
            event_mask_inds = torch.cat((event_mask_inds, event_mask_inds[:, :1, :]), dim=1)

        n_obs_inds = xs_feats[:, :, n_vars:n_vars * 2].to(int).clip(0, self.n_obs_embedding.num_embeddings - 1)
        xs_feats = xs_feats.clone()
        xs_feats[:, :, n_vars:n_vars * 2] = self.n_obs_embedding(n_obs_inds).squeeze(-1)

        embedding_layer_input = torch.empty(
            xs_feats.shape[:-1] + (n_vars, 2), dtype=xs_feats.dtype, device=xs_feats.device)
        embedding_layer_input[:, :, :, 0] = xs_feats[:, :, :n_vars]
        embedding_layer_input[:, :, :, 1] = xs_feats[:, :, n_vars:n_vars * 2]

        psi = torch.zeros(
            (xs_feats.shape[0], xs_feats.shape[1] + 1, n_vars + 1, self.d_embedding),
            dtype=xs_feats.dtype, device=xs_feats.device) # [B, T+1, n_vars+1, d_embedding]
        
        for i, el in enumerate(self.embedding_layers):
            psi[:, :-1, i, :] = el(embedding_layer_input[:, :, i, :])

        psi[:, :-1, -1, :] = self.tab_encoder(xs_static).unsqueeze(1)
        psi[:, -1, :, :] = self.special_embeddings(
            self.REPRESENTATION_EMBEDDING_KEY.to(self.device)).unsqueeze(0).unsqueeze(1)

        mask_inds = torch.cat(
            (xs_feats[:, :, -1] == 1, torch.zeros((xs_feats.shape[0], 1), device=xs_feats.device, dtype=torch.bool)), dim=1)
        psi[mask_inds, :, :] = self.special_embeddings(self.MASKED_EMBEDDING_KEY.to(self.device))
        if self.predict_events:
            psi[event_mask_inds, :] = self.special_embeddings(self.MASKED_EMBEDDING_KEY.to(self.device))

        time_embeddings = self.full_time_embedding(xs_times.unsqueeze(2))
        time_embeddings = torch.cat(
            (time_embeddings, self.full_rep_embedding.weight.T.unsqueeze(0).expand(xs_feats.shape[0], -1, -1)), dim=1)
        
        for event_transformer, time_transformer in zip(self.event_transformers, self.time_transformers):
            et_out_shape = (psi.shape[0], psi.shape[2], psi.shape[1], psi.shape[3])
            """
            # Event Transformer: 변수 간 상호작용
            시간축과 변수축을 바꾸고, 마지막 두 축을 합침. [B, T+1, n_vars+1, d_embedding] → [B, n_vars+1, (T+1)*d_embedding]
            - Event transformer에서 하나의 token은 "특정 변수의 전체 시간 궤적"
            ex) x_hr = [psi_{1, hr}, psi_{2, hr}, ..., psi_{T, hr}]
            - Event transformer에서는 self-attention을 통해 여러 임상 변수의 전체 24시간 패턴 간 관계를 학습함.
            """
            embeddings = psi.transpose(1, 2).flatten(2) + self.full_event_embedding.weight.unsqueeze(0)
            event_outs = event_transformer(embeddings).view(et_out_shape).transpose(1, 2) # Event transformer 통과 후 다시 원래 shape로
            tt_out_shape = event_outs.shape
            """
            # Time Transformer: 시간 간 상호작용
            - Time transformer에서 하나의 token은 특정 시간의 모든 변수 상태
            ex) x_{t=1} = [psi_{1,hr}, psi_{1, bp}, ..., psi_{1, temp}]
            - 서로 다른 시간 bin의 전체 임상 상태 관계를 학습함.
            """
            embeddings = event_outs.flatten(2) + time_embeddings
            psi = time_transformer(embeddings).view(tt_out_shape) # 다시 [B, T+1, V+1, d] 형태로 reshape

        transformed = psi.flatten(2)  # [B, T+1, d]
        return transformed, psi


# Load a pretrained DuETT SSL/FT checkpoint as a feature extractor
def load_duett_backbone(ckpt_path: str,
                        d_static_num: int,
                        d_time_series_num: int,
                        n_timesteps: int,
                        freeze: bool = False,
                        aug_noise: float = 0.0,
                        aug_mask: float = 0.0,
                        transformer_dropout: float = 0.0) -> DuettFeatureExtractor:
    model = DuettFeatureExtractor.load_from_checkpoint(
        ckpt_path,
        pretrain=False,
        d_static_num=d_static_num,
        d_time_series_num=d_time_series_num,
        d_target=1,
        masked_transform_timesteps=n_timesteps,
        max_len=n_timesteps,
        aug_noise=aug_noise,
        aug_mask=aug_mask,
        transformer_dropout=transformer_dropout,
        strict=False,
    )
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    return model


# =============================================================================
# CXR encoder: RAD-DINO CLS
# =============================================================================
class CXREncoder(nn.Module):
    """RAD-DINO wrapper.

    return_patches=False (default): CLS 토큰만 [B, D] 반환
    return_patches=True:            (cls, patches) 튜플 반환 — patches [B, N, D]
    """

    def __init__(self, model_name: str = "microsoft/rad-dino",
                 freeze: bool = True, return_patches: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.d_out = self.backbone.config.hidden_size
        self.return_patches = return_patches
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self._frozen = freeze

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor):
        out = self.backbone(pixel_values=pixel_values)
        tokens = out.last_hidden_state
        cls = tokens[:, 0]                       # [B, D]
        if self.return_patches:
            return cls, tokens[:, 1:]            # ([B, D], [B, N, D])
        return cls


##########################################################################################
##########################################################################################
# Teacher

# =============================================================================
# Temporal Perceiver fusion
# =============================================================================
"""
여러 버전으로 실험 진행함
[1] Time token + [REP] + IMG [CLS]
[2] CXR이 T축 attend
[3] Latent가 시계열 먼저 보고, 이미지를 보게 하는 방법
"""

# [1] Time token + [REP] + IMG [CLS]
# class TemporalPerceiver(nn.Module):
#     def __init__(self, d_ts: int, d_img: int, d_latent: int = 256,
#                  n_latents: int = 16, n_layers: int = 2, n_heads: int = 4,
#                  dropout: float = 0.1):
#         super().__init__()
#         self.d_latent = d_latent
#         self.n_latents = n_latents
#         self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)

#         self.ts_proj = nn.Linear(d_ts, d_latent)
#         self.img_proj = nn.Linear(d_img, d_latent)

#         self.blocks = nn.ModuleList([
#             _PerceiverBlock(d_latent, n_heads, dropout) for _ in range(n_layers)
#         ])
#         self.norm_out = nn.LayerNorm(d_latent)

#     def forward(self, ts_tokens: torch.Tensor, img_cls: torch.Tensor) -> torch.Tensor:
#         """
#         ts_tokens: [B, T', d_ts]
#         img_cls:   [B, d_img]
#         returns:   [B, d_latent] (mean over latents)
#         """
#         B = ts_tokens.size(0)
        
#         # 두 모달리티를 같은 latent dim으로 project
#         ts_kv = self.ts_proj(ts_tokens)                       # [B, 24+1, D]
#         img_kv = self.img_proj(img_cls).unsqueeze(1)          # [B, 1, D]

#         # T축을 기준으로 concat함
#         kv = torch.cat([ts_kv, img_kv], dim=1)                # [B, 25+1, D]

#         # 학습 가능한 latent query와 cross attention
#         latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
#         for blk in self.blocks:
#             latents = blk(latents, kv)
#         latents = self.norm_out(latents)
#         return latents.mean(dim=1)                             # [B, D] (Global Average Pooling)

# [2] CXR이 T축 attend → [REP, CXR_CLS, img_ctx] 3-token fusion
# class TemporalPerceiver(nn.Module):
#     def __init__(self, d_ts: int, d_img: int, d_latent: int = 256, n_latents: int = 16, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
#         super().__init__()
#         self.d_latent = d_latent
#         self.n_latents = n_latents
#         self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)

#         self.ts_proj = nn.Linear(d_ts, d_latent)
#         self.img_proj = nn.Linear(d_img, d_latent)

#         # Stage 1: CXR CLS가 시간 토큰을 attend → img_ctx
#         self.img_query_ts = _PerceiverBlock(d_latent, n_heads, dropout)

#         # Stage 2: latent 쿼리가 [REP, CXR_CLS, img_ctx]를 attend
#         self.blocks = nn.ModuleList([
#             _PerceiverBlock(d_latent, n_heads, dropout) for _ in range(n_layers)
#         ])
#         self.norm_out = nn.LayerNorm(d_latent)

#     def forward(self, ts_tokens: torch.Tensor, img_cls: torch.Tensor) -> torch.Tensor:
#         B = ts_tokens.size(0)

#         # 두 modality를 같은 latent dim으로 project
#         ts_kv = self.ts_proj(ts_tokens)                    # [B, T+1, D]
#         img_q = self.img_proj(img_cls).unsqueeze(1)        # [B, 1, D]

#         # DuETT encode에서 psi[:, -1, :]에 REP 임베딩을 넣으므로 마지막이 REP
#         ts_time = ts_kv[:, :-1, :]                         # [B, T, D]  시간 토큰만
#         rep     = ts_kv[:, -1:, :]                         # [B, 1, D]  REP token

#         # Stage 1 — CXR가 시간 토큰들을 attend해서 image-conditioned TS 요약
#         img_ctx = self.img_query_ts(img_q, ts_time)        # [B, 1, D]

#         # Fusion 집합: DuETT의 REP / CXR 원본 / CXR이 뽑은 TS 요약
#         summary = torch.cat([rep, img_q, img_ctx], dim=1)  # [B, 3, D]

#         # Stage 2 — latent 쿼리가 3개 summary 토큰을 cross-attend
#         latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
#         for blk in self.blocks:
#             latents = blk(latents, summary)
#         latents = self.norm_out(latents)
#         return latents.mean(dim=1)                             # [B, D]


# [3] Latent가 시계열 먼저 보고, 이미지를 보게 하는 방법
# class TemporalPerceiver(nn.Module):
#     def __init__(self, d_ts: int, d_img: int, d_latent: int = 256, n_latents: int = 16, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
#         super().__init__()
#         self.d_latent = d_latent
#         self.n_latents = n_latents
#         self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)

#         self.ts_proj = nn.Linear(d_ts, d_latent)

#         self.ts_blocks = nn.ModuleList([_PerceiverBlock(d_latent, n_heads, dropout) for _ in range(n_layers)])
#         self.img_blocks = nn.ModuleList([_PerceiverBlock(d_latent, n_heads, dropout) for _ in range(n_layers)])

#         self.norm_out = nn.LayerNorm(d_latent)

#     def forward(self, ts_tokens: torch.Tensor, img_kv: torch.Tensor) -> torch.Tensor:
#         B = ts_tokens.size(0)

#         ts_kv = self.ts_proj(ts_tokens)                        # [B, 24+1, D]

#         latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
#         for ts_blk, img_blk in zip(self.ts_blocks, self.img_blocks):
#             # latents = ts_blk(latents, ts_kv)     
#             # latents = img_blk(latents, img_kv) # img_kv는 이미 projected된 상태

#             # 이번 실험에서는 이미지부터 보고, 거기에 맞는 시계열 정보를 추출하는 형태로 변경해볼 것.
#             latents = img_blk(latents, img_kv) # img_kv는 이미 projected된 상태
#             latents = ts_blk(latents, ts_kv)     

#         latents = self.norm_out(latents)
#         return latents.mean(dim=1)                             # [B, D] (Global Average Pooling)

# [4] Pathology Perceiver
# class PathologyPerceiver(nn.Module):
#     """4개(또는 K개) pathology query 기반 이미지-시계열 fusion.

#     구조:
#         Stage 0: query bank [K, d] (Edema, Cardiomegaly, Effusion, Pneumonia ...)
#         Stage 1: queries × image patches (cross-attn) → K image tokens
#         Stage 2: query self-attn (pathology 간 정보교류, image-only)
#         Stage 3: stage2 tokens × DuETT TS tokens (cross-attn) → K multimodal tokens
#         Stage 4: query self-attn (pathology 간 정보교류, multimodal)

#     Aux logits:
#         stage2_logits: image-only per-pathology 예측 [B, K]
#         stage4_logits: multimodal per-pathology 예측 [B, K]

#     최종 main task = stage4_logits[:, 0] (index 0은 Edema로 예약).
#     """

#     def __init__(self,
#                  n_pathologies: int,
#                  d_ts: int,
#                  d_latent: int = 256,
#                  n_heads: int = 4,
#                  dropout: float = 0.1,
#                  head_hidden: int = 64,
#                  head_dropout: float = 0.1):
#         super().__init__()
#         self.n_pathologies = n_pathologies
#         self.d_latent = d_latent

#         # Pathology query bank: 각 query가 하나의 라벨 역할
#         self.queries = nn.Parameter(torch.randn(n_pathologies, d_latent) * 0.02)

#         # DuETT tokens → d_latent 로 projection (image쪽은 TeacherModel.img_proj가 처리하고 넘어옴)
#         self.ts_proj = nn.Linear(d_ts, d_latent)

#         # Cross / self attention blocks (self-attn은 _PerceiverBlock(x, x)로 활용했는데 그냥 향후에 self_attn으로 바꿔도 될 듯함)
#         self.img_cross = _PerceiverBlock(d_latent, n_heads, dropout)
#         self.stage2_self = _PerceiverBlock(d_latent, n_heads, dropout)
#         self.ts_cross = _PerceiverBlock(d_latent, n_heads, dropout)
#         self.stage4_self = _PerceiverBlock(d_latent, n_heads, dropout)

#         # Per-pathology heads — pathology마다 별도 MLP (label 특화 학습 허용)
#         def _mk_head():
#             return nn.Sequential(
#                 nn.Linear(d_latent, head_hidden), nn.GELU(), nn.Dropout(head_dropout),
#                 nn.Linear(head_hidden, 1),
#             )
#         self.stage2_heads = nn.ModuleList([_mk_head() for _ in range(n_pathologies)])
#         self.stage4_heads = nn.ModuleList([_mk_head() for _ in range(n_pathologies)])

#     def forward(self, ts_tokens: torch.Tensor, img_patches_proj: torch.Tensor,
#                 return_attn: bool = False, ts_ablation: str="full") -> dict:
#         B = ts_tokens.size(0)
#         queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, d]
#         if ts_ablation == "full":
#             ts_selected = ts_tokens                            # [B, T+1, d_ts]
#         elif ts_ablation == "hourly_only":
#             ts_selected = ts_tokens[:, :-1, :]                 # [B, T, d_ts]
#         elif ts_ablation == "rep_only":
#             ts_selected = ts_tokens[:, -1:, :]                 # [B, 1, d_ts]
#         else:
#             raise ValueError(
#                 f"unknown ts_ablation={ts_ablation!r}; "
#                 "expected one of {'full', 'hourly_only', 'rep_only'}"
#             )

#         ts_kv = self.ts_proj(ts_selected)                        # [B, T+1, d]

#         img_attn = None
#         ts_attn = None

#         # Stage 1: queries attend image patches → K image tokens
#         if return_attn:
#             img_tokens, img_attn = self.img_cross(queries, img_patches_proj, return_attn=True)
#         else:
#             img_tokens = self.img_cross(queries, img_patches_proj)

#         # Stage 2: pathology 간 self-attn (image-only) — Edema↔Cardiomegaly 등 정보교류
#         stage2_tokens = self.stage2_self(img_tokens, img_tokens)

#         # Stage 3: 각 pathology token이 DuETT TS 토큰을 cross-attend
#         if return_attn:
#             mm_tokens, ts_attn = self.ts_cross(stage2_tokens, ts_kv, return_attn=True)
#         else:
#             mm_tokens = self.ts_cross(stage2_tokens, ts_kv)

#         # Stage 4: pathology 간 self-attn (multimodal) — TS 정보 반영 후 재교류
#         stage4_tokens = self.stage4_self(mm_tokens, mm_tokens) # 이건 꼭 필요할까에 대한 고민이 있음 -> 주석 처리하고 실험해볼 것!

#         # Per-pathology heads → [B, K]
#         stage2_logits = torch.stack(
#             [h(stage2_tokens[:, i]).squeeze(-1) for i, h in enumerate(self.stage2_heads)],
#             dim=1)
#         stage4_logits = torch.stack(
#             [h(stage4_tokens[:, i]).squeeze(-1) for i, h in enumerate(self.stage4_heads)],
#             dim=1)

#         out = {
#             "stage2_logits": stage2_logits,
#             "stage4_logits": stage4_logits,
#             "stage2_tokens": stage2_tokens,
#             "stage4_tokens": stage4_tokens,
#         }
#         if return_attn:
#             out["img_attn"] = img_attn   # [B, K, N]
#             out["ts_attn"] = ts_attn     # [B, K, T+1]
#         return out


# [4] PathologyPerceiver 모달리티별 분리
class PatchDualPathologyPerceiver(nn.Module):
    def __init__(
            self,
            n_pathologies: int,
            d_ts: int,
            d_latent: int = 256,
            n_heads: int = 4,
            dropout: float = 0.1,
            head_hidden: int = 64,
            head_dropout: float = 0.1
        ):
        super().__init__()
        self.n_pathologies = n_pathologies
        self.d_latent = d_latent
        
        # self.image_queries    = nn.Parameter(torch.randn(n_pathologies, d_latent) * 0.02)
        # self.temporal_queries = nn.Parameter(torch.randn(n_pathologies, d_latent) * 0.02)

        self.pathology_queries = nn.Parameter(torch.randn(n_pathologies, d_latent) * 0.02)

        self.ts_proj = nn.Linear(d_ts, d_latent)
        self.img_cross = _PerceiverBlock(d_latent, n_heads, dropout)
        self.img_self  = _PerceiverBlock(d_latent, n_heads, dropout)
        self.ts_cross  = _PerceiverBlock(d_latent, n_heads, dropout)
        self.ts_self   = _PerceiverBlock(d_latent, n_heads, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(2 * d_latent, 4 * d_latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_latent, d_latent),
        )
        def _mk_head(): # logit 출력기
            return nn.Sequential(
                nn.Linear(d_latent, head_hidden), nn.GELU(), nn.Dropout(head_dropout),
                nn.Linear(head_hidden, 1),
            )

        # pathology별 heads
        # self.image_heads    = nn.ModuleList([_mk_head() for _ in range(n_pathologies)])
        # self.temporal_heads = nn.ModuleList([_mk_head() for _ in range(n_pathologies)])
        # self.fusion_heads   = nn.ModuleList([_mk_head() for _ in range(n_pathologies)])

        # 모달리티별 공통된 헤드로 질병을 분류하도록 테스트함.
        self.image_head = _mk_head()
        self.temporal_head = _mk_head()
        self.fusion_head = _mk_head()

        # Prevalence 차이를 허용하기 위해 
        self.image_label_bias = nn.Parameter(torch.zeros(n_pathologies))
        self.temporal_label_bias = nn.Parameter(torch.zeros(n_pathologies))
        self.fusion_label_bias = nn.Parameter(torch.zeros(n_pathologies))

    def forward(self, ts_tokens, img_patches_proj, return_attn=False, ts_ablation="hourly_only"):
        B = ts_tokens.size(0)
        # img_q = self.image_queries.unsqueeze(0).expand(B, -1, -1)
        img_q = self.pathology_queries.unsqueeze(0).expand(B, -1, -1)

        I, img_attn = (self.img_cross(img_q, img_patches_proj, return_attn=True)
                       if return_attn else (self.img_cross(img_q, img_patches_proj), None))
        I = self.img_self(I, I)

        ##########################################################################################
        if ts_ablation == "full":         ts_selected = ts_tokens
        elif ts_ablation == "hourly_only": ts_selected = ts_tokens[:, :-1, :]
        elif ts_ablation == "rep_only":    ts_selected = ts_tokens[:, -1:, :]
        else: raise ValueError(ts_ablation)
        ##########################################################################################

        ts_kv = self.ts_proj(ts_selected)
        # ts_q = self.temporal_queries.unsqueeze(0).expand(B, -1, -1)
        ts_q = self.pathology_queries.unsqueeze(0).expand(B, -1, -1)

        T_tok, ts_attn = (self.ts_cross(ts_q, ts_kv, return_attn=True)
                          if return_attn else (self.ts_cross(ts_q, ts_kv), None))
        T_tok = self.ts_self(T_tok, T_tok)
        F_tok = self.fusion(torch.cat([I, T_tok], dim=-1))
        # img_logits    = torch.stack([h(I[:, k]).squeeze(-1)     for k, h in enumerate(self.image_heads)],    dim=1)
        # ts_logits     = torch.stack([h(T_tok[:, k]).squeeze(-1) for k, h in enumerate(self.temporal_heads)], dim=1)
        # fusion_logits = torch.stack([h(F_tok[:, k]).squeeze(-1) for k, h in enumerate(self.fusion_heads)],   dim=1)

        # [B, K, d] → [B, K, 1] → [B, K]
        img_logits    = self.image_head(I).squeeze(-1) + self.image_label_bias.unsqueeze(0) # Prevalence Bias
        ts_logits     = self.temporal_head(T_tok).squeeze(-1) + self.temporal_label_bias.unsqueeze(0) 
        fusion_logits = self.fusion_head(F_tok).squeeze(-1) + self.fusion_label_bias.unsqueeze(0) 

        out = {"img_logits": img_logits, "ts_logits": ts_logits, "fusion_logits": fusion_logits,
               "img_tokens": I, "ts_tokens": T_tok, "fusion_tokens": F_tok}
        if return_attn: out["img_attn"] = img_attn; out["ts_attn"] = ts_attn
        return out


# ─── NEW: Pathology-wise residual fusion ─────────────────────────────────────
class DualPathologyPerceiver(nn.Module):
    """Image branch = frozen pretrained CXR head (TeacherModel에서 계산되어 4개 logit으로 주입).
    Temporal branch = pathology query × ts_tokens (cross-attn) + self-attn → per-pathology 벡터.
    Fusion = pathology별 residual: fusion_logit[k] = img_logit[k] + residual_head_k(T[:, k]).

    240k CXR로 pretrain된 image branch에 temporal이 "보정치"만 더하는 구조.
    """

    def __init__(self,
                 n_pathologies: int,
                 d_ts: int,
                 d_latent: int = 256,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 head_hidden: int = 64,
                 head_dropout: float = 0.1):
        super().__init__()
        self.n_pathologies = n_pathologies
        self.d_latent = d_latent

        self.temporal_queries = nn.Parameter(torch.randn(n_pathologies, d_latent) * 0.02)
        self.ts_proj = nn.Linear(d_ts, d_latent)
        self.ts_cross = _PerceiverBlock(d_latent, n_heads, dropout)
        self.ts_self  = _PerceiverBlock(d_latent, n_heads, dropout)

        def _mk_head():
            return nn.Sequential(
                nn.Linear(d_latent, head_hidden), nn.GELU(), nn.Dropout(head_dropout),
                nn.Linear(head_hidden, 1),
            )
        # ts_logits (aux, TS-only 예측) 용
        self.temporal_heads = nn.ModuleList([_mk_head() for _ in range(n_pathologies)])
        
        # EdemaResidualMLP, CardiomegalyResidualMLP, EffusionResidualMLP 
        self.residual_heads = nn.ModuleList([_mk_head() for _ in range(n_pathologies)])

    def forward(self, ts_tokens: torch.Tensor, img_logits: torch.Tensor,
                return_attn: bool = False, ts_ablation: str = "hourly_only") -> dict:
        """
        Args:
            ts_tokens:  [B, T+1, d_ts]  DuETT tokens (마지막이 [REP])
            img_logits: [B, K]           pretrained CXR head가 뽑은 K개 pathology logit (frozen)
        """
        B = ts_tokens.size(0)

        if ts_ablation == "full":
            ts_selected = ts_tokens
        elif ts_ablation == "hourly_only":
            ts_selected = ts_tokens[:, :-1, :]
        elif ts_ablation == "rep_only":
            ts_selected = ts_tokens[:, -1:, :]
        else:
            raise ValueError(
                f"unknown ts_ablation={ts_ablation!r}; "
                "expected one of {'full', 'hourly_only', 'rep_only'}")
        ts_kv = self.ts_proj(ts_selected)                          # [B, T*, d]

        ts_q = self.temporal_queries.unsqueeze(0).expand(B, -1, -1)
        ts_attn = None
        if return_attn:
            T_tok, ts_attn = self.ts_cross(ts_q, ts_kv, return_attn=True)
        else:
            T_tok = self.ts_cross(ts_q, ts_kv)
        T_tok = self.ts_self(T_tok, T_tok)                         # [B, K, d]

        ts_logits = torch.stack(
            [h(T_tok[:, k]).squeeze(-1) for k, h in enumerate(self.temporal_heads)], dim=1)
        residuals = torch.stack(
            [h(T_tok[:, k]).squeeze(-1) for k, h in enumerate(self.residual_heads)], dim=1)
        fusion_logits = img_logits + residuals                     # [B, K]
        # fusion_logits = img_logits                     # [B, K]


        out = {
            "img_logits":    img_logits,       # [B, K]  (passthrough, no grad)
            "ts_logits":     ts_logits,        # [B, K]  aux TS-only
            "fusion_logits": fusion_logits,    # [B, K]  main
            "ts_tokens":     T_tok,            # [B, K, d]
            "residuals":     residuals,        # [B, K]  진단용
        }
        if return_attn:
            out["ts_attn"] = ts_attn           # [B, K, T*]
        return out

##########################################################################################
##########################################################################################
class _PerceiverBlock(nn.Module):
    _DEBUG_NORMS: bool = False

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 4, d), nn.Dropout(dropout),
        )

    def forward(self, latents, kv, return_attn: bool = False):
        q = self.norm_q(latents)
        k = self.norm_kv(kv)
        a, attn_w = self.attn(q, k, k, need_weights=return_attn, average_attn_weights=True)

        if _PerceiverBlock._DEBUG_NORMS:
            print(f"latents norm={latents.norm(dim=-1).mean():.3f}, "
                  f"attn_out norm={a.norm(dim=-1).mean():.3f}, "
                  f"ratio={(a.norm(dim=-1) / latents.norm(dim=-1).clamp(min=1e-6)).mean():.4f}")

        # Residual Connection
        latents = latents + a
        latents = latents + self.ff(self.norm_ff(latents))
        if return_attn:
            return latents, attn_w
        return latents




# =============================================================================
# Teacher / Student
# =============================================================================
class TeacherModel(nn.Module):
    def __init__(
        self,
        duett_backbone: DuettFeatureExtractor,
        cxr_encoder: CXREncoder,
        perceiver,  # TemporalPerceiver / PathologyPerceiver / DualPathologyPerceiver
        head_hidden: int = 128,
        head_dropout: float = 0.1,
        cxr_return_patches: bool = True,
        d_img: int = 768,
        use_aux_cxr: bool = True,
        aux_head_hidden: int = 128,
        pathology_mode: bool = False,
        dual_pathology_mode: bool = False,
        patch_dual_pathology_mode: bool = False,
        pretrained_cxr_head_ckpt: Optional[str] = None,
        pathology_labels: Optional[tuple] = None,
    ):
        super().__init__()
        n_modes = sum([pathology_mode, dual_pathology_mode, patch_dual_pathology_mode])
        if n_modes > 1:
            raise ValueError("pathology_mode / dual_pathology_mode / patch_dual_pathology_mode "
                             "중 최대 하나만 True 이어야 함.")

        self.duett = duett_backbone
        self.cxr = cxr_encoder
        self.perceiver = perceiver
        self.cxr_return_patches = cxr_return_patches
        self.pathology_mode = pathology_mode
        self.dual_pathology_mode = dual_pathology_mode
        self.patch_dual_pathology_mode = patch_dual_pathology_mode

        d = perceiver.d_latent
        # img_proj를 TeacherModel로 옮겨옴 → main/aux head가 공유
        self.img_proj = nn.Linear(d_img, d)

        if pathology_mode or dual_pathology_mode or patch_dual_pathology_mode:
            # Query-based perceiver 모드: 별도 main head/aux CXR head 없음.
            # main_logit은 perceiver 출력의 index 0 (Edema query)에서 뽑음.
            self.head = None
            self.aux_cxr_head = None
            self.use_aux_cxr = False
        else:
            self.head = nn.Sequential(
                nn.Linear(d, head_hidden), nn.GELU(), nn.Dropout(head_dropout),
                nn.Linear(head_hidden, 1),
            )
            self.use_aux_cxr = use_aux_cxr
            if use_aux_cxr:
                self.aux_cxr_head = nn.Sequential(
                    nn.Linear(d, aux_head_hidden), nn.GELU(), nn.Dropout(head_dropout),
                    nn.Linear(aux_head_hidden, 1),
                )

        # ─── NEW dual: frozen pretrained CXR head + keep_indices ────────────
        if dual_pathology_mode:
            if pretrained_cxr_head_ckpt is None or pathology_labels is None:
                raise ValueError("dual_pathology_mode requires pretrained_cxr_head_ckpt AND pathology_labels")
            state = torch.load(pretrained_cxr_head_ckpt, map_location="cpu", weights_only=False)
            num_pretrained = int(state["num_classes"])
            pretrained_labels = list(state["label_cols"])
            missing = [lbl for lbl in pathology_labels if lbl not in pretrained_labels]
            if missing:
                raise ValueError(f"pretrained CXR head에 없는 pathology_labels: {missing}. "
                                 f"pretrained labels: {pretrained_labels}")
            keep_idx = [pretrained_labels.index(lbl) for lbl in pathology_labels]

            self.pretrained_cxr_head = nn.Linear(d_img, num_pretrained)
            # ckpt classifier_state_dict의 key는 'Sequential[1] = Linear'이므로 1.weight/1.bias
            clf_sd = state["classifier_state_dict"]
            self.pretrained_cxr_head.load_state_dict({
                "weight": clf_sd["1.weight"],
                "bias":   clf_sd["1.bias"],
            })
            for p in self.pretrained_cxr_head.parameters():
                p.requires_grad = False
            self.register_buffer("cxr_head_keep_idx", torch.tensor(keep_idx, dtype=torch.long))
            print(f"[keep_idx debug] pathology_labels={pathology_labels}, "
                f"pretrained_labels={pretrained_labels}, keep_idx={keep_idx.tolist() if hasattr(keep_idx,'tolist') else keep_idx}")



    def forward(self,
                x_ts_list, x_static_list, bin_ends_list,
                pixel_values: torch.Tensor,
                batch_size: Optional[int] = None,
                return_attn: bool = False):
        """Returns:
            - dual_pathology_mode=True : dict(main_logit, img_logits, ts_logits, fusion_logits[, tokens, attn])
            - pathology_mode=True      : dict(main_logit, stage2_logits, stage4_logits[, tokens, attn])
            - use_aux_cxr=True         : tuple(main_logit, aux_logit)  (legacy)
            - else                     : Tensor main_logit             (legacy)

        return_attn=True는 pathology / dual_pathology 모드 전용 (viz용 inference).
        """

        if batch_size is None:
            batch_size = pixel_values.shape[0]

        x = (x_ts_list, x_static_list, bin_ends_list)
        duett_in = self.duett.feats_to_input(x, batch_size)
        ts_tokens, ts_grid = self.duett.encode(duett_in)   # [B, T+1, D]

        # ts_tokens 말고 grid를 쓰는 방식을 추후 고안
        dynamic_grid = ts_grid[:, :-1, :-1, :] # [REP], static token 제거 ([B, 24, V, D])

        # grid cross를 만들어야 할 듯? (Q: pathology queries, K,V: grid)
        # -> [B, K, 24V]


        if self.patch_dual_pathology_mode:
            img_cls, img_patches = self.cxr(pixel_values)
            img_patches_proj = self.img_proj(img_patches)
            out = self.perceiver(ts_tokens, img_patches_proj, return_attn=return_attn)            

            # 7 by 7 spatial pooling 
            # h = w = int(N ** 0.5)                           # 37 → 37×37
            # img_spatial = img_patches.transpose(1, 2).reshape(B, D, h, w)
            # img_pooled  = F.adaptive_avg_pool2d(img_spatial, (7, 7))
            # img_tokens  = img_pooled.flatten(2).transpose(1, 2)         # [B, 49, 768]
            # img_patches_proj = self.img_proj(img_tokens)                # [B, 49, D]
            # out = self.perceiver(ts_tokens, img_patches_proj, return_attn=return_attn)
            
            result = {
                "main_logit":    out["fusion_logits"][:, 0],
                "img_logits":    out["img_logits"],
                "ts_logits":     out["ts_logits"],
                "fusion_logits": out["fusion_logits"],
            }
            if return_attn:
                result["img_tokens"]    = out["img_tokens"]
                result["ts_tokens"]     = out["ts_tokens"]
                result["fusion_tokens"] = out["fusion_tokens"]
                result["img_attn"]      = out["img_attn"]
                result["ts_attn"]       = out["ts_attn"]
            return result

        # ─── NEW dual: CLS-only + frozen pretrained head + residual fusion ───
        if self.dual_pathology_mode:
            cls = self.cxr(pixel_values)
            if isinstance(cls, tuple):    # cxr_return_patches=True 설정이라도 CLS만 사용
                cls = cls[0]
            with torch.no_grad():
                pretrained_logits = self.pretrained_cxr_head(cls)             # [B, C_pretrain]
            img_logits = pretrained_logits[:, self.cxr_head_keep_idx]         # [B, K]
            out = self.perceiver(ts_tokens, img_logits, return_attn=return_attn)
            result = {
                "main_logit":    out["fusion_logits"][:, 0],
                "img_logits":    out["img_logits"],
                "ts_logits":     out["ts_logits"],
                "fusion_logits": out["fusion_logits"],
            }
            if return_attn:
                result["ts_tokens"] = out["ts_tokens"]
                result["ts_attn"]   = out["ts_attn"]
                result["residuals"] = out["residuals"]
            return result

        if self.cxr_return_patches:
            img_cls, img_patches = self.cxr(pixel_values)

            B, N, D = img_patches.shape
            h = w = int(N ** 0.5)  # 37

            # patch → 2D spatial → adaptive avg pool → token sequence
            img_spatial = img_patches.transpose(1, 2).reshape(B, D, h, w)    # [B, 768, 37, 37]
            img_pooled  = F.adaptive_avg_pool2d(img_spatial, (7, 7))         # [B, 768, 7, 7]
            img_tokens  = img_pooled.flatten(2).transpose(1, 2)              # [B, 49, 768]

            if self.pathology_mode:
                # 4 pathology queries × 7×7=49 patches (CLS 미사용)
                img_patches_proj = self.img_proj(img_tokens)                 # [B, 49, D]
                out = self.perceiver(ts_tokens, img_patches_proj, return_attn=return_attn)
                result = {
                    "main_logit":    out["stage4_logits"][:, 0],  # Edema (index 0)
                    "stage2_logits": out["stage2_logits"],        # [B, K]
                    "stage4_logits": out["stage4_logits"],        # [B, K]
                }
                if return_attn:
                    result["stage2_tokens"] = out["stage2_tokens"]
                    result["stage4_tokens"] = out["stage4_tokens"]
                    result["img_attn"] = out["img_attn"]           # [B, K, 49]
                    result["ts_attn"]  = out["ts_attn"]            # [B, K, T+1]
                return result

            img_kv = torch.cat([img_cls.unsqueeze(1), img_tokens], dim=1)    # [B, 50, 768]

            img_kv_proj = self.img_proj(img_kv)                          # [B, 50, D]

            fused = self.perceiver(ts_tokens, img_kv_proj)

            main_logit = self.head(fused).squeeze(-1)                        # [B]

            if self.use_aux_cxr:
                cxr_summary = img_kv_proj[:, 0]                              # [B, D]  CLS 위치
                aux_logit = self.aux_cxr_head(cxr_summary).squeeze(-1)       # [B]
                return main_logit, aux_logit
            return main_logit

        else: 
            img_cls = self.cxr(pixel_values)
            fused = self.perceiver(ts_tokens, img_cls)  # [B, d_latent]
            logit = self.head(fused).squeeze(-1)      # [B]
            return logit

##########################################################################################
##########################################################################################
# Student
class StudentModel(nn.Module):
    """DuETT(TS) + linear head. Backbone is trained end-to-end."""

    def __init__(
        self,
        duett_backbone: DuettFeatureExtractor,
        pool: str = "mean",
        head_hidden: int = 128,
        head_dropout: float = 0.1
        ):
        super().__init__()
        self.duett = duett_backbone
        self.pool = pool
        d_rep = duett_backbone.d_representation
        self.head = nn.Sequential(
            nn.Linear(d_rep, head_hidden), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x_ts_list, x_static_list, bin_ends_list,
                batch_size: Optional[int] = None) -> torch.Tensor:
        if batch_size is None:
            batch_size = len(x_ts_list)
        x = (x_ts_list, x_static_list, bin_ends_list)
        duett_in = self.duett.feats_to_input(x, batch_size)
        ts_tokens = self.duett.encode(duett_in)   # [B, T+1, d_rep]
        if self.pool == "rep_token":
            feat = ts_tokens[:, -1, :]
        elif self.pool == "mean":
            feat = ts_tokens[:, :-1, :].mean(dim=1) # 맨 마지막 REP token 제외하고 mean pooling
        else:
            raise ValueError(f"unknown pool: {self.pool}")
        logit = self.head(feat).squeeze(-1)
        return logit
