import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

# from models.encoder import TransformerTSEncoder, TSMixerEncoder, PatchTSTTSEncoder
from models.encoder import PatchTSTTSEncoder
from utils.utils import timer


def slot_to_patch_idx(t, patch_len, stride, num_patch):
    """
    Map 30-min slot index t (0..T-1) to the nearest patch index (0..num_patch-1).

    Patch i covers slots [i*stride, i*stride + patch_len).
    Its center is i*stride + patch_len/2. The patch index closest to slot t is:
        round((t - patch_len/2) / stride)
    Clamped to [0, num_patch-1] for boundary slots.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    p = torch.round((t.float() - patch_len / 2.0) / stride).long()
    return torch.clamp(p, min=0, max=num_patch - 1)


# ────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers: 패치/어텐션 collapse 측정
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _inter_patch_cosine(x, mask=None):
    """
    패치(또는 슬롯) 임베딩 간 평균 pairwise cosine similarity (off-diagonal).
    1.0에 가까울수록 패치가 서로 비슷 (collapse), 0에 가까울수록 다양.

    x: [B, P, D]
    mask: [B, P] bool, True = padded (계산에서 제외)

    Batch row가 전부 padded이거나 NaN인 경우 그 row를 제외하고 평균.
    """
    if x is None or x.size(1) < 2:
        return float('nan')
    xn = F.normalize(x.float(), p=2, dim=-1)
    sim = xn @ xn.transpose(-2, -1)             # [B, P, P]
    B, P, _ = sim.shape
    if mask is not None:
        valid = ~mask
    else:
        valid = torch.ones(B, P, dtype=torch.bool, device=x.device)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    eye = torch.eye(P, device=x.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    pair_valid = pair_valid & ~eye
    n_pairs = pair_valid.sum(dim=(1, 2)).float()
    sim_masked = torch.nan_to_num(sim.float() * pair_valid.float(), nan=0.0)
    per_b = sim_masked.sum(dim=(1, 2)) / n_pairs.clamp(min=1)
    # n_pairs==0 인 row (valid pair 없음) 와 원래 NaN row 제외
    row_valid = (n_pairs > 0) & ~torch.isnan(per_b)
    if not row_valid.any():
        return float('nan')
    return float(per_b[row_valid].mean().item())


@torch.no_grad()
def _attn_inter_patch_cosine(attn_weights, query_mask=None):
    """
    각 query(패치)별 attention 분포 간 평균 cosine.
    1.0에 가까우면 모든 패치가 동일한 KV에 attend (collapse),
    낮으면 패치마다 다른 KV에 attend (정상).

    attn_weights: [B, T_q, T_k] (softmax 후)
    query_mask: [B, T_q] bool, True = padded query

    Key가 전부 padded인 row는 attention이 NaN (softmax(-inf)=NaN). 그 row는 제외하고 평균.
    """
    if attn_weights is None or attn_weights.size(1) < 2:
        return float('nan')
    a = attn_weights.float()
    # NaN 분포(전체 key가 padded인 row) → 정규화 전에 0으로 치환하고 row_valid 별도 추적
    nan_row = torch.isnan(a).any(dim=-1)           # [B, T_q]
    a_clean = torch.nan_to_num(a, nan=0.0)
    an = F.normalize(a_clean, p=2, dim=-1)
    cos = an @ an.transpose(-2, -1)                 # [B, T_q, T_q]
    B, P, _ = cos.shape
    if query_mask is not None:
        valid = ~query_mask
    else:
        valid = torch.ones(B, P, dtype=torch.bool, device=a.device)
    # nan이었던 query는 valid에서 제외
    valid = valid & ~nan_row
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    eye = torch.eye(P, device=a.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    pair_valid = pair_valid & ~eye
    n_pairs = pair_valid.sum(dim=(1, 2)).float()
    cos_masked = cos * pair_valid.float()
    per_b = cos_masked.sum(dim=(1, 2)) / n_pairs.clamp(min=1)
    row_valid = (n_pairs > 0) & ~torch.isnan(per_b)
    if not row_valid.any():
        return float('nan')
    return float(per_b[row_valid].mean().item())


# ────────────────────────────────────────────────────────────────────────────
# Temporal Causal Mask
# ────────────────────────────────────────────────────────────────────────────
def create_causal_mask(q_time, k_time, key_padding_mask=None):
    """
    임상 인과성: TS 패치 q는 시간상 자기 시점(q_time) 이전+동시간(<=)의 KV만 attend.

    Args:
      q_time: [B, T_q] float — query (TS 패치)의 slot 시간
      k_time: [B, T_k] float — key (img/text token)의 slot 시간
      key_padding_mask: [B, T_k] bool — True = padding (legacy 마스크)

    Returns:
      block:           [B, T_q, T_k] bool — True = 차단 (causal 위반 OR padding)
      fully_blocked_q: [B, T_q] bool — True = 이 query는 유효 KV가 0개 (empty row)

    Safety: fully_blocked query에 대해 position 0을 강제 unblock (NaN 방지).
            이런 query의 출력은 호출자가 zero-out 해야 함.
    """
    # 시간 비교
    time_diff = q_time.unsqueeze(2) - k_time.unsqueeze(1)   # [B, T_q, T_k]
    causal_allow = time_diff >= 0                             # True = 시간상 허용

    block = ~causal_allow                                     # True = causal 위반

    # padding과 결합
    if key_padding_mask is not None:
        pad_block = key_padding_mask.unsqueeze(1).expand_as(block)
        block = block | pad_block

    # Fully-blocked queries 검출
    fully_blocked_q = block.all(dim=-1)                       # [B, T_q]

    # NaN 방지: fully-blocked query는 position 0 unblock
    if fully_blocked_q.any():
        first_pos = torch.zeros_like(block)
        first_pos[..., 0] = True
        first_pos_unblock = first_pos & fully_blocked_q.unsqueeze(-1)
        block = block & ~first_pos_unblock

    return block, fully_blocked_q


class ResidualPreProj(nn.Module):
    """
    Residual pre-projection: skip connection으로 raw 다양성 보존 + MLP로 task-relevant feature 학습.
    Projection collapse 방지에 효과적 (skip이 패치별 차이를 직접 출력으로 전달).

    구조:
      out = LayerNorm( skip_proj(x) + mlp(x) )
        - mlp: Linear(in→2*out) → GELU → Dropout → Linear(2*out→out)
        - skip_proj: Linear(in→out) (in == out 이면 Identity)
        - LayerNorm: magnitude 정규화 (패치 간 차이는 보존)
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.skip_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        out = self.skip_proj(x) + self.mlp(x)
        return self.norm(out)



########################################################################################################################
########################################################################################################################

class MultiModalEncoder(nn.Module):
    """
    - CXR/Text는 raw 입력이 아닌 사전 계산된 임베딩을 받아 projection만 학습.
    - Segment 임베딩을 K/V로 사용한 cross-attention으로 CXR 표현을 강화.
        - lung, heart를 사용한 별도의 태스크 설계도 작성하자.
    - (Future) lesion_emb를 K/V에 추가 가능
    """
    def __init__(self, args, disable_cxr=False, disable_txt=False, disable_prompt=False):
        super().__init__()

        self.disable_cxr = disable_cxr
        self.disable_txt = disable_txt
        self.disable_prompt = disable_prompt

        self.img_emb_dim = args.img_emb_dim
        self.text_emb_dim = args.text_emb_dim

        # ==================== Modality-Specific Encoders ====================
        self.ts_encoder = PatchTSTTSEncoder(
            input_size=args.ts_encoder_input_size,
            hidden_size=args.ts_encoder_hidden_size,  # var_pool 제거 후 미사용 (호환 인자)
            window_size=args.window_size,
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.patchtst_d_model,
            n_heads=args.patchtst_n_heads,
            n_layers=args.patchtst_n_layers,
            d_ff=args.patchtst_d_model * 4,
            dropout=0.1,
            shared_embedding=True,
            pretrained_path=args.patchtst_pretrained_path,
            freeze_backbone=bool(args.patchtst_freeze),
            unfreeze_last_n=args.patchtst_unfreeze_last_n,
            var_pool_type=args.var_pool_type,             # 미사용
        )
        self.ts_hidden_size = args.ts_encoder_input_size * args.patchtst_d_model
        self.patch_len  = args.patch_len
        self.stride     = args.stride
        self.num_patch  = self.ts_encoder.num_patch

        self.num_cxr_tokens = 2
        self.cxr_token_type_emb = nn.Embedding(self.num_cxr_tokens, self.img_emb_dim)
        nn.init.trunc_normal_(self.cxr_token_type_emb.weight, std=0.02)

        # ==================== Shared pre-projection (Fusion + Alignment) ====================
        self.align_d_model = args.align_d_model

        # Residual pre-projection (projection collapse 방지: skip이 raw 다양성 보존)
        self.ts_pre_proj   = ResidualPreProj(self.ts_hidden_size, self.align_d_model)
        self.img_pre_proj  = ResidualPreProj(self.img_emb_dim,    self.align_d_model)
        self.text_pre_proj = ResidualPreProj(self.text_emb_dim,   self.align_d_model)

        self.ts_centric_fusion = TimeSeriesCentricCrossAttention_v7(
            args=args,
            d_model=self.align_d_model,
            num_heads=8,
            ts_input_dim=self.align_d_model,
            img_input_dim=self.align_d_model,
            txt_input_dim=self.align_d_model,
            disable_cxr=disable_cxr,
            disable_txt=disable_txt,
        )

        # ==================== Text Encoder (PubMedBERT, frozen) ====================
        if not disable_txt:
            self.text_encoder = AutoModel.from_pretrained(
                args.text_model_path,
                output_attentions=True,
            )
            # 반드시 freeze: requires_grad=False (optimizer에서 자동 제외) + eval mode (dropout 비활성화)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()
            self.text_max_tokens = args.text_max_tokens
            # Attention-based masking (학습 중만), CLS-attention top-K 토큰을 bank에서 제외)
            self.text_attn_mask_prob = args.text_attn_mask_prob
            print(f"[Text Encoder] {args.text_model_path} loaded (frozen, output_attentions=True), "
                  f"max_tokens={self.text_max_tokens}, "
                  f"attn_mask_prob={self.text_attn_mask_prob}")
        else:
            self.text_encoder = None
            self.text_max_tokens = args.text_max_tokens
            self.text_attn_mask_prob = 0.0

        self._diag_enabled = False
        self.last_diag = {}

    def train(self, mode=True):
        """
        model.train()이 호출돼도 frozen BERT는 eval 유지 (dropout 비활성).
        """
        super().train(mode)
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def forward(self, ts_series, cxr_data, text_data, has_cxr, has_text,
                time_steps=None, ts_valid_mask=None, force_disable_txt=False):
        device = ts_series.device
        B, T, _ = ts_series.shape

        disable_txt_local = self.disable_txt or force_disable_txt

        # ================ Time-series Encoding ================
        with timer("TS Encoder", None):
            ts_embeddings, ts_kpm = self.ts_encoder(ts_series, ts_valid_mask=ts_valid_mask)
            seq_len = self.num_patch
            def _to_seq_idx(t_pos):
                return slot_to_patch_idx(t_pos, self.patch_len, self.stride, self.num_patch)

        # ================ CXR (RadDino lung + heart anatomy tokens) ================
        # Each CXR slot contributes 2 tokens (lung, heart) — both at the same time index
        # but tagged with a learnable token-type embedding so cross-attention can
        # distinguish them under shared W_K, W_V projections.
        K_cxr = self.num_cxr_tokens   # 2
        if not self.disable_cxr:
            with timer("IMG-internal Fusion", None):
                unique_img_embs = cxr_data['unique_embs']           # [N_img, 2, img_emb_dim]
                unique_indices = cxr_data['unique_indices']         # [num_positions]
                pos = cxr_data['positions']                         # [num_positions, 2]

                # [B, seq_len, K_cxr, img_emb_dim] — per-slot per-token-type slot
                img_tensor = torch.zeros(B, seq_len, K_cxr, self.img_emb_dim, device=device, dtype=ts_embeddings.dtype)
                has_img = torch.zeros(B, seq_len, device=device, dtype=torch.bool)

                if unique_img_embs.numel() > 0 and pos.numel() > 0:
                    b_pos, t_pos = pos[:, 0].long(), pos[:, 1].long()
                    p_pos = _to_seq_idx(t_pos)   # slot → patch index

                    # Scatter the [num_pos, 2, img_emb_dim] tensor into the slot.
                    scattered_img = unique_img_embs[unique_indices].to(dtype=ts_embeddings.dtype)
                    img_tensor[b_pos, p_pos] = scattered_img
                    has_img[b_pos, p_pos] = True

                # Add token-type embedding (learnable [global=N/A, lung=0, heart=1])
                type_ids = torch.arange(K_cxr, device=device)
                type_emb = self.cxr_token_type_emb(type_ids).to(dtype=ts_embeddings.dtype)  # [K_cxr, dim]
                img_tensor = img_tensor + type_emb[None, None, :, :]   # broadcast

                # Flatten anatomy axis → cross-attention sees one img sequence
                # of length seq_len * K_cxr.
                img_embeddings = img_tensor.reshape(B, seq_len * K_cxr, self.img_emb_dim)
                has_img_flat   = has_img.unsqueeze(-1).expand(-1, -1, K_cxr).reshape(B, seq_len * K_cxr)
        else:
            img_embeddings = torch.zeros(B, seq_len * K_cxr, self.img_emb_dim, device=device, dtype=ts_embeddings.dtype)
            has_img = torch.zeros(B, seq_len, device=device, dtype=torch.bool)
            has_img_flat = torch.zeros(B, seq_len * K_cxr, device=device, dtype=torch.bool)
            has_cxr = torch.zeros_like(has_cxr)

        # ================ Text (BERT 모델 내 forward → 토큰 시퀀스 → window text bank) ================
        if not disable_txt_local:
            with timer("Text BERT + Bank", None):
                uniq_ids  = text_data['unique_input_ids'].to(device)   # [N_uniq, 128]
                uniq_mask = text_data['unique_attn_mask'].to(device)   # [N_uniq, 128]
                pos       = text_data['positions']                     # [num_pos, 2]
                uniq_idx  = text_data['unique_indices']                # [num_pos]

                if uniq_ids.numel() > 0 and pos.numel() > 0:
                    # Attention-based masking은 학습 중만 적용 (eval에선 마스킹 안 함)
                    need_attn = self.training and self.text_attn_mask_prob > 0

                    with torch.no_grad():
                        bert_out = self.text_encoder(
                            input_ids=uniq_ids,
                            attention_mask=uniq_mask,
                            # output_attentions은 config에서 True로 설정됨 (forward param도 명시)
                            output_attentions=True,
                        )
                    unique_text_tokens = bert_out.last_hidden_state.to(dtype=ts_embeddings.dtype) # [N_uniq, 128, 768]

                    # ── Attention-based 표적 마스킹 (학습 중만) ──
                    # CLS attention 상위 K% 토큰을 downstream bank에서 제외 → teacher가
                    # 자기가 가장 중요시한 토큰들 (label leakage 토큰 포함 가능성 큼)을
                    # 사용 못 하게 강제 → text 의존도 ↓
                    # Defensive: attentions가 비어있는 경우 (드물게 모델별 quirk) 마스킹 skip
                    if need_attn and bert_out.attentions is not None and len(bert_out.attentions) > 0:
                        last_layer_attn = bert_out.attentions[-1]
                        cls_attention = last_layer_attn[:, :, 0, :].mean(dim=1)

                        # CLS와 padding token은 제거 대상에서 제외
                        cls_attention = cls_attention * uniq_mask.float()
                        cls_attention[:, 0] = 0.0

                        valid_lens = uniq_mask.sum(dim=1)
                        drop_k = (valid_lens.float() * self.text_attn_mask_prob).long()

                        # 원본 input tensor 보호: clone 후 수정
                        uniq_mask = uniq_mask.clone()
                        for i in range(uniq_ids.size(0)):
                            k = int(drop_k[i].item())
                            if k > 0:
                                _, top_idx = torch.topk(cls_attention[i], k)
                                uniq_mask[i, top_idx] = 0

                    # Window-level text bank 구성 + 토큰별 event slot 시간 (Fix A)
                    text_embeddings, has_text_tok, text_token_slots = self._build_text_kv_bank(
                        unique_text_tokens, uniq_mask, uniq_idx, pos,
                        B=B, device=device, dtype=ts_embeddings.dtype,
                    )

                else:
                    # 배치 전체에 텍스트 없음 → 최소 길이 placeholder
                    text_embeddings   = torch.zeros(B, 1, self.text_emb_dim, device=device, dtype=ts_embeddings.dtype)
                    has_text_tok      = torch.zeros(B, 1, device=device, dtype=torch.bool)
                    text_token_slots  = torch.zeros(B, 1, device=device, dtype=ts_embeddings.dtype)
        else:
            text_embeddings   = torch.zeros(B, 1, self.text_emb_dim, device=device, dtype=ts_embeddings.dtype)
            has_text_tok      = torch.zeros(B, 1, device=device, dtype=torch.bool)
            text_token_slots  = torch.zeros(B, 1, device=device, dtype=ts_embeddings.dtype)
            has_text = torch.zeros_like(has_text)

        # ================ Shared pre-projection (Fusion + Alignment) ================
        ts_emb_p   = self.ts_pre_proj(ts_embeddings)        # [B, seq_len,           align_d_model]
        img_emb_p  = self.img_pre_proj(img_embeddings)      # [B, seq_len*K_cxr,     align_d_model]
        text_emb_p = self.text_pre_proj(text_embeddings)    # [B, seq_len,           align_d_model]

        # ================ 진단: PatchTST 직후 raw TS 패치 다양성 ================
        self.last_diag = {}
        if self._diag_enabled:
            self.last_diag['ts_postpatch_inter_patch_cos'] = _inter_patch_cosine(
                ts_embeddings, mask=ts_kpm
            )
            self.last_diag['ts_proj_inter_patch_cos'] = _inter_patch_cosine(
                ts_emb_p, mask=ts_kpm
            )

        # ================ Multimodal Fusion ================
        with timer("TS-Centric Fusion", None):
            patch_center_slot = torch.arange(
                self.num_patch, device=device, dtype=ts_emb_p.dtype
            ) * self.stride + self.patch_len / 2.0       # [P]
            time_idx = patch_center_slot.unsqueeze(0).expand(B, -1)   # [B, P]

            img_time_idx = time_idx.unsqueeze(-1).expand(-1, -1, K_cxr).reshape(B, seq_len * K_cxr)

            # Fusion에 diag 플래그 전파 (진단 모드일 때만 attention weight 저장 + cosine 계산)
            self.ts_centric_fusion._diag_enabled = self._diag_enabled

            fused_kwargs = dict(
                ts_embeddings=ts_emb_p,                     # [B, seq_len, align_d_model]
                img_embeddings=img_emb_p,                   # [B, seq_len * K_cxr, align_d_model]
                text_embeddings=text_emb_p,                 # [B, L_max, align_d_model]
                text_token_slots=text_token_slots,          # [B, L_max] event slot 시간 (Fix A)
                time_indices=time_idx,
                img_time_indices=img_time_idx,
                img_key_padding_mask=~has_img_flat,
                text_key_padding_mask=~has_text_tok,
            )
            fused_kwargs['ts_key_padding_mask'] = ts_kpm

            fused_embeddings = self.ts_centric_fusion(**fused_kwargs)

            # Fusion 측 진단 결과 merge
            if self._diag_enabled:
                self.last_diag.update(self.ts_centric_fusion.last_diag)

        # ts_kpm 반환: readout이 padded 패치를 제외하고 pooling하도록
        return fused_embeddings, ts_kpm

    def _build_text_kv_bank(self, unique_text_tokens, uniq_attn_mask, uniq_indices, positions,
                            B, device, dtype):
        """
        Window-level text bank 구성 + 토큰별 event slot 시간 트래킹.

        Args:
          unique_text_tokens: [N_uniq, 128, 768]   BERT 출력
          uniq_attn_mask    : [N_uniq, 128]        valid token mask (1=valid)
          uniq_indices      : [num_pos]            positions[k]가 가리키는 unique 이벤트 index
          positions         : [num_pos, 2]         [batch_row, slot_idx] 쌍
          B                 : batch size

        Returns:
          bank        : [B, L_max, 768]  window 텍스트 토큰 concat (padding 포함)
          mask        : [B, L_max] bool  valid 토큰 마스크 (True = 유효)
          token_slots : [B, L_max] float 각 토큰이 속한 텍스트 이벤트의 슬롯 시간
                                       (패딩 위치는 0; downstream에서 mask로 제외됨)
        """
        # 1) batch row별로 (event_idx, slot_t) 모음.
        #    positions는 nonzero() 결과라 [row, col] 사전순 → 같은 row 내에서 slot 오름차순 보장.
        per_row_events = [[] for _ in range(B)]
        if positions.numel() > 0:
            b_vec  = positions[:, 0].long().tolist()
            s_vec  = positions[:, 1].long().tolist()  # window-local slot idx
            ev_vec = uniq_indices.long().tolist()
            for b, s, ev in zip(b_vec, s_vec, ev_vec):
                per_row_events[b].append((ev, s))

        # 2) row별 valid 토큰 길이
        row_valid_lens = uniq_attn_mask.sum(dim=1).long().tolist()  # [N_uniq]
        row_lens = [sum(row_valid_lens[ev] for ev, _ in events) for events in per_row_events]
        L_max = max(row_lens) if row_lens and max(row_lens) > 0 else 1

        # 3) Bank + mask + token_slots alloc
        bank        = torch.zeros(B, L_max, unique_text_tokens.size(-1), device=device, dtype=dtype)
        mask        = torch.zeros(B, L_max, device=device, dtype=torch.bool)
        token_slots = torch.zeros(B, L_max, device=device, dtype=dtype)

        # 4) 각 row를 채움
        for b, events in enumerate(per_row_events):
            offset = 0
            for ev, slot_t in events:
                vlen = row_valid_lens[ev]
                if vlen == 0:
                    continue
                bank[b, offset:offset + vlen]        = unique_text_tokens[ev, :vlen]
                mask[b, offset:offset + vlen]        = True
                token_slots[b, offset:offset + vlen] = float(slot_t)
                offset += vlen

        return bank, mask, token_slots


class MultiModalMultiTaskModel(nn.Module):
    """
    Shared encoder + shared readout LUPI 구조.
    - 단일 encoder가 deploy(text 차단) / priv(text 포함) 두 mode로 forward됨.
    - shared_readout: priv/deploy 모두 같은 GAPReadout 통과 → logit 일관성, gradient 2x.
    - Separated 구조는 priv 학습 부진으로 폐기 (롤백). 분리 시도 코드는 주석으로 보존.
    """
    def __init__(self, args, encoder=None):
        super().__init__()

        # ===== Shared encoder (단일 인코더) =====
        # encoder 인자는 외부에서 받을 수도 있고 (호환성), 안 받으면 내부 생성
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = MultiModalEncoder(
                args,
                disable_cxr=args.disable_cxr,
                disable_txt=args.disable_txt,
            )

        # ===== (참고) Encoder 분리 시도 — 폐기, 보존만 =====
        # self.deploy_encoder = MultiModalEncoder(args, disable_txt=True)
        # self.priv_encoder   = MultiModalEncoder(args, disable_txt=False)

        d = args.align_d_model

        # ===== Shared readout (단일 readout) =====
        self.shared_readout = GAPReadout(d_model=d, num_classes=1)

        # ===== Subtype auxiliary head (3-class soft CE: p_mixed/p_ncpe/p_cpe) =====
        # Applied on deploy path's pooled feature; loss masked by subtype_mask.
        self.subtype_head = nn.Linear(d, 3)

        # ===== (참고) Readout 분리 시도 — 폐기, 보존만 =====
        # self.priv_readout   = GAPReadout(d_model=d, num_classes=1)
        # self.deploy_readout = GAPReadout(d_model=d, num_classes=1)

        # 진단
        self.diag_enabled = False
        self.diag_deploy  = {}
        self.diag_priv    = {}

    def forward(self, args, ts_series, cxr_data, text_data, has_cxr, has_text,
                time_steps=None, ts_valid_mask=None, lupi_mode=True,
                disable_txt_priv=False):

        # 진단 플래그 encoder로 전파
        self.encoder._diag_enabled = self.diag_enabled

        # ============= Deploy Path (Student) — text 차단 =============
        fused_deploy, ts_kpm_deploy = self.encoder(
            ts_series, cxr_data, text_data,
            has_cxr=has_cxr, has_text=has_text,
            time_steps=time_steps, ts_valid_mask=ts_valid_mask,
            force_disable_txt=True,
        )
        if self.diag_enabled:
            self.diag_deploy = dict(self.encoder.last_diag)

        logit_deploy, feat_deploy = self.shared_readout(
            fused_deploy, ts_kpm=ts_kpm_deploy, return_feature=True
        )
        subtype_logits_deploy = self.subtype_head(feat_deploy)

        if not lupi_mode:
            if self.diag_enabled:
                self.diag_priv = {}
                self.encoder._diag_enabled = False
            return {
                'logit_priv':   None,
                'logit_deploy': logit_deploy,
                'fused_priv':   None,
                'fused_deploy': fused_deploy,
                'feat_priv':    None,
                'feat_deploy':  feat_deploy,
                'ts_kpm_priv':  None,
                'ts_kpm_deploy': ts_kpm_deploy,
                'subtype_logits_deploy': subtype_logits_deploy,
            }

        # ============= Privileged Path (Teacher) — text 포함 (or modality dropout 시 차단) =============
        fused_priv, ts_kpm_priv = self.encoder(
            ts_series, cxr_data, text_data,
            has_cxr=has_cxr, has_text=has_text,
            time_steps=time_steps, ts_valid_mask=ts_valid_mask,
            force_disable_txt=disable_txt_priv,
        )
        if self.diag_enabled:
            self.diag_priv = dict(self.encoder.last_diag)
            self.encoder._diag_enabled = False

        logit_priv, feat_priv = self.shared_readout(
            fused_priv, ts_kpm=ts_kpm_priv, return_feature=True
        )
        subtype_logits_priv = self.subtype_head(feat_priv)

        return {
            'logit_priv':   logit_priv,
            'logit_deploy': logit_deploy,
            'fused_priv':   fused_priv,
            'fused_deploy': fused_deploy,
            'feat_priv':    feat_priv,
            'feat_deploy':  feat_deploy,
            'subtype_logits_deploy': subtype_logits_deploy,
            'subtype_logits_priv':   subtype_logits_priv,
        }


class ImageSegmentCrossAttention(nn.Module):
    """
    Query=CXR(RadDino) 임베딩, Key/Value=Segment(Lung/Heart) 임베딩.
    사전 임베딩은 frozen이고, caching해서 뽑은 임베딩을 lookup table 형태로 사용하며 본 모듈에서는 projection/MHA만 학습.
    (Future) lesion_emb를 추가할 경우 K/V를 [seg_kv; lesion_kv] 형태로 concat하여 확장 가능.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, img_q, seg_kv, has_img):
        B, T, D = img_q.shape

        q = img_q.reshape(B * T, 1, D)        # [B*T, 1, D] - 쿼리는 1개 (CXR)
        kv = seg_kv.reshape(B * T, 2, D)      # [B*T, 2, D] - 키/밸류는 2개 (Lung, Heart)

        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(kv)

        out, _ = self.mha(q_norm, kv_norm, kv_norm) # attn_weight는 일단 처리 안함.
        out = out.view(B, T, D)
        out = out * has_img.unsqueeze(-1).to(dtype=out.dtype)  # 이미지 없는 타임스텝은 0으로 처리
        return self.ln_out(img_q + out)


###########################################################################
###########################################################################
class TaskReadout(nn.Module):
    def __init__(self, d_model=256, num_queries=1, num_classes=1, num_heads=4):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable Output Query Array
        self.query = nn.Parameter(torch.randn(1, num_queries, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
        self.classifier = nn.Linear(num_queries * d_model, num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(num_queries * d_model, num_classes)
        # )

    def forward(self, latent_embeddings, return_feature=False):
        """
        latent_embeddings: [B, L, 256] (Encoder에서 나온 L개의 Latent 토큰)
        return_feature=True: classifier 직전 flat feature [B, num_queries*d]도 함께 반환 (RD 용)
        """
        B = latent_embeddings.size(0)
        q = self.query.expand(B, -1, -1)

        attn_out, _ = self.cross_attn(query=q, key=latent_embeddings, value=latent_embeddings)
        flat_out = attn_out.reshape(B, -1)              # [B, num_queries*d]
        logits = self.classifier(flat_out)

        if return_feature:
            return logits, flat_out
        return logits


class GAPReadout(nn.Module):
    def __init__(self, d_model=256, num_classes=1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, latent_embeddings, ts_kpm=None, return_feature=False):
        if ts_kpm is not None:
            # masked mean: valid 패치만 평균
            valid = (~ts_kpm).to(dtype=latent_embeddings.dtype).unsqueeze(-1)   # [B, L, 1]
            sum_x = (latent_embeddings * valid).sum(dim=1)                       # [B, d]
            count = valid.sum(dim=1).clamp(min=1.0)                              # [B, 1]
            pooled_raw = sum_x / count                                           # [B, d]
        else:
            pooled_raw = latent_embeddings.mean(dim=1)

        pooled = self.ln(pooled_raw)                                             # [B, d_model]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        if return_feature:
            return logits, pooled
        return logits


def build_hard_segments(T, L):
    """
    T개 time step을 L개 구간으로 균등 분할함
    """
    seg_size = T // L
    segments = []
    for i in range(L):
        start = i * seg_size
        end = T if i == L - 1 else (i + 1) * seg_size # 마지막 구간은 남은 모든 time step 포함 (현 모델 구조에서는 해당 사례 없음.)
        segments.append((start, end))
    return segments


class TemporalMultiheadAttention_v2(nn.Module):
    """
    Modality-speicifc input을 받아 projection 후 MHA를 수행함.
    Latent Query는 256차원으로 고정함.
    """
    def __init__(self, d_model, num_heads, key_input_dim, attn_dropout=0.1):
        super().__init__()
        self.d_model = d_model                          # Query dim
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attn_dropout = attn_dropout                # Attention dropout
        self.q_proj = nn.Linear(d_model, d_model)       # Query Projection

        k_in = key_input_dim
        v_in = k_in

        self.k_proj = nn.Linear(k_in, d_model)          # Modality dim → 256
        self.v_proj = nn.Linear(v_in, d_model)          # Modality dim → 256
        self.out_proj = nn.Linear(d_model, d_model)     # 256 → 256

        self.ln_query = nn.LayerNorm(d_model)
        self.ln_key = nn.LayerNorm(k_in)
        self.ln_value = nn.LayerNorm(v_in)

        self.save_attn = False
        self.last_attn = None

    def forward(self, query, key, value, key_padding_mask=None, attn_mask_full=None):
        """
        attn_mask_full: [B, T_q, T_k] bool, True = block. Causal mask 등 per-query 마스크.
                        제공되면 key_padding_mask보다 우선.
        key_padding_mask: [B, T_k] bool, True = padded (legacy).
        """
        B, T_q, D = query.shape
        T_k = key.size(1)

        # Pre-LayerNorm
        query_norm = self.ln_query(query)
        key_norm = self.ln_key(key)
        value_norm = self.ln_value(value)

        # Multi-head로 분할
        Q = self.q_proj(query_norm).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key_norm).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value_norm).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)

        # Attention mask 생성 (full mask 우선, 없으면 padding mask)
        attn_mask = None
        if attn_mask_full is not None:
            # [B, T_q, T_k] → [B, 1, T_q, T_k] (head 차원 brodcast)
            attn_mask = attn_mask_full.unsqueeze(1)
            attn_mask = torch.where(attn_mask, float('-inf'), 0.0)
        elif key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(B, 1, T_q, T_k)
            attn_mask = torch.where(attn_mask, float('-inf'), 0.0)

        # ============================================================
        # Check latent embedding attention (For visualization)
        if self.save_attn:
            # scores: [B, H, T_q, T_k]
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            if attn_mask is not None:
                scores = scores + attn_mask

            attn = torch.softmax(scores, dim=-1)
            self.last_attn = attn.mean(dim=1).detach()
        # ============================================================
        # Standard SDPA
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False
        )

        # MHA 연산 합치기 [B, H, T_q, d_k] → [B, T_q, D]
        out = out.transpose(1, 2).reshape(B, T_q, D)
        out = self.out_proj(out)
        out = F.dropout(out, p=0.1, training=self.training)
        return out


class TimeSeriesCentricCrossAttention_v7(nn.Module):
    def __init__(self, args, d_model=256, num_heads=8,
                ts_input_dim=512, img_input_dim=1024, txt_input_dim=768,
                disable_cxr=False, disable_txt=False,
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_iteration = args.num_iteration
        self.disable_cxr = disable_cxr
        self.disable_txt = disable_txt
        # Causal attention mask: TS 패치는 자기 시점 이전+동시간 KV만 attend (임상 인과성)
        self.use_causal_mask = bool(args.causal_attn_mask)

        self.img_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
        )
        self.text_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
        )

        # Modality-specific Time2Vec
        self.time2vec_ts  = Time2Vec(ts_input_dim)
        self.time2vec_img = Time2Vec(img_input_dim)
        self.time2vec_txt = Time2Vec(txt_input_dim)

        self.ln_time_ts  = nn.LayerNorm(ts_input_dim)
        self.ln_time_img = nn.LayerNorm(img_input_dim)
        self.ln_time_txt = nn.LayerNorm(txt_input_dim)

        # TS Self-attention + FFN (cross-attention 진입 전 raw TS의 시간 컨텍스트 형성, 1회 적용)
        self.ts_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=0.1, batch_first=True,
        )
        self.ts_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_ts_sa  = nn.LayerNorm(d_model)
        self.ln_ts_ffn = nn.LayerNorm(d_model)

        # Query Self-attention + FFN
        self.query_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=0.1, batch_first=True,
        )
        self.query_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_sa  = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)

        # 진단 캡처용
        self._diag_enabled = False
        self.last_diag = {}

    def forward(
            self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
            img_time_indices=None, text_token_slots=None,
            img_key_padding_mask=None, text_key_padding_mask=None,
            ts_key_padding_mask=None, seq_valid_mask=None,
        ):
        B, P, _ = ts_embeddings.shape
        if img_time_indices is None:
            img_time_indices = time_indices

        # ── 진단 모드: cross-attn weight 저장 토글 ──
        self.last_diag = {}
        if self._diag_enabled:
            self.img_cross_attn.save_attn  = True
            self.text_cross_attn.save_attn = True
            self.img_cross_attn.last_attn  = None
            self.text_cross_attn.last_attn = None
        else:
            self.img_cross_attn.save_attn  = False
            self.text_cross_attn.save_attn = False

        # Time embedding (patch-level)
        time_emb_ts  = self.ln_time_ts(self.time2vec_ts(time_indices.unsqueeze(-1)))
        time_emb_img = self.ln_time_img(self.time2vec_img(img_time_indices.unsqueeze(-1)))

        # ── Fix A: 토큰별 event slot 시간 인코딩 ──
        # 같은 텍스트 이벤트의 모든 토큰에는 그 이벤트의 슬롯 시간을 broadcast.
        # 이로써 TS 패치 i가 자기 시간 근처의 이벤트 토큰을 attention으로 retrieve할 수 있게 됨.
        if (text_embeddings is not None
                and text_embeddings.size(1) > 0
                and text_token_slots is not None):
            time_emb_txt = self.ln_time_txt(
                self.time2vec_txt(text_token_slots.unsqueeze(-1).to(dtype=text_embeddings.dtype))
            )
            text_with_time = text_embeddings + time_emb_txt
        else:
            text_with_time = text_embeddings

        ts_with_time   = ts_embeddings   + time_emb_ts
        img_with_time  = img_embeddings  + time_emb_img if img_embeddings is not None else None

        # TS 자체를 Query로 활용
        query = ts_with_time

        # ── 진단: PatchTST → projection → +time2vec_ts 후 패치 다양성 ──
        if self._diag_enabled:
            self.last_diag['ts_with_time_inter_patch_cos'] = _inter_patch_cosine(
                query, mask=ts_key_padding_mask
            )

        # TS Self-attention pass (cross-attention 진입 전 1회) — raw TS 슬롯의 시간 컨텍스트 형성
        ts_normed = self.ln_ts_sa(query)
        ts_sa_out, _ = self.ts_self_attn(
            ts_normed, ts_normed, ts_normed,
            key_padding_mask=ts_key_padding_mask,
        )
        query = query + F.dropout(ts_sa_out, p=0.2, training=self.training)
        query = query + F.dropout(
            self.ts_ffn(self.ln_ts_ffn(query)),
            p=0.1, training=self.training,
        )

        # ── 진단: TS self-attn + FFN 후 패치 다양성 ──
        if self._diag_enabled:
            self.last_diag['ts_after_sa_inter_patch_cos'] = _inter_patch_cosine(
                query, mask=ts_key_padding_mask
            )

        # ── Causal mask 계산 (iteration 진입 전 1회) ──
        # TS 패치 q는 시간상 q_time 이전+동시간 KV만 attend.
        # 임상 인과성: 미래 텍스트/CXR을 보고 결정 내리면 안 됨.
        img_causal_block = None
        img_fully_blocked = None
        if self.use_causal_mask and not self.disable_cxr and img_embeddings is not None:
            img_causal_block, img_fully_blocked = create_causal_mask(
                time_indices, img_time_indices, img_key_padding_mask
            )

        text_causal_block = None
        text_fully_blocked = None
        if (self.use_causal_mask and not self.disable_txt
                and text_embeddings is not None
                and text_embeddings.size(1) > 0
                and text_token_slots is not None):
            text_causal_block, text_fully_blocked = create_causal_mask(
                time_indices, text_token_slots, text_key_padding_mask
            )

        for iter in range(self.num_iteration):
            # IMG -> Query (causal mask 적용 시 full mask 사용)
            if not self.disable_cxr and img_embeddings is not None:
                if img_causal_block is not None:
                    img_out = self.img_cross_attn(
                        query=query, key=img_with_time, value=img_with_time,
                        attn_mask_full=img_causal_block,
                    )
                    # Empty KV row의 query 출력은 zero-out (의미 없는 attention 결과 제거)
                    img_out = img_out.masked_fill(img_fully_blocked.unsqueeze(-1), 0.0)
                else:
                    img_out = self.img_cross_attn(
                        query=query, key=img_with_time, value=img_with_time,
                        key_padding_mask=img_key_padding_mask,
                    )
                query = query + F.dropout(img_out, p=0.1, training=self.training)

            # Text -> Query (causal mask 적용 시 full mask 사용)
            if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
                if text_causal_block is not None:
                    text_out = self.text_cross_attn(
                        query=query, key=text_with_time, value=text_with_time,
                        attn_mask_full=text_causal_block,
                    )
                    # Empty KV row의 query 출력은 zero-out
                    text_out = text_out.masked_fill(text_fully_blocked.unsqueeze(-1), 0.0)
                else:
                    text_out = self.text_cross_attn(
                        query=query, key=text_with_time, value=text_with_time,
                        key_padding_mask=text_key_padding_mask,
                    )
                query = query + F.dropout(text_out, p=0.1, training=self.training)

            # Query Self-attention + FFN (TS를 기반으로 업데이트된 Context Integration)
            query_normed = self.ln_sa(query)

            sa_out, _ = self.query_self_attn(
                query_normed, query_normed, query_normed,
                key_padding_mask=ts_key_padding_mask
            )
            query = query + F.dropout(sa_out, p=0.2, training=self.training)

            query = query + F.dropout(
                self.query_ffn(self.ln_ffn(query)),
                p=0.2, training=self.training,
            )

        # ── 진단: 모든 iteration 후 최종 query 패치 다양성 + cross-attn pattern 다양성 ──
        if self._diag_enabled:
            self.last_diag['ts_final_inter_patch_cos'] = _inter_patch_cosine(
                query, mask=ts_key_padding_mask
            )
            self.last_diag['img_attn_inter_patch_cos'] = _attn_inter_patch_cosine(
                self.img_cross_attn.last_attn, query_mask=ts_key_padding_mask
            )
            self.last_diag['text_attn_inter_patch_cos'] = _attn_inter_patch_cosine(
                self.text_cross_attn.last_attn, query_mask=ts_key_padding_mask
            )
            # 메모리 해제
            self.img_cross_attn.last_attn  = None
            self.text_cross_attn.last_attn = None
            self.img_cross_attn.save_attn  = False
            self.text_cross_attn.save_attn = False

        return query





# class TimeSeriesCentricCrossAttention_v5(nn.Module):
#     def __init__(self, args, d_model=256, num_heads=8,
#                 ts_input_dim=512, img_input_dim=1024, txt_input_dim=768,
#                 disable_cxr=False, disable_txt=False,
#         ):
#         super().__init__()
#         self.d_model = d_model                      # latent embedding dimension
#         self.num_heads = num_heads                  # Multi-head attention head 개수
#         self.num_latents = args.num_latents         # Latent array query 개수
#         self.num_iteration = args.num_iteration     # Iterative fusion 반복 횟수
#         self.disable_cxr = disable_cxr
#         self.disable_txt = disable_txt

#         # Latent embeddings
#         self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.trunc_normal_(self.latent_init, std=0.02)

#         self.latent_pos_embed = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

#         # Cross-attention modules with modality-specific input dimensions
#         self.ts_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
#         )
#         self.img_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
#         )
#         self.text_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
#         )

#         # Modality-specific Time2Vec for time encoding
#         self.time2vec_ts = Time2Vec(ts_input_dim)
#         self.time2vec_img = Time2Vec(img_input_dim)
#         self.time2vec_txt = Time2Vec(txt_input_dim)

#         self.ln_time_ts = nn.LayerNorm(ts_input_dim)
#         self.ln_time_img = nn.LayerNorm(img_input_dim)
#         self.ln_time_txt = nn.LayerNorm(txt_input_dim)

#         self.latent_self_attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.latent_ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.GELU(),
#             nn.Linear(d_model * 4, d_model),
#         )

#         self.ln_sa  = nn.LayerNorm(d_model)  # self-attn 전
#         self.ln_ffn = nn.LayerNorm(d_model)  # FFN 전

#         self.debug_ts_attn = None

#     def forward(
#             self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
#             img_key_padding_mask=None, text_key_padding_mask=None, seq_valid_mask=None,
#         ):

#         B, T, _ = ts_embeddings.shape
#         L = self.num_latents

#         # ================ Time Embedding ================
#         time_emb_ts  = self.ln_time_ts(self.time2vec_ts(time_indices.unsqueeze(-1)))
#         time_emb_img = self.ln_time_img(self.time2vec_img(time_indices.unsqueeze(-1)))
#         time_emb_txt = self.ln_time_txt(self.time2vec_txt(time_indices.unsqueeze(-1)))

#         latent = (self.latent_init + self.latent_pos_embed).expand(B, -1, -1)

#         # T개 time step을 L개 구간으로 나눔.
#         segments = build_hard_segments(T, L)

#         ts_with_time = ts_embeddings + time_emb_ts
#         img_with_time = img_embeddings + time_emb_img
#         text_with_time = text_embeddings + time_emb_txt

#         # ================ Iterative Fusion ================
#         for iter in range(self.num_iteration):
#             self.ts_cross_attn.save_attn = (iter == 0) # 첫 iteration만 attention 저장함. (첫 에포크 첫 배치 시각화용)

#             # ==================== TS -> Latent ====================
#             latent_updates = []
#             all_attention_weights = []

#             # 각 segment 별 독립적으로 cross-attention 수행함.
#             for i, (s, e) in enumerate(segments):
#                 q_i = latent[:, i:i+1, :] # [B, 1, D] - i번째 latent query
#                 k_i = ts_with_time[:, s:e, :] # [B, seg, D] - i번째 구간의 TS

#                 out_i = self.ts_cross_attn(query=q_i, key=k_i, value=k_i)

#                 # For visualization
#                 if self.ts_cross_attn.last_attn is not None:
#                     attn = self.ts_cross_attn.last_attn.squeeze(1)  # [B, 1, seg] -> [B, seg]
#                     attn_full = torch.zeros(B, T, device=attn.device)
#                     attn_full[:, s:e] = attn
#                     all_attention_weights.append(attn_full)

#                 latent_updates.append(out_i)

#             ts_out = torch.cat(latent_updates, dim=1) # [B, L, 256]
#             latent = latent + F.dropout(ts_out, p=0.2, training=self.training)

#             if len(all_attention_weights) > 0: # For debugging
#                 self.debug_ts_attn = torch.stack(all_attention_weights, dim=1)

#             # ==================== IMG -> Latent ====================
#             if not self.disable_cxr and img_embeddings is not None:
#                 img_out = self.img_cross_attn(
#                     query=latent,
#                     key=img_with_time,
#                     value=img_with_time,
#                     key_padding_mask=img_key_padding_mask
#                 )
#                 latent = latent + F.dropout(img_out, p=0.2, training=self.training)

#             # ==================== Text -> Latent ====================
#             if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
#                 text_out = self.text_cross_attn(
#                     query=latent,
#                     key=text_with_time,
#                     value=text_with_time,
#                     key_padding_mask=text_key_padding_mask
#                 )

#                 # Text modality random dropout
#                 # if self.training and torch.rand(1).item() < 0.50:
#                 #     text_out = text_out * 0.0
                    
#                 latent = latent + F.dropout(text_out, p=0.2, training=self.training)

#             # ==================== Temporal Mixing ====================
#             latent_normed = self.ln_sa(latent)
#             sa_out, _ = self.latent_self_attn(
#                 query=latent_normed,
#                 key=latent_normed,
#                 value=latent_normed
#             )
#             latent = latent + F.dropout(sa_out, p=0.2, training=self.training)

#             latent = latent + F.dropout(
#                 self.latent_ffn(self.ln_ffn(latent)),
#                 p=0.2, training=self.training
#             )

#         return latent


class TimeSeriesCentricCrossAttention_v6(nn.Module):
    def __init__(self, args, d_model=256, num_heads=8,
                ts_input_dim=512, img_input_dim=1024, txt_input_dim=768,
                disable_cxr=False, disable_txt=False,
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_latents = args.num_latents
        self.num_iteration = args.num_iteration
        self.disable_cxr = disable_cxr
        self.disable_txt = disable_txt

        # Latent queries
        self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
        nn.init.trunc_normal_(self.latent_init, std=0.02)
        self.latent_pos_embed = nn.Parameter(torch.empty(1, self.num_latents, d_model))
        nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

        # Modality-specific cross-attention (same as v5)
        self.ts_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
        )
        self.img_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
        )
        self.text_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
        )

        # Modality-specific Time2Vec — same shape contract as v5, just fed patch-level time
        self.time2vec_ts  = Time2Vec(ts_input_dim)
        self.time2vec_img = Time2Vec(img_input_dim)
        self.time2vec_txt = Time2Vec(txt_input_dim)

        self.ln_time_ts  = nn.LayerNorm(ts_input_dim)
        self.ln_time_img = nn.LayerNorm(img_input_dim)
        self.ln_time_txt = nn.LayerNorm(txt_input_dim)

        # Latent self-attention + FFN (same as v5)
        self.latent_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=0.1, batch_first=True,
        )
        self.latent_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_sa  = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)

        self.debug_ts_attn = None

    def forward(
            self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
            img_time_indices=None,
            img_key_padding_mask=None, text_key_padding_mask=None,
            ts_key_padding_mask=None, seq_valid_mask=None,
        ):
        B, P, _ = ts_embeddings.shape
        # img may have a longer sequence (e.g. K_cxr tokens per slot); fall back to time_indices.
        if img_time_indices is None:
            img_time_indices = time_indices

        # Time embedding (patch-level)
        time_emb_ts  = self.ln_time_ts(self.time2vec_ts(time_indices.unsqueeze(-1)))
        time_emb_img = self.ln_time_img(self.time2vec_img(img_time_indices.unsqueeze(-1)))
        time_emb_txt = self.ln_time_txt(self.time2vec_txt(time_indices.unsqueeze(-1)))

        latent = (self.latent_init + self.latent_pos_embed).expand(B, -1, -1)

        ts_with_time   = ts_embeddings   + time_emb_ts
        img_with_time  = img_embeddings  + time_emb_img
        text_with_time = text_embeddings + time_emb_txt

        for iter in range(self.num_iteration):
            self.ts_cross_attn.save_attn = (iter == 0)

            # TS -> Latent
            ts_out = self.ts_cross_attn(
                query=latent, key=ts_with_time, value=ts_with_time,
                key_padding_mask=ts_key_padding_mask,
            )
            latent = latent + F.dropout(ts_out, p=0.2, training=self.training)

            if self.ts_cross_attn.last_attn is not None:
                self.debug_ts_attn = self.ts_cross_attn.last_attn

            # IMG -> Latent
            if not self.disable_cxr and img_embeddings is not None:
                img_out = self.img_cross_attn(
                    query=latent, key=img_with_time, value=img_with_time,
                    key_padding_mask=img_key_padding_mask,
                )
                latent = latent + F.dropout(img_out, p=0.2, training=self.training)

            # Text -> Latent
            if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
                text_out = self.text_cross_attn(
                    query=latent, key=text_with_time, value=text_with_time,
                    key_padding_mask=text_key_padding_mask,
                )
                latent = latent + F.dropout(text_out, p=0.2, training=self.training)

            # Latent self-attention + FFN
            latent_normed = self.ln_sa(latent)
            sa_out, _ = self.latent_self_attn(latent_normed, latent_normed, latent_normed)
            latent = latent + F.dropout(sa_out, p=0.2, training=self.training)

            latent = latent + F.dropout(
                self.latent_ffn(self.ln_ffn(latent)),
                p=0.2, training=self.training,
            )

        return latent


class SimpleConcatFusion(nn.Module):
    def __init__(self, args, d_model=256, num_heads=8,
                 ts_input_dim=512, img_input_dim=768, txt_input_dim=768,
                 disable_cxr=False, disable_txt=False):
        super().__init__()
        del num_heads  # unused (kept for signature compatibility with v6)
        del args       # num_latents not needed any more
        self.d_model     = d_model
        self.disable_cxr = disable_cxr
        self.disable_txt = disable_txt

        self.ts_proj = nn.Linear(ts_input_dim, d_model)

        if not disable_cxr:
            self.img_proj = nn.Linear(img_input_dim, d_model)
            self.cxr_missing_token = nn.Parameter(torch.empty(img_input_dim))
            nn.init.trunc_normal_(self.cxr_missing_token, std=0.02)

        if not disable_txt:
            self.text_proj = nn.Linear(txt_input_dim, d_model)
            self.text_missing_token = nn.Parameter(torch.empty(txt_input_dim))
            nn.init.trunc_normal_(self.text_missing_token, std=0.02)

        in_dim = d_model \
               + (0 if disable_cxr else d_model) \
               + (0 if disable_txt else d_model)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model),
        )

    @staticmethod
    def _masked_mean_or_missing(emb, kpm, missing_token):
        """
        emb            : [B, K, dim]
        kpm            : [B, K]   True = padded (없음), False = real. None → all real.
        missing_token  : [dim]    learnable [MISSING] vector for empty windows.
        Returns          [B, dim] — masked mean if any token valid, else missing_token.
        """
        if kpm is None:
            return emb.mean(dim=1)
        valid    = (~kpm).float().unsqueeze(-1)        # [B, K, 1]
        count    = valid.sum(dim=1)                    # [B, 1]
        mean_emb = (emb * valid).sum(dim=1) / count.clamp(min=1)
        has_any  = (count > 0).float()                 # [B, 1]
        # has_any=1 → real mean; has_any=0 → missing token (broadcast)
        return has_any * mean_emb + (1.0 - has_any) * missing_token.unsqueeze(0)

    def forward(
            self, ts_embeddings, img_embeddings=None, text_embeddings=None,
            time_indices=None, img_time_indices=None,
            img_key_padding_mask=None, text_key_padding_mask=None,
            ts_key_padding_mask=None, seq_valid_mask=None,
        ):
        del time_indices, img_time_indices, seq_valid_mask  # unused

        # TS: masked mean over patches
        if ts_key_padding_mask is not None:
            ts_valid = (~ts_key_padding_mask).float().unsqueeze(-1)
            ts_pool = (ts_embeddings * ts_valid).sum(1) / ts_valid.sum(1).clamp(min=1)
        else:
            ts_pool = ts_embeddings.mean(dim=1)
        pooled = [self.ts_proj(ts_pool)]

        if not self.disable_cxr and img_embeddings is not None:
            img_pool = self._masked_mean_or_missing(
                img_embeddings, img_key_padding_mask, self.cxr_missing_token,
            )
            pooled.append(self.img_proj(img_pool))

        if not self.disable_txt and text_embeddings is not None:
            text_pool = self._masked_mean_or_missing(
                text_embeddings, text_key_padding_mask, self.text_missing_token,
            )
            pooled.append(self.text_proj(text_pool))

        fused = torch.cat(pooled, dim=-1)
        return self.mlp(fused)   # [B, d_model]




# class TimeSeriesCentricCrossAttention_v5(nn.Module):
#     def __init__(self, args, d_model=256, num_heads=8,
#                 ts_input_dim=512, img_input_dim=1024, txt_input_dim=768,
#                 disable_cxr=False, disable_txt=False,
#         ):
#         super().__init__()
#         self.d_model = d_model                      # latent embedding dimension
#         self.num_heads = num_heads                  # Multi-head attention head 개수
#         self.num_latents = args.num_latents         # Latent array query 개수
#         self.num_iteration = args.num_iteration     # Iterative fusion 반복 횟수
#         self.disable_cxr = disable_cxr
#         self.disable_txt = disable_txt

#         # Latent embeddings
#         self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.trunc_normal_(self.latent_init, std=0.02)
#         self.latent_pos_embed = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

#         # Cross-attention modules with modality-specific input dimensions
#         self.ts_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
#         )
#         self.img_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
#         )
#         # self.text_cross_attn = TemporalMultiheadAttention_v2(
#         #     d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
#         # )

#         # Modality-specific Time2Vec for time encoding
#         self.time2vec_ts = Time2Vec(ts_input_dim)
#         self.time2vec_img = Time2Vec(img_input_dim)
#         # self.time2vec_txt = Time2Vec(txt_input_dim)

#         self.ln_time_ts = nn.LayerNorm(ts_input_dim)
#         self.ln_time_img = nn.LayerNorm(img_input_dim)
#         # self.ln_time_txt = nn.LayerNorm(txt_input_dim)

#         self.latent_self_attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.latent_ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.GELU(),
#             nn.Linear(d_model * 4, d_model),
#         )

#         self.ln_sa  = nn.LayerNorm(d_model)  # self-attn 전
#         self.ln_ffn = nn.LayerNorm(d_model)  # FFN 전

#         self.debug_ts_attn = None

#     def forward(
#             self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
#             img_key_padding_mask=None, text_key_padding_mask=None,
#         ):

#         B, T, _ = ts_embeddings.shape
#         L = self.num_latents

#         # ================ Time Embedding ================
#         time_emb_ts  = self.ln_time_ts(self.time2vec_ts(time_indices.unsqueeze(-1)))
#         time_emb_img = self.ln_time_img(self.time2vec_img(time_indices.unsqueeze(-1)))
#         # time_emb_txt = self.ln_time_txt(self.time2vec_txt(time_indices.unsqueeze(-1)))

#         latent = (self.latent_init + self.latent_pos_embed).expand(B, -1, -1)

#         # T개 time step을 L개 구간으로 나눔.
#         segments = build_hard_segments(T, L)

#         ts_with_time = ts_embeddings + time_emb_ts
#         img_with_time = img_embeddings + time_emb_img

#         # if text_embeddings.size(1) == T:
#         #     text_with_time = text_embeddings + time_emb_txt
#         # else:
#         #     text_with_time = text_embeddings

#         # ================ Iterative Fusion ================
#         for iter in range(self.num_iteration):
#             self.ts_cross_attn.save_attn = (iter == 0) # 첫 iteration만 attention 저장함. (첫 에포크 첫 배치 시각화용)

#             # ==================== TS -> Latent ====================
#             latent_updates = []
#             all_attention_weights = []

#             # 각 segment 별 독립적으로 cross-attention 수행함.
#             for i, (s, e) in enumerate(segments):
#                 q_i = latent[:, i:i+1, :] # [B, 1, D] - i번째 latent query
#                 k_i = ts_with_time[:, s:e, :] # [B, seg, D] - i번째 구간의 TS

#                 out_i = self.ts_cross_attn(
#                     query=q_i,
#                     key=k_i,
#                     value=k_i,
#                 )

#                 # For visualization
#                 if self.ts_cross_attn.last_attn is not None:
#                     attn = self.ts_cross_attn.last_attn.squeeze(1)  # [B, 1, seg] -> [B, seg]
#                     attn_full = torch.zeros(B, T, device=attn.device)
#                     attn_full[:, s:e] = attn
#                     all_attention_weights.append(attn_full)

#                 latent_updates.append(out_i)

#             ts_out = torch.cat(latent_updates, dim=1) # [B, L, 256]
#             latent = latent + F.dropout(ts_out, p=0.2, training=self.training)

#             if len(all_attention_weights) > 0: # For debugging
#                 self.debug_ts_attn = torch.stack(all_attention_weights, dim=1)

#             # ==================== IMG -> Latent ====================
#             if not self.disable_cxr and img_embeddings is not None:
#                 img_latent_updates = []

#                 for i, (s, e) in enumerate(segments):
#                     q_i = latent[:, i:i+1, :]          # (B, 1, 256)
#                     k_i = img_with_time[:, s:e, :]     # (B, seg_len, 1024)

#                     # 해당 구간에 유효한 이미지가 있는지 확인
#                     # img_key_padding_mask: True = 패딩(이미지 없음)
#                     kp_i = img_key_padding_mask[:, s:e] if img_key_padding_mask is not None else None

#                     # 구간 내 유효 이미지가 하나도 없으면 latent 그대로 유지
#                     if kp_i is not None and kp_i.all():
#                         img_latent_updates.append(torch.zeros_like(q_i))
#                         continue

#                     img_out = self.img_cross_attn(
#                         query=latent,
#                         key=img_with_time,
#                         value=img_with_time,
#                         key_padding_mask=img_key_padding_mask
#                     )
#                     img_latent_updates.append(out_i)

#                 img_out = torch.cat(img_latent_updates, dim=1)  # (B, L, 256)
#                 latent = latent + F.dropout(img_out, p=0.2, training=self.training)

#             # ==================== Text -> Latent ====================
#             # if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
#             #     text_out = self.text_cross_attn(
#             #         query=latent,
#             #         key=text_with_time,
#             #         value=text_with_time,
#             #         key_padding_mask=text_key_padding_mask
#             #     )
#             #     # latent = latent + text_out
#             #     if self.training and torch.rand(1).item() < 0.50:
#             #         text_out = text_out * 0.0
                    
#             #     latent = latent + F.dropout(text_out, p=0.2, training=self.training)

#             # ==================== Temporal Mixing ====================
#             latent_normed = self.ln_sa(latent)
#             sa_out, _ = self.latent_self_attn(
#                 query=latent_normed,
#                 key=latent_normed,
#                 value=latent_normed
#             )
#             latent = latent + F.dropout(sa_out, p=0.2, training=self.training)

#             latent = latent + F.dropout(
#                 self.latent_ffn(self.ln_ffn(latent)),
#                 p=0.2, training=self.training
#             )

#         return latent


# class TimeSeriesCentricCrossAttention_v4(nn.Module):
#     def __init__(self, args, d_model=256, num_heads=8,
#                 ts_input_dim=512, img_input_dim=1024, txt_input_dim=768,
#                 disable_cxr=False, disable_txt=False,
#         ):
#         super().__init__()
#         self.d_model = d_model                      # latent embedding dimension
#         self.num_heads = num_heads                  # Multi-head attention head 개수
#         self.num_latents = args.num_latents         # Latent array query 개수
#         self.disable_cxr = disable_cxr
#         self.disable_txt = disable_txt

#         # Latent embeddings
#         self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.trunc_normal_(self.latent_init, std=0.02)

#         # Latent에 순서 정보를 부여하는 위치 임베딩 추가
#         self.latent_pos_embed = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

#         # Cross-attention modules with modality-specific input dimensions
#         self.ts_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
#         )
#         self.img_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
#         )
#         self.text_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
#         )

#         # latent 간 정보 교환
#         self.tsmixer = TSMixerEncoder(
#             d_model=d_model,
#             max_seq_len=self.num_latents,
#             num_layers=2
#         )

#         # Modality-specific Time2Vec for time encoding
#         self.time2vec_ts = Time2Vec(ts_input_dim) # time2vec도 다시 추가해줌.
#         self.time2vec_img = Time2Vec(img_input_dim)
#         self.time2vec_txt = Time2Vec(txt_input_dim)

#         self.ln_time_ts = nn.LayerNorm(ts_input_dim)
#         self.ln_time_img = nn.LayerNorm(img_input_dim)
#         self.ln_time_txt = nn.LayerNorm(txt_input_dim)
#         self.ln_latent = nn.LayerNorm(d_model)

#         self.debug_ts_attn = None

#     def forward(
#             self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
#             img_key_padding_mask=None, text_key_padding_mask=None, seq_valid_mask=None,
#             num_iterations=2
#         ):

#         B, T, _ = ts_embeddings.shape
#         L = self.num_latents

#         # ================ Time emb add to TS, Img, Text modality after projection ================
#         time_emb_ts_raw = self.time2vec_ts(time_indices.unsqueeze(-1))  # [B, T, 512]
#         time_emb_ts = self.ln_time_ts(time_emb_ts_raw)

#         time_emb_img_raw = self.time2vec_img(time_indices.unsqueeze(-1))  # [B, T, 1024]
#         time_emb_img = self.ln_time_img(time_emb_img_raw)

#         time_emb_txt_raw = self.time2vec_txt(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_txt = self.ln_time_txt(time_emb_txt_raw)

#         # latent = self.latent_init.expand(B, -1, -1)
#         latent = (self.latent_init + self.latent_pos_embed).expand(B, -1, -1)

#         # 유효하지 않은 time step 마스킹.
#         ts_key_padding_mask = None
#         if seq_valid_mask is not None:
#             ts_key_padding_mask = ~seq_valid_mask.bool()

#         # T개 time step을 L개 구간으로 나눔.
#         segments = build_hard_segments(T, L)

#         ts_with_time = ts_embeddings + time_emb_ts
#         img_with_time = img_embeddings + time_emb_img
        
#         if text_embeddings.size(1) == T:
#             text_with_time = text_embeddings + time_emb_txt
#         else:
#             text_with_time = text_embeddings

#         # ================ Iterative Fusion ================
#         for iter in range(num_iterations):
#             self.ts_cross_attn.save_attn = (iter == 0) # 첫 iteration만 attention 저장함. (첫 에포크 첫 배치 시각화용)

#             # ==================== TS -> Latent ====================
#             latent_updates = []
#             all_attention_weights = []

#             # 각 segment 별 독립적으로 cross-attention 수행함.
#             for i, (s, e) in enumerate(segments):
#                 q_i = latent[:, i:i+1, :] # [B, 1, D] - i번째 latent query
#                 k_i = ts_with_time[:, s:e, :] # [B, seg, D] - i번째 구간의 TS
#                 v_i = k_i

#                 kp_i = None
#                 if ts_key_padding_mask is not None:
#                     kp_i = ts_key_padding_mask[:, s:e]  # [B, seg] - padding mask

#                 out_i = self.ts_cross_attn(
#                     query=q_i,
#                     key=k_i,
#                     value=v_i,
#                     key_padding_mask=kp_i
#                 )

#                 # For visualization
#                 if self.ts_cross_attn.last_attn is not None:
#                     attn = self.ts_cross_attn.last_attn.squeeze(1)  # [B, 1, seg] -> [B, seg]
#                     attn_full = torch.zeros(B, T, device=attn.device)
#                     attn_full[:, s:e] = attn
#                     all_attention_weights.append(attn_full)

#                 latent_updates.append(out_i)

#             ts_out = torch.cat(latent_updates, dim=1) # [B, L, 256]
#             latent = latent + F.dropout(ts_out, p=0.2, training=self.training)
#             # latent = latent + ts_out

#             if len(all_attention_weights) > 0: # For debugging
#                 self.debug_ts_attn = torch.stack(all_attention_weights, dim=1)

#             # ==================== IMG -> Latent ====================
#             if not self.disable_cxr and img_embeddings is not None and img_embeddings.size(1) > 0:
#                 img_out = self.img_cross_attn(
#                     query=latent,
#                     key=img_with_time,
#                     value=img_with_time,
#                     key_padding_mask=img_key_padding_mask
#                 )
#                 # latent = latent + img_out
#                 latent = latent + F.dropout(img_out, p=0.2, training=self.training)

#             # ==================== Text -> Latent ====================
#             # if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
#             #     text_out = self.text_cross_attn(
#             #         query=latent,
#             #         key=text_with_time,
#             #         value=text_with_time,
#             #         key_padding_mask=text_key_padding_mask
#             #     )
#             #     # latent = latent + text_out
#             #     if self.training and torch.rand(1).item() < 0.50:
#             #         text_out = text_out * 0.0
                    
#             #     latent = latent + F.dropout(text_out, p=0.2, training=self.training)

#             # ==================== Temporal Mixing ====================
#             latent = self.ln_latent(latent)
#             # latent = self.tsmixer(latent, src_key_padding_mask=None)  # [B, L, 256]

#         return latent


class Time2Vec(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.linear = nn.Linear(1, d_model)

        self.w = nn.Parameter(torch.randn(1, d_model))
        self.b = nn.Parameter(torch.randn(1, d_model))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.constant_(self.linear.bias, 0.0)

        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.b, -0.1, 0.1)

    def forward(self, t):
        t_lin = self.linear(t)
        t_periodic = torch.sin(t * self.w + self.b)
        time_emb = t_lin + t_periodic
        return time_emb


# class TimeSeriesCentricCrossAttention_v4_text_cxr(nn.Module):
#     def __init__(self, args, d_model=256, num_heads=8,
#                 ts_input_dim=512, img_input_dim=768, txt_input_dim=768,
#                 disable_cxr=False, disable_txt=False, disable_prompt=False
#         ):
#         super().__init__()
#         self.d_model = d_model                      # latent embedding dimension
#         self.num_heads = num_heads                  # Multi-head attention head 개수
#         self.num_latents = args.num_latents         # Latent array query 개수
#         self.disable_cxr = disable_cxr
#         self.disable_txt = disable_txt

#         # Latent embeddings
#         self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.uniform_(self.latent_init, -0.02, 0.02)

#         # Cross-attention modules with modality-specific input dimensions
#         self.ts_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
#         )
#         self.img_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
#         )
#         self.text_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
#         )

#         # latent 간 정보 교환
#         self.tsmixer = TSMixerEncoder(
#             d_model=d_model,
#             max_seq_len=self.num_latents,
#             num_layers=2
#         )

#         # Modality-specific Time2Vec for time encoding
#         self.time2vec_ts = Time2Vec(ts_input_dim) # time2vec도 다시 추가해줌.
#         self.time2vec_img = Time2Vec(img_input_dim)
#         self.time2vec_txt = Time2Vec(txt_input_dim)

#         self.ln_time_ts = nn.LayerNorm(ts_input_dim)
#         self.ln_time_img = nn.LayerNorm(img_input_dim)
#         self.ln_time_txt = nn.LayerNorm(txt_input_dim)
#         self.ln_latent = nn.LayerNorm(d_model)

#         self.debug_ts_attn = None

#     def forward(
#             self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
#             img_key_padding_mask=None, text_key_padding_mask=None, seq_valid_mask=None,
#             num_iterations=2
#         ):

#         B, T, _ = ts_embeddings.shape
#         L = self.num_latents

#         # ================ Time emb add to TS, Img, Text modality after projection ================
#         time_emb_ts_raw = self.time2vec_ts(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_ts = self.ln_time_ts(time_emb_ts_raw)

#         time_emb_img_raw = self.time2vec_img(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_img = self.ln_time_img(time_emb_img_raw)

#         time_emb_txt_raw = self.time2vec_txt(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_txt = self.ln_time_txt(time_emb_txt_raw)

#         latent = self.latent_init.expand(B, -1, -1)

#         # 유효하지 않은 time step 마스킹.
#         ts_key_padding_mask = None
#         if seq_valid_mask is not None:
#             ts_key_padding_mask = ~seq_valid_mask.bool()

#         # T개 time step을 L개 구간으로 나눔.
#         segments = build_hard_segments(T, L)

#         # 각 segment가 유효한 데이터를 포함하는지 확인함.
#         seg_valid = torch.zeros(B, L, device=ts_embeddings.device, dtype=torch.bool)
#         if seq_valid_mask is not None:
#             seq_mask_bool = seq_valid_mask.bool()
#             for i, (s, e) in enumerate(segments):
#                 seg_valid[:, i] = seq_mask_bool[:, s:e].any(dim=1)
#         else:
#             seg_valid[:, :] = True

#         # ================ Iterative Fusion ================
#         for iter in range(num_iterations):
#             self.ts_cross_attn.save_attn = (iter == 0) # 첫 iteration만 attention 저장함. (첫 에포크 첫 배치 시각화용)

#             # ==================== Text -> Latent ====================
#             if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
#                 text_with_time = text_embeddings + time_emb_txt

#                 text_out = self.text_cross_attn(
#                     query=latent,
#                     key=text_with_time,
#                     value=text_with_time,
#                     key_padding_mask=text_key_padding_mask
#                 )
#                 latent = latent + text_out

#             # ==================== TS -> Latent ====================
#             latent_updates = []
#             all_attention_weights = []
#             ts_with_time = ts_embeddings + time_emb_ts

#             # 각 segment 별 독립적으로 cross-attention 수행함.
#             for i, (s, e) in enumerate(segments):
#                 q_i = latent[:, i:i+1, :] # [B, 1, D] - i번째 latent query
#                 k_i = ts_with_time[:, s:e, :] # [B, seg, D] - i번째 구간의 TS
#                 v_i = k_i

#                 kp_i = None
#                 if ts_key_padding_mask is not None:
#                     kp_i = ts_key_padding_mask[:, s:e]  # [B, seg] - padding mask

#                 out_i = self.ts_cross_attn(
#                     query=q_i,
#                     key=k_i,
#                     value=v_i,
#                     key_padding_mask=kp_i
#                 )

#                 # For visualization
#                 if self.ts_cross_attn.last_attn is not None:
#                     attn = self.ts_cross_attn.last_attn.squeeze(1)  # [B, 1, seg] -> [B, seg]
#                     attn_full = torch.zeros(B, T, device=attn.device)
#                     attn_full[:, s:e] = attn
#                     all_attention_weights.append(attn_full)

#                 latent_updates.append(out_i)

#             ts_out = torch.cat(latent_updates, dim=1) # [B, L, 256]
#             latent = latent + ts_out

#             if len(all_attention_weights) > 0: # For debugging
#                 self.debug_ts_attn = torch.stack(all_attention_weights, dim=1)

#             # ==================== IMG -> Latent ====================
#             if not self.disable_cxr and img_embeddings is not None and img_embeddings.size(1) > 0:
#                 img_with_time = img_embeddings + time_emb_img

#                 img_out = self.img_cross_attn(
#                     query=latent,
#                     key=img_with_time,
#                     value=img_with_time,
#                     key_padding_mask=img_key_padding_mask
#                 )
#                 latent = latent + img_out

#             # ==================== Temporal Mixing ====================
#             seg_padding_mask = ~seg_valid
#             latent = self.tsmixer(latent, src_key_padding_mask=seg_padding_mask) # [B, L, 256]
#             latent = self.ln_latent(latent)

#         return latent, seg_valid


# class TimeSeriesCentricCrossAttention_v6(nn.Module):
#     """
#     - Global Token 도입
#         - Global token -> Edema detection
#         - Local token -> Subtype classification
#     """
#     def __init__(self, args, d_model=256, num_heads=8,
#                 ts_input_dim=512, img_input_dim=768, txt_input_dim=768,
#                 disable_cxr=False, disable_txt=False, disable_prompt=False
#         ):
#         super().__init__()
#         self.d_model = d_model                          # latent embedding dimension
#         self.num_heads = num_heads                      # Multi-head attention head 개수
#         self.num_latents = args.num_latents             # Latent array query 개수
#         self.total_latents = self.num_latents + 1       # Global token 1개 추가
    
#         self.disable_cxr = disable_cxr
#         self.disable_txt = disable_txt

#         # Latent embeddings
#         self.latent_init = nn.Parameter(torch.empty(1, self.total_latents, d_model))
#         nn.init.trunc_normal_(self.latent_init, std=0.02)
#         self.latent_pos_embed = nn.Parameter(torch.empty(1, self.total_latents, d_model)) # Latent에 순서 정보를 부여하는 위치 임베딩 추가
#         nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

#         # Cross-attention modules with modality-specific input dimensions
#         self.ts_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
#         )
#         self.img_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
#         )
#         self.text_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
#         )

#         # latent 간 정보 교환
#         self.tsmixer = TSMixerEncoder(
#             d_model=d_model,
#             max_seq_len=self.total_latents,
#             num_layers=2
#         )

#         # Modality-specific Time2Vec for time encoding
#         self.time2vec_ts = Time2Vec(ts_input_dim)
#         self.time2vec_img = Time2Vec(img_input_dim)
#         self.time2vec_txt = Time2Vec(txt_input_dim)

#         self.ln_time_ts = nn.LayerNorm(ts_input_dim)
#         self.ln_time_img = nn.LayerNorm(img_input_dim)
#         self.ln_time_txt = nn.LayerNorm(txt_input_dim)
#         self.ln_latent = nn.LayerNorm(d_model)

#         self.ln_text_final = nn.LayerNorm(txt_input_dim)

#         self.debug_ts_attn = None

#     def forward(
#             self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
#             img_key_padding_mask=None, text_key_padding_mask=None,
#             num_iterations=2
#         ):

#         B, T, _ = ts_embeddings.shape
#         L = self.num_latents

#         # ================ 1. Time emb add to TS & Text modality  ================
#         time_emb_ts_raw = self.time2vec_ts(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_ts = self.ln_time_ts(time_emb_ts_raw)
#         ts_with_time = ts_embeddings + time_emb_ts

#         if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
#             time_emb_txt_raw = self.time2vec_txt(time_indices.unsqueeze(-1))  
#             time_emb_txt = self.ln_time_txt(time_emb_txt_raw)
#             text_with_time = text_embeddings + time_emb_txt

#             text_with_time = self.ln_text_final(text_with_time)

#         time_emb_img_raw = self.time2vec_img(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_img = self.ln_time_img(time_emb_img_raw)

#         # ================ 2. Image Region Sequence 처리 ================
#         img_with_time = None
#         img_key_padding_mask_flat = None
        
#         if not self.disable_cxr and img_embeddings is not None and img_embeddings.size(1) > 0:
#             # img_embeddings shape: [B, T, 6, 768]
#             time_emb_img_raw = self.time2vec_img(time_indices.unsqueeze(-1))  # [B, T, 768]
#             time_emb_img = self.ln_time_img(time_emb_img_raw)
            
#             # Time emb를 6개 region에 동일하게 브로드캐스팅하여 더함
#             img_with_time = img_embeddings + time_emb_img.unsqueeze(2) # [B, T, 6, 768]
            
#             # Attention을 위해 시퀀스를 길게 펼침: [B, T, 6, 768] -> [B, T * 6, 768]
#             Num_Regions = img_with_time.size(2)
#             img_with_time = img_with_time.view(B, T * Num_Regions, -1)
            
#             # Mask도 동일하게 펼침: [B, T] -> [B, T, 6] -> [B, T * 6]
#             if img_key_padding_mask is not None:
#                 expanded_mask = img_key_padding_mask.unsqueeze(2).expand(-1, -1, Num_Regions)
#                 img_key_padding_mask_flat = expanded_mask.reshape(B, T * Num_Regions)

#         # ================ 3. Latent 초기화 ================
#         latent = (self.latent_init + self.latent_pos_embed).expand(B, -1, -1)

#         # In window-level batching, all timesteps are valid - no padding needed
#         ts_key_padding_mask = None

#         # T개 time step을 L개 구간으로 나눔
#         segments = build_hard_segments(T, L)

#         # ================ 4. Iterative Fusion ================
#         for iter in range(num_iterations):
#             # Latent 분리
#             global_latent = latent[:, 0:1, :]  # [B, 1, 256]
#             local_latents = latent[:, 1:, :]   # [B, L, 256]

#             # ==================== 4-1. Ts to Latent matrix ====================
#             latent_updates = []

#             # 각 segment 별 독립적으로 cross-attention 수행함.
#             for i, (s, e) in enumerate(segments):
#                 q_i = local_latents[:, i:i+1, :]  # Use local_latents instead of latent
#                 k_i = ts_with_time[:, s:e, :]

#                 kp_i = ts_key_padding_mask[:, s:e] if ts_key_padding_mask is not None else None

#                 out_i = self.ts_cross_attn(query=q_i, key=k_i, value=k_i, key_padding_mask=kp_i)

#                 latent_updates.append(out_i)

#             local_latents = local_latents + torch.cat(latent_updates, dim=1)

#             # ==================== 4-2. Ts to Global vector ====================
#             global_ts_out = self.ts_cross_attn(query=global_latent, key=ts_with_time, value=ts_with_time, key_padding_mask=ts_key_padding_mask)
#             global_latent = global_latent + global_ts_out

#             # ==================== 4-3. IMG to Global vector ====================
#             if img_with_time is not None:
#                 img_out = self.img_cross_attn(query=global_latent, key=img_with_time, value=img_with_time, key_padding_mask=img_key_padding_mask_flat)
#                 global_latent = global_latent + img_out

#             # ==================== 4-4. Text to Global vector ====================
#             if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
#                 text_out = self.text_cross_attn(query=global_latent, key=text_with_time, value=text_with_time, key_padding_mask=text_key_padding_mask)
#                 global_latent = global_latent + text_out

#             # ==================== Temporal Mixing ====================
#             # Global Token이 얻어온 이미지/텍스트 정보를 Local Token들과 교환
#             latent = torch.cat([global_latent, local_latents], dim=1)  # [B, L+1, 256]

#             latent = self.tsmixer(latent, src_key_padding_mask=None)  # [B, L+1, 256]
#             latent = self.ln_latent(latent)

#         return latent  # [B, L+1, 256]








# class TimeSeriesCentricCrossAttention_v4_cxr_prior(nn.Module):
#     def __init__(self, args, d_model=256, num_heads=8,
#                 ts_input_dim=512, img_input_dim=768, txt_input_dim=768,
#                 disable_cxr=False, disable_txt=False, disable_prompt=False
#         ):
#         super().__init__()
#         self.d_model = d_model                      # latent embedding dimension
#         self.num_heads = num_heads                  # Multi-head attention head 개수
#         self.num_latents = args.num_latents         # Latent array query 개수
#         self.disable_cxr = disable_cxr
#         self.disable_txt = disable_txt

#         # Latent embeddings
#         self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.uniform_(self.latent_init, -0.02, 0.02)

#         # Cross-attention modules with modality-specific input dimensions
#         self.ts_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
#         )
#         self.img_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
#         )
#         self.text_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
#         )

#         # latent 간 정보 교환
#         self.tsmixer = TSMixerEncoder(
#             d_model=d_model,
#             max_seq_len=self.num_latents,
#             num_layers=2
#         )

#         # Modality-specific Time2Vec for time encoding
#         self.time2vec_ts = Time2Vec(ts_input_dim) # time2vec도 다시 추가해줌.
#         self.time2vec_img = Time2Vec(img_input_dim)
#         self.time2vec_txt = Time2Vec(txt_input_dim)

#         self.ln_time_ts = nn.LayerNorm(ts_input_dim)
#         self.ln_time_img = nn.LayerNorm(img_input_dim)
#         self.ln_time_txt = nn.LayerNorm(txt_input_dim)
#         self.ln_latent = nn.LayerNorm(d_model)

#         self.debug_ts_attn = None

#     def forward(
#             self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
#             img_key_padding_mask=None, text_key_padding_mask=None, seq_valid_mask=None,
#             num_iterations=2
#         ):

#         B, T, _ = ts_embeddings.shape
#         L = self.num_latents

#         # ================ Time emb add to TS, Img, Text modality after projection ================
#         time_emb_ts_raw = self.time2vec_ts(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_ts = self.ln_time_ts(time_emb_ts_raw)

#         time_emb_img_raw = self.time2vec_img(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_img = self.ln_time_img(time_emb_img_raw)

#         time_emb_txt_raw = self.time2vec_txt(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_txt = self.ln_time_txt(time_emb_txt_raw)

#         latent = self.latent_init.expand(B, -1, -1)

#         # 유효하지 않은 time step 마스킹.
#         ts_key_padding_mask = None
#         if seq_valid_mask is not None:
#             ts_key_padding_mask = ~seq_valid_mask.bool()

#         # T개 time step을 L개 구간으로 나눔.
#         segments = build_hard_segments(T, L)

#         # 각 segment가 유효한 데이터를 포함하는지 확인함.
#         seg_valid = torch.zeros(B, L, device=ts_embeddings.device, dtype=torch.bool)
#         if seq_valid_mask is not None:
#             seq_mask_bool = seq_valid_mask.bool()
#             for i, (s, e) in enumerate(segments):
#                 seg_valid[:, i] = seq_mask_bool[:, s:e].any(dim=1)
#         else:
#             seg_valid[:, :] = True

#         # ================ Iterative Fusion ================
#         for iter in range(num_iterations):
#             self.ts_cross_attn.save_attn = (iter == 0) # 첫 iteration만 attention 저장함. (첫 에포크 첫 배치 시각화용)

#             # ==================== IMG -> Latent ====================
#             if not self.disable_cxr and img_embeddings is not None and img_embeddings.size(1) > 0:
#                 img_with_time = img_embeddings + time_emb_img

#                 img_out = self.img_cross_attn(
#                     query=latent,
#                     key=img_with_time,
#                     value=img_with_time,
#                     key_padding_mask=img_key_padding_mask
#                 )
#                 latent = latent + img_out
            
#             # ==================== TS -> Latent ====================
#             latent_updates = []
#             all_attention_weights = []
#             ts_with_time = ts_embeddings + time_emb_ts

#             # 각 segment 별 독립적으로 cross-attention 수행함.
#             for i, (s, e) in enumerate(segments):
#                 q_i = latent[:, i:i+1, :] # [B, 1, D] - i번째 latent query
#                 k_i = ts_with_time[:, s:e, :] # [B, seg, D] - i번째 구간의 TS
#                 v_i = k_i

#                 kp_i = None
#                 if ts_key_padding_mask is not None:
#                     kp_i = ts_key_padding_mask[:, s:e]  # [B, seg] - padding mask

#                 out_i = self.ts_cross_attn(
#                     query=q_i,
#                     key=k_i,
#                     value=v_i,
#                     key_padding_mask=kp_i
#                 )

#                 # For visualization
#                 if self.ts_cross_attn.last_attn is not None:
#                     attn = self.ts_cross_attn.last_attn.squeeze(1)  # [B, 1, seg] -> [B, seg]
#                     attn_full = torch.zeros(B, T, device=attn.device)
#                     attn_full[:, s:e] = attn
#                     all_attention_weights.append(attn_full)

#                 latent_updates.append(out_i)

#             ts_out = torch.cat(latent_updates, dim=1) # [B, L, 256]
#             latent = latent + ts_out

#             if len(all_attention_weights) > 0: # For debugging
#                 self.debug_ts_attn = torch.stack(all_attention_weights, dim=1)

#             # ==================== Text -> Latent ====================
#             if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
#                 text_with_time = text_embeddings + time_emb_txt

#                 text_out = self.text_cross_attn(
#                     query=latent,
#                     key=text_with_time,
#                     value=text_with_time,
#                     key_padding_mask=text_key_padding_mask
#                 )
#                 latent = latent + text_out

#             # ==================== Temporal Mixing ====================
#             seg_padding_mask = ~seg_valid
#             latent = self.tsmixer(latent, src_key_padding_mask=seg_padding_mask) # [B, L, 256]
#             latent = self.ln_latent(latent)

#         return latent, seg_valid

# class TimeSeriesCentricCrossAttention_v4_ts_last(nn.Module):
#     def __init__(self, args, d_model=256, num_heads=8,
#                 ts_input_dim=512, img_input_dim=768, txt_input_dim=768,
#                 disable_cxr=False, disable_txt=False, disable_prompt=False
#         ):
#         super().__init__()
#         self.d_model = d_model                      # latent embedding dimension
#         self.num_heads = num_heads                  # Multi-head attention head 개수
#         self.num_latents = args.num_latents         # Latent array query 개수
#         self.disable_cxr = disable_cxr
#         self.disable_txt = disable_txt

#         # Latent embeddings
#         self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.uniform_(self.latent_init, -0.02, 0.02)

#         # Cross-attention modules with modality-specific input dimensions
#         self.ts_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
#         )
#         self.img_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
#         )
#         self.text_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
#         )

#         # latent 간 정보 교환
#         self.tsmixer = TSMixerEncoder(
#             d_model=d_model,
#             max_seq_len=self.num_latents,
#             num_layers=2
#         )

#         # Modality-specific Time2Vec for time encoding
#         self.time2vec_ts = Time2Vec(ts_input_dim) # time2vec도 다시 추가해줌.
#         self.time2vec_img = Time2Vec(img_input_dim)
#         self.time2vec_txt = Time2Vec(txt_input_dim)

#         self.ln_time_ts = nn.LayerNorm(ts_input_dim)
#         self.ln_time_img = nn.LayerNorm(img_input_dim)
#         self.ln_time_txt = nn.LayerNorm(txt_input_dim)
#         self.ln_latent = nn.LayerNorm(d_model)

#         self.debug_ts_attn = None

#     def forward(
#             self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
#             img_key_padding_mask=None, text_key_padding_mask=None, seq_valid_mask=None,
#             num_iterations=2
#         ):

#         B, T, _ = ts_embeddings.shape
#         L = self.num_latents

#         # ================ Time emb add to TS, Img, Text modality after projection ================
#         time_emb_ts_raw = self.time2vec_ts(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_ts = self.ln_time_ts(time_emb_ts_raw)

#         time_emb_img_raw = self.time2vec_img(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_img = self.ln_time_img(time_emb_img_raw)

#         time_emb_txt_raw = self.time2vec_txt(time_indices.unsqueeze(-1))  # [B, T, 768]
#         time_emb_txt = self.ln_time_txt(time_emb_txt_raw)

#         latent = self.latent_init.expand(B, -1, -1)

#         # 유효하지 않은 time step 마스킹.
#         ts_key_padding_mask = None
#         if seq_valid_mask is not None:
#             ts_key_padding_mask = ~seq_valid_mask.bool()

#         # T개 time step을 L개 구간으로 나눔.
#         segments = build_hard_segments(T, L)

#         # 각 segment가 유효한 데이터를 포함하는지 확인함.
#         seg_valid = torch.zeros(B, L, device=ts_embeddings.device, dtype=torch.bool)
#         if seq_valid_mask is not None:
#             seq_mask_bool = seq_valid_mask.bool()
#             for i, (s, e) in enumerate(segments):
#                 seg_valid[:, i] = seq_mask_bool[:, s:e].any(dim=1)
#         else:
#             seg_valid[:, :] = True

#         # ================ Iterative Fusion ================
#         for iter in range(num_iterations):
#             self.ts_cross_attn.save_attn = (iter == 0) # 첫 iteration만 attention 저장함. (첫 에포크 첫 배치 시각화용)

#             # ==================== IMG -> Latent ====================
#             if not self.disable_cxr and img_embeddings is not None and img_embeddings.size(1) > 0:
#                 img_with_time = img_embeddings + time_emb_img

#                 img_out = self.img_cross_attn(
#                     query=latent,
#                     key=img_with_time,
#                     value=img_with_time,
#                     key_padding_mask=img_key_padding_mask
#                 )
#                 latent = latent + img_out

#             # ==================== Text -> Latent ====================
#             if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
#                 text_with_time = text_embeddings + time_emb_txt

#                 text_out = self.text_cross_attn(
#                     query=latent,
#                     key=text_with_time,
#                     value=text_with_time,
#                     key_padding_mask=text_key_padding_mask
#                 )
#                 latent = latent + text_out
            
#             # ==================== TS -> Latent ====================
#             latent_updates = []
#             all_attention_weights = []
#             ts_with_time = ts_embeddings + time_emb_ts

#             # 각 segment 별 독립적으로 cross-attention 수행함.
#             for i, (s, e) in enumerate(segments):
#                 q_i = latent[:, i:i+1, :] # [B, 1, D] - i번째 latent query
#                 k_i = ts_with_time[:, s:e, :] # [B, seg, D] - i번째 구간의 TS
#                 v_i = k_i

#                 kp_i = None
#                 if ts_key_padding_mask is not None:
#                     kp_i = ts_key_padding_mask[:, s:e]  # [B, seg] - padding mask

#                 out_i = self.ts_cross_attn(
#                     query=q_i,
#                     key=k_i,
#                     value=v_i,
#                     key_padding_mask=kp_i
#                 )

#                 # For visualization
#                 if self.ts_cross_attn.last_attn is not None:
#                     attn = self.ts_cross_attn.last_attn.squeeze(1)  # [B, 1, seg] -> [B, seg]
#                     attn_full = torch.zeros(B, T, device=attn.device)
#                     attn_full[:, s:e] = attn
#                     all_attention_weights.append(attn_full)

#                 latent_updates.append(out_i)

#             ts_out = torch.cat(latent_updates, dim=1) # [B, L, 256]
#             latent = latent + ts_out

#             if len(all_attention_weights) > 0: # For debugging
#                 self.debug_ts_attn = torch.stack(all_attention_weights, dim=1)

#             # ==================== Temporal Mixing ====================
#             seg_padding_mask = ~seg_valid
#             latent = self.tsmixer(latent, src_key_padding_mask=seg_padding_mask) # [B, L, 256]
#             latent = self.ln_latent(latent)

#         return latent, seg_valid



class AnatomicalSpatialPooling(nn.Module):
    """
    CXR patch tokens → CLS + 5개 해부학적 regional 임베딩
    폐부종 진단에 중요한 중심부(심장/종격)에 초기 bias 부여
    """
    def __init__(self, dim=768):
        super().__init__()
        init = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])    # [cls, center(Perihilar region), Upper Left, Upper Right, Lower Left, Lower Right] / 중심부에 높은 초기값
        self.region_logits = nn.Parameter(init.log())          # softmax 전 logit
        self.proj = nn.Linear(dim, dim)

    def forward(self, cls_token, patch_tokens):
        N = patch_tokens.size(0)
        sp = patch_tokens.reshape(N, 16, 16, 768)

        regions = [
            cls_token,                             # global
            sp[:, 4:12, 4:12, :].mean(dim=(1,2)),  # center  (심장/종격/내측폐)
            sp[:, 0:6,  0:8,  :].mean(dim=(1,2)),  # upper_left
            sp[:, 0:6,  8:16, :].mean(dim=(1,2)),  # upper_right
            sp[:, 10:,  0:8,  :].mean(dim=(1,2)),  # lower_left
            sp[:, 10:,  8:16, :].mean(dim=(1,2)),  # lower_right
        ]
        stacked = torch.stack(regions, dim=1)

        # learnable weighted sum
        w = self.region_logits.softmax(dim=0)               # [6]
        fused = (stacked * w[None, :, None]).sum(dim=1)     # [N, 768]
        return self.proj(fused)
    


class AnatomicalSpatialPooling_v2(nn.Module):
    """
    CXR patch tokens → CLS + anatomical regional token을 '시퀀스'로 유지하여 반환
    """
    def __init__(self, dim=768):
        super().__init__()
        init = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])    # [cls, center(Perihilar region), Upper Left, Upper Right, Lower Left, Lower Right] / 중심부에 높은 초기값
        self.region_logits = nn.Parameter(init.log())          # softmax 전 logit
        self.proj = nn.Linear(dim, dim)

    def forward(self, cls_token, patch_tokens):
        N = patch_tokens.size(0)
        sp = patch_tokens.reshape(N, 16, 16, 768)

        regions = [
            cls_token,                             # global
            sp[:, 4:12, 4:12, :].mean(dim=(1,2)),  # center  (심장/종격/내측폐)
            sp[:, 0:6,  0:8,  :].mean(dim=(1,2)),  # upper_left
            sp[:, 0:6,  8:16, :].mean(dim=(1,2)),  # upper_right
            sp[:, 10:,  0:8,  :].mean(dim=(1,2)),  # lower_left
            sp[:, 10:,  8:16, :].mean(dim=(1,2)),  # lower_right
        ]

        stacked = torch.stack(regions, dim=1)

        w = self.region_logits.softmax(dim=0)          
        weighted_seq = stacked * w[None, :, None]     # [N, 6, 768]
        return self.proj(weighted_seq)


class AnatomicalSpatialPooling_DenseNet(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        init = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])    # [cls, center(Perihilar region), Upper Left, Upper Right, Lower Left, Lower Right] / 중심부에 높은 초기값
        self.region_logits = nn.Parameter(init.log())          # softmax 전 logit
        self.proj = nn.Linear(dim, dim)

    def forward(self, cls_token, patch_tokens):
        N = patch_tokens.size(0)
        sp = patch_tokens.reshape(N, 7, 7, 1024)

        regions = [
            cls_token,                         # global
            sp[:, 2:5, 2:5, :].mean(dim=(1,2)),  # center (심장/종격)
            sp[:, 0:3, 0:4, :].mean(dim=(1,2)),  # upper_left
            sp[:, 0:3, 3:7, :].mean(dim=(1,2)),  # upper_right
            sp[:, 4:,  0:4, :].mean(dim=(1,2)),  # lower_left
            sp[:, 4:,  3:7, :].mean(dim=(1,2)),  # lower_right
        ]

        stacked = torch.stack(regions, dim=1)
        w = self.region_logits.softmax(dim=0)          
        weighted_seq = stacked * w[None, :, None]     # [N, 6, 1024]
        return self.proj(weighted_seq)

# Multi-layer AttentionPooling
# class AttentionPooling(nn.Module):
#     """
#     L개의 latent를 하나의 window embedding으로 압축함.
#     """
#     def __init__(self, input_dim, hidden_dim=256):
#         super().__init__()
        
#         # Multi-layer attention scoring
#         self.attn_mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, latent_emb):
#         attn_scores = self.attn_mlp(latent_emb).squeeze(-1)      # [N, L]
#         attn_weights = torch.softmax(attn_scores, dim=1)                      # [N, L]
#         weighted_emb = (latent_emb * attn_weights.unsqueeze(-1)).sum(dim=1)   # [N, D]
#         return weighted_emb


# Single-layer AttentionPooling
# class AttentionPooling(nn.Module):
#     """
#     L개의 latent를 하나의 window embedding으로 압축함.
#     """
#     def __init__(self, input_dim):
#         super().__init__()

#         self.attn_fc = nn.Linear(input_dim, 1)

#     def forward(self, latent_emb, seg_valid_mask=None):
#         attn_scores = self.attn_fc(latent_emb).squeeze(-1)      # [N, L]

#         # 유효하지 않은 latent는 attention에서 제외함.
#         if seg_valid_mask is not None:
#             attn_scores = attn_scores.masked_fill(~seg_valid_mask, float('-inf'))

#         attn_weights = torch.softmax(attn_scores, dim=1)                      # [N, L]
#         weighted_emb = (latent_emb * attn_weights.unsqueeze(-1)).sum(dim=1)   # [N, D]
#         return weighted_emb


# class RegressionHead(nn.Module):
#     """
#     Regression head for predicting raw score_diff (-7~11 range)
#     Only applied to windows with Edema==1
#     """
#     def __init__(self, input_dim=256):
#         super().__init__()
#         self.regressor = nn.Linear(input_dim, 1)

#     def forward(self, x):
#         return self.regressor(x)