import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import TransformerTSEncoder, TSMixerEncoder
from utils.utils import timer


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

        # 사전 임베딩 차원
        self.img_emb_dim = args.img_emb_dim
        self.seg_emb_dim = args.seg_emb_dim
        self.text_emb_dim = args.text_emb_dim
        self.img_shared_dim = args.img_shared_dim
        self.ts_hidden_size = args.ts_encoder_hidden_size

        # ==================== Modality-Specific Encoders ====================
        self.ts_encoder = TransformerTSEncoder(
            input_size=args.ts_encoder_input_size,
            hidden_size=args.ts_encoder_hidden_size,
            window_size=args.window_size,
            num_layers=args.ts_encoder_num_layers,
            num_heads=8,
            dropout=0.1
        )

        # CXR(raddino) → shared dim
        self.img_proj = nn.Linear(self.img_emb_dim, self.img_shared_dim)

        # Segment(hybridgnet) → shared dim
        self.seg_proj = nn.Linear(self.seg_emb_dim, self.img_shared_dim)

        # self.lesion_proj = nn.Linear(args.lesion_emb_dim, self.img_shared_dim) # Future
        
        self.img_info_combine_attn = ImageSegmentCrossAttention(
            d_model=self.img_shared_dim, num_heads=8, dropout=0.1
        )

        # TS-Centric Fusion Module
        self.ts_centric_fusion = TimeSeriesCentricCrossAttention_v5(
            args=args,
            d_model=256,
            num_heads=8,
            ts_input_dim=self.ts_hidden_size,
            img_input_dim=self.img_shared_dim,
            txt_input_dim=self.text_emb_dim,
            disable_cxr=disable_cxr,
            disable_txt=disable_txt,
        )

    def forward(self, args, ts_series, cxr_data, text_data, has_cxr, has_text, time_steps=None):
        device = ts_series.device
        B, T, _ = ts_series.shape

        # ================ Time-series Encoding ================
        with timer("TS Encoder", None):
            ts_embeddings = self.ts_encoder(ts_series)  # [B, T, ts_hidden_size]

        # ================ CXR (RadDino) + Segment (HybridGNet) ================
        # 인코더는 freeze 상태이므로 사전 임베딩 lookup만 수행하고, projection + image-segment cross-attention 구조만 학습(🔥)
        if not self.disable_cxr:
            with timer("IMG-internal Fusion", None):
                unique_img_embs = cxr_data['unique_embs']           # [N_img, img_emb_dim]
                unique_seg_embs = cxr_data['unique_segment_embs']   # [N_seg, seg_emb_dim]
                unique_ctr = cxr_data['unique_ctr']                 # [N_seg], CTR scalar
                unique_indices = cxr_data['unique_indices']         # [num_positions]
                pos = cxr_data['positions']                         # [num_positions, 2]
                seg_index_tensor = cxr_data['segment_index_tensor'] # [B, T] long, -1 = no seg
                # (Future) unique_lesion_embs = cxr_data['unique_lesion_embs']
                # (Future) lesion_index_tensor = cxr_data['lesion_index_tensor']

                img_tensor = torch.zeros(B, T, self.img_emb_dim, device=device, dtype=ts_embeddings.dtype)
                seg_tensor = torch.zeros(B, T, 2, self.seg_emb_dim, device=device, dtype=ts_embeddings.dtype)
                ctr_tensor = torch.zeros(B, T, device=device, dtype=ts_embeddings.dtype)
                has_img = torch.zeros(B, T, device=device, dtype=torch.bool)

                if unique_img_embs.numel() > 0 and pos.numel() > 0:
                    b_pos, t_pos = pos[:, 0].long(), pos[:, 1].long()

                    # RadDino 임베딩 scatter
                    scattered_img = unique_img_embs[unique_indices].to(dtype=ts_embeddings.dtype)
                    img_tensor[b_pos, t_pos] = scattered_img
                    has_img[b_pos, t_pos] = True

                    # Segment 임베딩 scatter — segment_index_tensor를 그대로 사용 (cxr_flag==1일 때만 채워짐)
                    # unique_seg_embs: [N_seg, 2, seg_emb_dim] (lung, heart)
                    seg_local_idx = seg_index_tensor[b_pos, t_pos]  # [num_positions]
                    valid_seg = seg_local_idx >= 0
                    if valid_seg.any() and unique_seg_embs.numel() > 0:
                        scattered_seg = unique_seg_embs[seg_local_idx[valid_seg]].to(dtype=ts_embeddings.dtype)
                        seg_tensor[b_pos[valid_seg], t_pos[valid_seg]] = scattered_seg

                    # CTR scatter — segment와 동일 인덱스 공간
                    if valid_seg.any() and unique_ctr.numel() > 0:
                        scattered_ctr = unique_ctr[seg_local_idx[valid_seg]].to(dtype=ts_embeddings.dtype)
                        ctr_tensor[b_pos[valid_seg], t_pos[valid_seg]] = scattered_ctr

                # Projection: shared latent space
                img_proj = self.img_proj(img_tensor)   # [B, T, img_shared_dim]
                seg_proj = self.seg_proj(seg_tensor)   # [B, T, 2, img_shared_dim]
                # (Future) lesion_proj = self.lesion_proj(lesion_tensor)
                # (Future) K/V = torch.cat([seg_proj, lesion_proj], dim=2) 형태로 확장

                # Q=CXR, K/V=Segment(lung, heart) cross-attention
                img_embeddings = self.img_info_combine_attn(
                    img_q=img_proj, seg_kv=seg_proj, has_img=has_img
                )  # [B, T, img_shared_dim]

        else:
            img_embeddings = torch.zeros(B, T, self.img_shared_dim, device=device, dtype=ts_embeddings.dtype)
            seg_proj = torch.zeros(B, T, 2, self.img_shared_dim, device=device, dtype=ts_embeddings.dtype)
            ctr_tensor = torch.zeros(B, T, device=device, dtype=ts_embeddings.dtype)
            has_img = torch.zeros(B, T, device=device, dtype=torch.bool)
            has_cxr = torch.zeros_like(has_cxr)

        # ================ Text (사전 임베딩 lookup만) ================
        if not self.disable_txt:
            with timer("Text Lookup", None):
                text_tensor = torch.zeros(B, T, self.text_emb_dim, device=device, dtype=ts_embeddings.dtype)
                has_text_tok = torch.zeros(B, T, device=device, dtype=torch.bool)

                unique_text_embs = text_data['unique_embs']     # [N_txt, text_emb_dim]
                unique_indices = text_data['unique_indices']
                pos = text_data['positions']

                if unique_text_embs.numel() > 0 and pos.numel() > 0:
                    scattered = unique_text_embs[unique_indices].to(dtype=ts_embeddings.dtype)
                    b, t = pos[:, 0].long(), pos[:, 1].long()
                    text_tensor[b, t] = scattered
                    has_text_tok[b, t] = True

                text_embeddings = text_tensor

        else:
            text_embeddings = torch.zeros(B, T, self.text_emb_dim, device=device, dtype=ts_embeddings.dtype)
            has_text_tok = torch.zeros(B, T, device=device, dtype=torch.bool)
            has_text = torch.zeros_like(has_text)

        # ================ Multimodal Fusion ================
        with timer("TS-Centric Fusion", None):
            time_idx = time_steps.to(dtype=ts_embeddings.dtype)  # [B, T]

            fused_embeddings = self.ts_centric_fusion(
                ts_embeddings=ts_embeddings,                # [B, T, ts_hidden_size]
                img_embeddings=img_embeddings,              # [B, T, img_shared_dim]
                text_embeddings=text_embeddings,            # [B, T, text_emb_dim]
                time_indices=time_idx,
                img_key_padding_mask=~has_img,
                text_key_padding_mask=~has_text_tok,
            )

        # ================ CTR Anchor: window 내 가장 마지막 CXR의 CTR ================
        # has_img: [B, T]. CXR이 하나도 없으면 0으로 둠 (cxr_anchor_mask=0이라 loss에서 마스킹됨)
        t_idx = torch.arange(T, device=device)
        masked_t = t_idx.unsqueeze(0).expand(B, T).masked_fill(~has_img, -1)
        last_t = masked_t.max(dim=1).values             # [B], CXR 없는 행은 -1
        valid_b = last_t >= 0
        last_t_safe = last_t.clamp(min=0)
        b_arange = torch.arange(B, device=device)
        ctr_anchor = ctr_tensor[b_arange, last_t_safe]  # [B]
        ctr_anchor = torch.where(valid_b, ctr_anchor, torch.zeros_like(ctr_anchor))

        return fused_embeddings, seg_proj, has_img, text_embeddings, has_text_tok, ctr_anchor


class MultiModalMultiTaskModel(nn.Module):
    """
    End-to-end model: encoder + edema readout head.
    Subtype head는 비활성화(주석) — 학습 타깃이 edema_soft 단일로 정리됨.
    """
    def __init__(self, args, encoder):
        super().__init__()
        self.encoder = encoder

        self.edema_readout = TaskReadout(d_model=256, num_queries=2, num_classes=1) # 기존 4

        # +1: window 내 마지막 CXR의 CTR scalar
        self.cardiomegaly_head = nn.Sequential(nn.Linear(args.img_shared_dim + 1, 64), nn.ReLU(), nn.Linear(64, 1))
        self.pneumonia_head = nn.Sequential(nn.Linear(args.img_shared_dim, 64), nn.ReLU(), nn.Linear(64, 1))

        # EHR + CXR fused vector projection
        self.contrastive_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.text_contrastive_proj = nn.Sequential(
            nn.Linear(args.text_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # self.subtype_readout = TaskReadout(
        #     d_model=256,
        #     num_queries=1,
        #     num_classes=3
        # )

    def forward(self, args, ts_series, cxr_data, text_data, has_cxr, has_text, time_steps=None):
        batch_embeddings, seg_proj, has_img, text_embeddings, has_text_tok, ctr_anchor = self.encoder(
            args, ts_series, cxr_data, text_data, has_cxr, has_text, time_steps
        )

        edema_logits = self.edema_readout(batch_embeddings)
        # subtype_logits = self.subtype_readout(batch_embeddings)

        mask = has_img.float().unsqueeze(-1)               # [B, T, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)             # [B, 1]

        lung_tokens_t, heart_tokens_t = seg_proj[:, :, 0, :], seg_proj[:, :, 1, :]

        lung_window = (lung_tokens_t * mask).sum(dim=1) / denom
        heart_window = (heart_tokens_t * mask).sum(dim=1) / denom

        # 마지막 CXR의 CTR(scalar)을 cardio head 입력에 concat
        cardio_input = torch.cat([heart_window, ctr_anchor.unsqueeze(-1)], dim=-1)  # [B, img_shared_dim + 1]
        cardiomegaly_logits = self.cardiomegaly_head(cardio_input)
        pneumonia_logits = self.pneumonia_head(lung_window)

        clinical_vector = batch_embeddings.mean(dim=1)            # [B, 256] # 일단 GAP
        proj_emb = self.contrastive_proj(clinical_vector)         # [B, 128]

        txt_mask = has_text_tok.float().unsqueeze(-1)            # [B, T, 1]
        txt_denom = txt_mask.sum(dim=1).clamp(min=1.0)           # [B, 1]
        
        text_vector = (text_embeddings * txt_mask).sum(dim=1) / txt_denom  # [B, 768]
        proj_text_emb = self.text_contrastive_proj(text_vector)

        valid_txt_mask = (has_text_tok.sum(dim=1) > 0)

        return {
            'edema_logits': edema_logits,
            # 'subtype_logits': subtype_logits,
            'cardiomegaly_logits': cardiomegaly_logits,
            'pneumonia_logits': pneumonia_logits,
            'proj_emb': proj_emb,                   # EHR + CXR
            'proj_text_emb': proj_text_emb,         # 텍스트
            'valid_txt_mask': valid_txt_mask        # 매칭 마스크
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
        
        # self.classifier = nn.Linear(num_queries * d_model, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_queries * d_model, num_classes)
        )

    def forward(self, latent_embeddings):
        """
        latent_embeddings: [B, L, 256] (Encoder에서 나온 L개의 Latent 토큰)
        """
        B = latent_embeddings.size(0)
        q = self.query.expand(B, -1, -1)
        
        attn_out, _ = self.cross_attn(query=q, key=latent_embeddings, value=latent_embeddings)
        
        flat_out = attn_out.reshape(B, -1)
        
        logits = self.classifier(flat_out)
        
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

    def forward(self, query, key, value, key_padding_mask=None):
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

        # Padding mask를 수식 받아 attention mask로 변환하여 사용함.
        attn_mask = None
        if key_padding_mask is not None:
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


class TimeSeriesCentricCrossAttention_v5(nn.Module):
    def __init__(self, args, d_model=256, num_heads=8,
                ts_input_dim=512, img_input_dim=1024, txt_input_dim=768,
                disable_cxr=False, disable_txt=False,
        ):
        super().__init__()
        self.d_model = d_model                      # latent embedding dimension
        self.num_heads = num_heads                  # Multi-head attention head 개수
        self.num_latents = args.num_latents         # Latent array query 개수
        self.disable_cxr = disable_cxr
        self.disable_txt = disable_txt

        # Latent embeddings
        self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
        nn.init.trunc_normal_(self.latent_init, std=0.02)
        self.latent_pos_embed = nn.Parameter(torch.empty(1, self.num_latents, d_model))
        nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

        # Cross-attention modules with modality-specific input dimensions
        self.ts_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
        )
        self.img_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
        )
        # self.text_cross_attn = TemporalMultiheadAttention_v2(
        #     d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
        # )

        # Modality-specific Time2Vec for time encoding
        self.time2vec_ts = Time2Vec(ts_input_dim)
        self.time2vec_img = Time2Vec(img_input_dim)
        # self.time2vec_txt = Time2Vec(txt_input_dim)

        self.ln_time_ts = nn.LayerNorm(ts_input_dim)
        self.ln_time_img = nn.LayerNorm(img_input_dim)
        # self.ln_time_txt = nn.LayerNorm(txt_input_dim)

        self.latent_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.latent_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )

        self.ln_sa  = nn.LayerNorm(d_model)  # self-attn 전
        self.ln_ffn = nn.LayerNorm(d_model)  # FFN 전

        self.debug_ts_attn = None

    def forward(
            self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
            img_key_padding_mask=None, text_key_padding_mask=None, seq_valid_mask=None,
            num_iterations=2
        ):

        B, T, _ = ts_embeddings.shape
        L = self.num_latents

        # ================ Time Embedding ================
        time_emb_ts  = self.ln_time_ts(self.time2vec_ts(time_indices.unsqueeze(-1)))
        time_emb_img = self.ln_time_img(self.time2vec_img(time_indices.unsqueeze(-1)))
        # time_emb_txt = self.ln_time_txt(self.time2vec_txt(time_indices.unsqueeze(-1)))

        latent = (self.latent_init + self.latent_pos_embed).expand(B, -1, -1)

        # 유효하지 않은 time step 마스킹.
        ts_key_padding_mask = None
        if seq_valid_mask is not None:
            ts_key_padding_mask = ~seq_valid_mask.bool()

        # T개 time step을 L개 구간으로 나눔.
        segments = build_hard_segments(T, L)

        ts_with_time = ts_embeddings + time_emb_ts
        img_with_time = img_embeddings + time_emb_img

        # if text_embeddings.size(1) == T:
        #     text_with_time = text_embeddings + time_emb_txt
        # else:
        #     text_with_time = text_embeddings

        # ================ Iterative Fusion ================
        for iter in range(num_iterations):
            self.ts_cross_attn.save_attn = (iter == 0) # 첫 iteration만 attention 저장함. (첫 에포크 첫 배치 시각화용)

            # ==================== TS -> Latent ====================
            latent_updates = []
            all_attention_weights = []

            # 각 segment 별 독립적으로 cross-attention 수행함.
            for i, (s, e) in enumerate(segments):
                q_i = latent[:, i:i+1, :] # [B, 1, D] - i번째 latent query
                k_i = ts_with_time[:, s:e, :] # [B, seg, D] - i번째 구간의 TS
                kp_i = ts_key_padding_mask[:, s:e] if ts_key_padding_mask is not None else None

                out_i = self.ts_cross_attn(
                    query=q_i,
                    key=k_i,
                    value=k_i,
                    key_padding_mask=kp_i
                )

                # For visualization
                if self.ts_cross_attn.last_attn is not None:
                    attn = self.ts_cross_attn.last_attn.squeeze(1)  # [B, 1, seg] -> [B, seg]
                    attn_full = torch.zeros(B, T, device=attn.device)
                    attn_full[:, s:e] = attn
                    all_attention_weights.append(attn_full)

                latent_updates.append(out_i)

            ts_out = torch.cat(latent_updates, dim=1) # [B, L, 256]
            latent = latent + F.dropout(ts_out, p=0.2, training=self.training)

            if len(all_attention_weights) > 0: # For debugging
                self.debug_ts_attn = torch.stack(all_attention_weights, dim=1)

            # ==================== IMG -> Latent ====================
            if not self.disable_cxr and img_embeddings is not None and img_embeddings.size(1) > 0:
                img_out = self.img_cross_attn(
                    query=latent,
                    key=img_with_time,
                    value=img_with_time,
                    key_padding_mask=img_key_padding_mask
                )
                # latent = latent + img_out
                latent = latent + F.dropout(img_out, p=0.2, training=self.training)

            # ==================== Text -> Latent ====================
            # if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
            #     text_out = self.text_cross_attn(
            #         query=latent,
            #         key=text_with_time,
            #         value=text_with_time,
            #         key_padding_mask=text_key_padding_mask
            #     )
            #     # latent = latent + text_out
            #     if self.training and torch.rand(1).item() < 0.50:
            #         text_out = text_out * 0.0
                    
            #     latent = latent + F.dropout(text_out, p=0.2, training=self.training)

            # ==================== Temporal Mixing ====================
            sa_out, _ = self.latent_self_attn(
                query=self.ln_sa(latent),
                key=self.ln_sa(latent),
                value=self.ln_sa(latent)
            )
            latent = latent + F.dropout(sa_out, p=0.2, training=self.training)

            latent = latent + F.dropout(
                self.latent_ffn(self.ln_ffn(latent)),
                p=0.2, training=self.training
            )

        return latent


class TimeSeriesCentricCrossAttention_v4(nn.Module):
    def __init__(self, args, d_model=256, num_heads=8,
                ts_input_dim=512, img_input_dim=1024, txt_input_dim=768,
                disable_cxr=False, disable_txt=False,
        ):
        super().__init__()
        self.d_model = d_model                      # latent embedding dimension
        self.num_heads = num_heads                  # Multi-head attention head 개수
        self.num_latents = args.num_latents         # Latent array query 개수
        self.disable_cxr = disable_cxr
        self.disable_txt = disable_txt

        # Latent embeddings
        self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
        nn.init.trunc_normal_(self.latent_init, std=0.02)

        # Latent에 순서 정보를 부여하는 위치 임베딩 추가
        self.latent_pos_embed = nn.Parameter(torch.empty(1, self.num_latents, d_model))
        nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

        # Cross-attention modules with modality-specific input dimensions
        self.ts_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=0.1
        )
        self.img_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=0.1
        )
        self.text_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=0.1
        )

        # latent 간 정보 교환
        self.tsmixer = TSMixerEncoder(
            d_model=d_model,
            max_seq_len=self.num_latents,
            num_layers=2
        )

        # Modality-specific Time2Vec for time encoding
        self.time2vec_ts = Time2Vec(ts_input_dim) # time2vec도 다시 추가해줌.
        self.time2vec_img = Time2Vec(img_input_dim)
        self.time2vec_txt = Time2Vec(txt_input_dim)

        self.ln_time_ts = nn.LayerNorm(ts_input_dim)
        self.ln_time_img = nn.LayerNorm(img_input_dim)
        self.ln_time_txt = nn.LayerNorm(txt_input_dim)
        self.ln_latent = nn.LayerNorm(d_model)

        self.debug_ts_attn = None

    def forward(
            self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
            img_key_padding_mask=None, text_key_padding_mask=None, seq_valid_mask=None,
            num_iterations=2
        ):

        B, T, _ = ts_embeddings.shape
        L = self.num_latents

        # ================ Time emb add to TS, Img, Text modality after projection ================
        time_emb_ts_raw = self.time2vec_ts(time_indices.unsqueeze(-1))  # [B, T, 512]
        time_emb_ts = self.ln_time_ts(time_emb_ts_raw)

        time_emb_img_raw = self.time2vec_img(time_indices.unsqueeze(-1))  # [B, T, 1024]
        time_emb_img = self.ln_time_img(time_emb_img_raw)

        time_emb_txt_raw = self.time2vec_txt(time_indices.unsqueeze(-1))  # [B, T, 768]
        time_emb_txt = self.ln_time_txt(time_emb_txt_raw)

        # latent = self.latent_init.expand(B, -1, -1)
        latent = (self.latent_init + self.latent_pos_embed).expand(B, -1, -1)

        # 유효하지 않은 time step 마스킹.
        ts_key_padding_mask = None
        if seq_valid_mask is not None:
            ts_key_padding_mask = ~seq_valid_mask.bool()

        # T개 time step을 L개 구간으로 나눔.
        segments = build_hard_segments(T, L)

        ts_with_time = ts_embeddings + time_emb_ts
        img_with_time = img_embeddings + time_emb_img
        
        if text_embeddings.size(1) == T:
            text_with_time = text_embeddings + time_emb_txt
        else:
            text_with_time = text_embeddings

        # ================ Iterative Fusion ================
        for iter in range(num_iterations):
            self.ts_cross_attn.save_attn = (iter == 0) # 첫 iteration만 attention 저장함. (첫 에포크 첫 배치 시각화용)

            # ==================== TS -> Latent ====================
            latent_updates = []
            all_attention_weights = []

            # 각 segment 별 독립적으로 cross-attention 수행함.
            for i, (s, e) in enumerate(segments):
                q_i = latent[:, i:i+1, :] # [B, 1, D] - i번째 latent query
                k_i = ts_with_time[:, s:e, :] # [B, seg, D] - i번째 구간의 TS
                v_i = k_i

                kp_i = None
                if ts_key_padding_mask is not None:
                    kp_i = ts_key_padding_mask[:, s:e]  # [B, seg] - padding mask

                out_i = self.ts_cross_attn(
                    query=q_i,
                    key=k_i,
                    value=v_i,
                    key_padding_mask=kp_i
                )

                # For visualization
                if self.ts_cross_attn.last_attn is not None:
                    attn = self.ts_cross_attn.last_attn.squeeze(1)  # [B, 1, seg] -> [B, seg]
                    attn_full = torch.zeros(B, T, device=attn.device)
                    attn_full[:, s:e] = attn
                    all_attention_weights.append(attn_full)

                latent_updates.append(out_i)

            ts_out = torch.cat(latent_updates, dim=1) # [B, L, 256]
            latent = latent + F.dropout(ts_out, p=0.2, training=self.training)
            # latent = latent + ts_out

            if len(all_attention_weights) > 0: # For debugging
                self.debug_ts_attn = torch.stack(all_attention_weights, dim=1)

            # ==================== IMG -> Latent ====================
            if not self.disable_cxr and img_embeddings is not None and img_embeddings.size(1) > 0:
                img_out = self.img_cross_attn(
                    query=latent,
                    key=img_with_time,
                    value=img_with_time,
                    key_padding_mask=img_key_padding_mask
                )
                # latent = latent + img_out
                latent = latent + F.dropout(img_out, p=0.2, training=self.training)

            # ==================== Text -> Latent ====================
            # if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
            #     text_out = self.text_cross_attn(
            #         query=latent,
            #         key=text_with_time,
            #         value=text_with_time,
            #         key_padding_mask=text_key_padding_mask
            #     )
            #     # latent = latent + text_out
            #     if self.training and torch.rand(1).item() < 0.50:
            #         text_out = text_out * 0.0
                    
            #     latent = latent + F.dropout(text_out, p=0.2, training=self.training)

            # ==================== Temporal Mixing ====================
            latent = self.ln_latent(latent)
            # latent = self.tsmixer(latent, src_key_padding_mask=None)  # [B, L, 256]

        return latent


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