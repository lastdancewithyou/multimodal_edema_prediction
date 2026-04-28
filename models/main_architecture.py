import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from models.encoder import TransformerTSEncoder, TSMixerEncoder
from utils.utils import timer


class MultiModalEncoder(nn.Module):
    def __init__(self, args, disable_cxr=False, disable_txt=False, disable_prompt=False):
        super().__init__()

        self.disable_cxr = disable_cxr
        self.disable_txt = disable_txt
        self.disable_prompt = disable_prompt

        # ==================== Modality-Specific Encoders ====================
        self.ts_encoder = TransformerTSEncoder(
            input_size=args.ts_encoder_input_size,
            hidden_size=args.ts_encoder_hidden_size,
            window_size=args.window_size,
            num_layers=args.ts_encoder_num_layers,
            num_heads=8,
            dropout=0.1
        )

        self.img_encoder = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        for param in self.img_encoder.parameters():
            param.requires_grad = False

        _densenet_unfreeze = [
            "features.denseblock4",
            "features.norm5",
        ]
        for name, param in self.img_encoder.named_parameters():
            if any(name.startswith(prefix) for prefix in _densenet_unfreeze):
                param.requires_grad = True

        # Text Encoder: BioClinicalBERT
        language_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        for param in language_model.parameters():
            param.requires_grad = False

        language_lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
            bias="none",
        )

        self.text_encoder = get_peft_model(language_model, language_lora_config)

        # Clinical Prompt Tokenizer (shares same BioClinicalBERT tokenizer)
        # self.prompt_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # TS-Centric Fusion Module
        self.ts_centric_fusion = TimeSeriesCentricCrossAttention_v4(
            args=args,
            d_model=256,
            num_heads=8,
            ts_input_dim=512,         # TS encoder output dim
            img_input_dim=1024,       # DenseNet121 feature dim
            txt_input_dim=768,        # BioClinicalBERT hidden size
            disable_cxr=disable_cxr,
            disable_txt=disable_txt,
            disable_prompt=disable_prompt
        )

        self.num_virtual_prompts = 4  # 가상 프롬프트 토큰 개수
        self.ts_dim = 512             
        self.img_dim = 1024           
        self.text_dim = 768           

        self.prompt_generator = nn.Sequential(
            nn.Linear(self.ts_dim + self.img_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.num_virtual_prompts * self.text_dim)
        )

    def forward(self, args, ts_series, cxr_data, text_data, _, has_cxr, has_text, time_steps=None, current_epoch=0, total_epochs=50):
        device = ts_series.device
        B, T, _ = ts_series.shape

        # ================ Time-series Encoding ================
        with timer("TS Encoder", None):
            ts_embeddings = self.ts_encoder(ts_series)

        # ================ Image Encoding ================
        if not self.disable_cxr:
            with timer("IMG Encoder", None):
                img_tensor = torch.zeros(B, T, 1024, device=device, dtype=ts_embeddings.dtype)
                has_img = torch.zeros(B, T, device=device, dtype=torch.bool)

                unique_images = cxr_data['unique_images']
                unique_indices = cxr_data['unique_indices']
                pos = cxr_data['positions']
                is_training = cxr_data.get('is_training', False)

                if unique_images.numel() > 0:
                    b_pos, t_pos = pos[:, 0].long(), pos[:, 1].long()

                    # Apply GPU augmentation during training
                    if is_training:
                        from training.engine import cxr_train_transform_gpu
                        unique_images = cxr_train_transform_gpu(unique_images)

                    # Feature Map 추출 및 Global Average Pooling
                    features = self.img_encoder.features(unique_images)
                    unique_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)  # [N, 1024]
                    scattered = unique_features[unique_indices].to(dtype=ts_embeddings.dtype)  # [num_positions, 1024]

                    img_tensor[b_pos, t_pos] = scattered
                    has_img[b_pos, t_pos] = True

                img_embeddings = img_tensor  # [B, T, 1024]

        # Turn off Image modality (For ablation study)
        else:
            img_embeddings = torch.zeros(B, T, 1024, device=device, dtype=ts_embeddings.dtype)
            has_img = torch.zeros(B, T, device=device, dtype=torch.bool)
            has_cxr = torch.zeros_like(has_cxr)

        # ================ Dynamic Prompt Generation (Instance-Specific) ================
        # # 시계열 요약 + 이미지 요약 (존재하는 이미지만 평균)
        ts_summary = ts_embeddings.mean(dim=1)
        img_summary = torch.zeros(B, self.img_dim, device=device, dtype=ts_embeddings.dtype)
        for b in range(B):
            if has_img[b].any():
                img_summary[b] = img_embeddings[b][has_img[b]].mean(dim=0)
        
        multimodal_summary = torch.cat([ts_summary, img_summary], dim=-1)
        virtual_prompt = self.prompt_generator(multimodal_summary).view(B, self.num_virtual_prompts, self.text_dim)

        # # ================ Text Encoding & Routing ================
        align_loss = torch.tensor(0.0, device=device, dtype=ts_embeddings.dtype)

        # ================ Text Encoding ================
        if not self.disable_txt:
            with timer("Text Encoder", None):
                """
                이미지 모달리티와 동일한 방식으로 unique한 텍스트만을 뽑아서 BioClinicalBERT를 통한 인코딩을 수행함.
                """
                text_tensor = torch.zeros(B, T, 768, device=device, dtype=ts_embeddings.dtype)
                has_text_tok = torch.zeros(B, T, device=device, dtype=torch.bool)

                unique_input_ids = text_data['unique_input_ids']
                unique_attention_mask = text_data['unique_attention_mask']
                unique_indices = text_data['unique_indices']
                pos = text_data['positions']

                if unique_input_ids.numel() > 0:
                    outputs = self.text_encoder(unique_input_ids, attention_mask=unique_attention_mask)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [N_unique, 768]
                    scattered = cls_embeddings[unique_indices]
                    scattered = scattered.to(dtype=ts_embeddings.dtype)

                    b, t = pos[:, 0].long(), pos[:, 1].long()
                    text_tensor[b, t] = scattered
                    has_text_tok[b, t] = True

                text_embeddings = text_tensor

            # [Alignment Loss] Cosine Similarity (방향성 정렬)
            v_p_sum = virtual_prompt.mean(dim=1)
            r_t_sum = (text_embeddings.sum(dim=1) / (has_text_tok.sum(dim=1, keepdim=True).float() + 1e-6)).detach()

            valid_text_mask = has_text_tok.sum(dim=1) > 0  # 텍스트가 있는 배치 인덱스 확인 [B]

            if valid_text_mask.sum() > 0:
                cos_sim = F.cosine_similarity(v_p_sum[valid_text_mask], r_t_sum[valid_text_mask], dim=-1)
                align_loss = (1.0 - cos_sim).mean()
            else:
                # 텍스트가 아예 없다면 Loss 0 처리
                align_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # [Scheduled Modality Routing]
            if self.training:
                # 에포크가 진행될수록 가상 프롬프트 사용 확률(p_virtual) 증가
                p_real = max(0.1, 1.0 - (current_epoch / total_epochs)) # 최소 10%는 실제 텍스트 유지
                if torch.rand(1).item() < p_real:
                    final_text_input = text_embeddings
                    final_text_mask = has_text_tok
                else:
                    final_text_input = virtual_prompt
                    final_text_mask = torch.ones(B, self.num_virtual_prompts, device=device, dtype=torch.bool)
            else:
                # 추론 시에는 100% 가상 프롬프트 (External Validation 대응)
                final_text_input = virtual_prompt
                final_text_mask = torch.ones(B, self.num_virtual_prompts, device=device, dtype=torch.bool)

        else: 
            final_text_input = virtual_prompt
            final_text_mask = torch.ones(B, self.num_virtual_prompts, device=device, dtype=torch.bool)

        # Turn off Text modality (For ablation study)
        # else:
        #     text_embeddings = torch.zeros(B, T, 768, device=device, dtype=ts_embeddings.dtype)
        #     has_text_tok = torch.zeros(B, T, device=device, dtype=torch.bool)
        #     has_text = torch.zeros_like(has_text)


        # ================ Multimodal Fusion ================
        with timer("TS-Centric Fusion", None):
            L = self.ts_centric_fusion.num_latents
            time_idx = time_steps.to(dtype=ts_embeddings.dtype)  # [B, T]

            fused_embeddings = self.ts_centric_fusion(
                ts_embeddings=ts_embeddings,           # [B, T, 512]
                img_embeddings=img_embeddings,         # [B, T, 1024]
                # text_embeddings=text_embeddings,       # [B, T, 768]
                text_embeddings=final_text_input,
                time_indices=time_idx,                 # [B, T] - 시간 정보
                img_key_padding_mask=~has_img,         # [B, T] - 이미지 없는 곳 표시 (True where no image)
                # text_key_padding_mask=~has_text_tok,   # [B, T] - 텍스트 없는 곳 표시 (True where no text)
                text_key_padding_mask=~final_text_mask,
                # num_iterations=args.num_iterations
            )  # Output: [B, L, 256]

        return fused_embeddings, align_loss
        # return fused_embeddings


class MultiModalMultiTaskModel(nn.Module):
    """
    End-to-end multi-task model: encoder + task readout heads trained jointly.
    """
    def __init__(self, args, encoder):
        super().__init__()
        self.encoder = encoder

        self.edema_readout = TaskReadout(
            d_model=256,
            num_queries=4,
            num_classes=1 
        )

        self.subtype_readout = TaskReadout(
            d_model=256,
            num_queries=1,
            num_classes=3
        )

    def forward(self, args, ts_series, cxr_data, text_data, prompt_data, has_cxr, has_text, time_steps=None, current_epoch=0, total_epoch=50):
        # batch_embeddings = self.encoder(
        batch_embeddings, align_loss = self.encoder(
            args, ts_series, cxr_data, text_data, prompt_data, has_cxr, has_text, time_steps,
            current_epoch=current_epoch, total_epochs=total_epoch
        )

        edema_logits = self.edema_readout(batch_embeddings)
        subtype_logits = self.subtype_readout(batch_embeddings)

        return {
            'edema_logits': edema_logits,
            'subtype_logits': subtype_logits,
            'align_loss': align_loss,
            'batch_embeddings': batch_embeddings,
        }


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
        return out


class TimeSeriesCentricCrossAttention_v4(nn.Module):
    def __init__(self, args, d_model=256, num_heads=8,
                ts_input_dim=512, img_input_dim=1024, txt_input_dim=768,
                disable_cxr=False, disable_txt=False, disable_prompt=False
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
            latent = latent + ts_out

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
                latent = latent + img_out

            # ==================== Text -> Latent ====================
            if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
                text_out = self.text_cross_attn(
                    query=latent,
                    key=text_with_time,
                    value=text_with_time,
                    key_padding_mask=text_key_padding_mask
                )
                latent = latent + text_out

            # ==================== Temporal Mixing ====================
            latent = self.ln_latent(latent)
            latent = self.tsmixer(latent, src_key_padding_mask=None)  # [B, L, 256]

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


# class TimeSeriesCentricCrossAttention_v5(nn.Module):
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