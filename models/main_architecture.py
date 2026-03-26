import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from transformers import AutoModel, AutoTokenizer

from models.encoder import TransformerTSEncoder, TSMixerEncoder
from models.cxrformer_model import CXformer, apply_lora_to_cxformer
from peft import LoraConfig, get_peft_model, TaskType
from utils.utils import timer


class MultiModalEncoder(nn.Module):
    def __init__(self, args, disable_cxr=False, disable_txt=False, disable_prompt=False):
        super().__init__()

        self.disable_cxr = disable_cxr
        self.disable_txt = disable_txt
        self.disable_prompt = disable_prompt

        # ==================== Modality-Specific Encoders ====================
        # Time-series Encoder
        self.ts_encoder = TransformerTSEncoder(
            input_size=args.ts_encoder_input_size,
            hidden_size=args.ts_encoder_hidden_size,
            window_size=args.window_size,
            num_layers=args.ts_encoder_num_layers,
            num_heads=8,
            dropout=0.1
        )

        ##########################################################################################
        # Image Encoder: DenseNet121 from torchxrayvision (pretrained on MIMIC-CXR)
        # self.img_encoder = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        # for param in self.img_encoder.parameters():
        #     param.requires_grad = False

        # _densenet_unfreeze = [
        #     "features.denseblock4",
        #     "features.norm5",
        # ]
        # for name, param in self.img_encoder.named_parameters():
        #     if any(name.startswith(prefix) for prefix in _densenet_unfreeze):
        #         param.requires_grad = True
        ##########################################################################################
        img_model = CXformer.from_pretrained("m42-health/CXformer-base", context_dim=768)

        # Enable gradient checkpointing to save memory
        img_model.gradient_checkpointing = True

        self.img_encoder = apply_lora_to_cxformer(
            img_model,
            r=16,
            alpha=32,
            dropout=0.1
        )

        self.spatial_pooling = AnatomicalSpatialPooling(dim=768)
        
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
        self.prompt_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # TS-Centric Fusion Module
        self.ts_centric_fusion = TimeSeriesCentricCrossAttention_v4(
            args=args,
            d_model=256,
            num_heads=8,
            ts_input_dim=512,       # TS encoder output dim
            img_input_dim=768,      # DenseNet121 feature dim
            txt_input_dim=768,      # BioClinicalBERT hidden size
            disable_cxr=disable_cxr,
            disable_txt=disable_txt,
            disable_prompt=disable_prompt
        )

        # Attention Pooling
        self.attention_pooling = AttentionPooling(input_dim=256)

    def forward(self, args, ts_series, cxr_data, text_data, prompt_data, has_cxr, has_text,
                window_mask, seq_valid_mask, time_steps=None):

        device = ts_series.device
        B, W, T, D = ts_series.shape

        # ================ Clinical Prompt Encoding ================
        with timer("Clinical Prompt Encoder", None):
            unique_prompt_texts = prompt_data['unique_prompt_texts']
            prompt_index_tensor = prompt_data['prompt_index_tensor']

            # Tokenize unique prompts
            tokenized_prompts = self.prompt_tokenizer(
                unique_prompt_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(device)

            # Encode with BioClinicalBERT
            prompt_outputs = self.text_encoder(
                input_ids=tokenized_prompts['input_ids'],
                attention_mask=tokenized_prompts['attention_mask']
            )

            # Extract CLS token embeddings [N_unique_prompts, 768]
            unique_prompt_embeddings = prompt_outputs.last_hidden_state[:, 0, :]

        # ================ Time-series Encoding ================
        with timer("TS Encoder", None):
            ts_series_flat = ts_series.view(B * W, T, D)
            seq_valid_mask_flat = seq_valid_mask.view(B * W, T)         # 윈도우 내 유효한 time step을 masking
            seq_valid_lengths_flat = seq_valid_mask_flat.sum(dim=-1)
            window_mask_flat = window_mask.reshape(B * W)               # 패딩되지 않은 유효한 window 선별

            valid_indices = window_mask_flat.nonzero(as_tuple=False).squeeze(1)     # 유효한 window 인덱스 추출
            valid_ts_series = ts_series_flat[valid_indices]                         # 유효한 window만 선택
            valid_seq_lengths = seq_valid_lengths_flat[valid_indices]

            # ===== Clinical Prompt Prefix Tuning for TS Encoder =====
            """
            1. 각 window의 마지막 valid timestep의 clinical prompt 선택
            2. Valid window에만 prompt 전달
            3. TS encoder에 clinical prompt를 전달
            """
            # Use window-level last timestep prompt for each window
            if not self.disable_prompt:
                # Get prompt for each window's last valid timestep
                window_prompt_indices = torch.zeros(B, W, dtype=torch.long, device=device)
                for b in range(B):
                    for w in range(W):
                        if window_mask[b, w]:
                            valid_mask = seq_valid_mask[b, w, :]  # Find last valid timestep in this window
                            if valid_mask.any():
                                last_valid_t = valid_mask.nonzero(as_tuple=False)[-1].item()
                                window_prompt_indices[b, w] = prompt_index_tensor[b, w, last_valid_t]

                # Get prompt embeddings for valid windows
                window_prompt_indices_flat = window_prompt_indices.view(B * W)
                valid_window_prompts = unique_prompt_embeddings[window_prompt_indices_flat[valid_indices]] # [BW, 768] -> [Nwin, 768]

            else:
                valid_window_prompts = None

            # Encode with clinical prompt prefix
            valid_ts_encoded = self.ts_encoder(
                valid_ts_series,
                valid_seq_lengths,
                clinical_prompt=valid_window_prompts  # [Nwin, 768] or None
            )

            ts_encoded = torch.zeros(
                B * W, T, valid_ts_encoded.shape[-1],
                device=device, dtype=valid_ts_encoded.dtype
            )
            ts_encoded[valid_indices] = valid_ts_encoded
            ts_embeddings = ts_encoded.view(B, W, T, -1)  # [B, W, T, 512] # time-series는 512차원 embedding 출력.

        # ================ Image Encoding ================
        """
        - 같은 이미지와 텍스트가 window 형태의 데이터 입력에서는 여러 시간대에 재사용됨.
        - Forward pass 과정에서 GPU에 고유한 이미지와 텍스트만 올림으로써 최적화를 도모함.
        - Clinical prompt context를 각 이미지의 시간적으로 정확한 context로 전달하여 referencing 가능하게 함.
        """
        if not self.disable_cxr:
            with timer("IMG Encoder", None):
                img_tensor = torch.zeros(B, W, T, 768, device=device, dtype=ts_embeddings.dtype) # CXFormer ViT-Base: 768
                has_img = torch.zeros(B, W, T, device=device, dtype=torch.bool)

                unique_images = cxr_data['unique_images']       # unique한 이미지만
                unique_indices = cxr_data['unique_indices']     # 각 위치가 어떤 unique image인지
                pos = cxr_data['positions']                     # (batch, window, timestep)

                if unique_images.numel() > 0:
                    # 현재 referencing은 중단 상태임.
                    # Get prompts for all positions first
                    b_pos, w_pos, t_pos = pos[:, 0].long(), pos[:, 1].long(), pos[:, 2].long()
                    # all_prompt_indices = prompt_index_tensor[b_pos, w_pos, t_pos]  # [1183] - 모든 위치의 prompt index

                    # # For each unique image, get the prompt from its first occurrence
                    # # unique_indices: [1183] - 각 위치가 몇 번째 unique image인지
                    # num_unique_images = unique_images.size(0)
                    # unique_prompt_indices = torch.zeros(num_unique_images, dtype=torch.long, device=device)

                    # for i in range(num_unique_images):
                    #     first_occurrence_mask = (unique_indices == i)
                    #     first_occurrence_idx = first_occurrence_mask.nonzero(as_tuple=False)[0].item()
                    #     unique_prompt_indices[i] = all_prompt_indices[first_occurrence_idx]

                    # image_context_embeddings = unique_prompt_embeddings[unique_prompt_indices]  # [79, 768]
                    # image_context_embeddings = image_context_embeddings.unsqueeze(1)  # [79, 1, 768]

                    # Encode images with clinical prompt as context
                    outputs = self.img_encoder(unique_images, context=None)
                    # unique_features = outputs["x_norm_clstoken"]
                    
                    #############################################
                    # Spatiality 부여
                    cls_token = outputs['x_norm_clstoken']
                    patch_tokens = outputs['x_norm_patchtokens']

                    unique_features = self.spatial_pooling(cls_token, patch_tokens)
                    #############################################

                    scattered = unique_features[unique_indices].to(dtype=ts_embeddings.dtype)
                    img_tensor[b_pos, w_pos, t_pos] = scattered     # 원래 위치 (b, w, t)에 이미지 임베딩 넣기
                    has_img[b_pos, w_pos, t_pos] = True             # 이미지 존재 여부 마스킹

                img_embeddings = img_tensor

        # Turn off Image modality (For ablation study)
        else:
            img_embeddings = torch.zeros(B, W, T, 768, device=device, dtype=ts_embeddings.dtype)
            has_img = torch.zeros(B, W, T, device=device, dtype=torch.bool)
            has_cxr = torch.zeros_like(has_cxr)

        # ================ Text Encoding ================
        if not self.disable_txt:
            with timer("Text Encoder", None):
                """
                이미지 모달리티와 동일한 방식으로 unique한 텍스트만을 뽑아서 BioClinicalBERT를 통한 인코딩을 수행함.
                """
                text_tensor = torch.zeros(B, W, T, 768, device=device, dtype=ts_embeddings.dtype)
                has_text_tok = torch.zeros(B, W, T, device=device, dtype=torch.bool)

                unique_input_ids = text_data['unique_input_ids']
                unique_attention_mask = text_data['unique_attention_mask']
                unique_indices = text_data['unique_indices']
                pos = text_data['positions']

                if unique_input_ids.numel() > 0:
                    outputs = self.text_encoder(unique_input_ids, attention_mask=unique_attention_mask)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [N_unique, 768]
                    scattered = cls_embeddings[unique_indices]
                    scattered = scattered.to(dtype=ts_embeddings.dtype)

                    b, w, t = pos[:, 0].long(), pos[:, 1].long(), pos[:, 2].long()
                    text_tensor[b, w, t] = scattered
                    has_text_tok[b, w, t] = True

                text_embeddings = text_tensor

        # Turn off Text modality (For ablation study)
        else:
            text_embeddings = torch.zeros(B, W, T, 768, device=device, dtype=ts_embeddings.dtype)
            has_text_tok = torch.zeros(B, W, T, device=device, dtype=torch.bool)
            has_text = torch.zeros_like(has_text)

        # ================ Multimodal Fusion ================
        with timer("TS-Centric Fusion", None):
            """
            Multi-modal Embeddings
            ts_embeddings: [B, W, T, 512]
            img_embeddings: [B, W, T, 768] - sparse
            text_embeddings: [B, W, T, 768] - sparse
            """
            BW = B * W
            ts_flat = ts_embeddings.reshape(BW, T, 512)
            seq_mask_flat = seq_valid_mask.reshape(BW, T).float()       # 실제 데이터가 들어있는 time step을 뽑아냄.
            ts_flat_masked = ts_flat * seq_mask_flat.unsqueeze(-1)

            img_flat = img_embeddings.reshape(BW, T, 768)
            txt_flat = text_embeddings.reshape(BW, T, 768)

            has_img_flat = has_img.reshape(BW, T)
            has_txt_flat = has_text_tok.reshape(BW, T)

            win_mask_flat = window_mask.reshape(BW)
            update_idx = win_mask_flat.nonzero(as_tuple=False).squeeze(1) # 유효한 window만 fusion을 수행할 수 있도록 함.

            L = self.ts_centric_fusion.num_latents
            fused_latent_flat = torch.zeros(BW, L, 256, device=device, dtype=ts_embeddings.dtype)
            seg_valid_flat = torch.zeros(BW, L, device=device, dtype=torch.bool)

            if update_idx.numel() > 0:
                ts_kv = ts_flat_masked[update_idx]
                img_kv = img_flat[update_idx]
                txt_kv = txt_flat[update_idx]

                img_pad = ~has_img_flat[update_idx]         # 어느 time step에 이미지가 없는가?
                txt_pad = ~has_txt_flat[update_idx]         # 어느 time step에 텍스트가 없는가?

                # Time indices
                if time_steps is not None:
                    # 환자의 실제 ICU 체류 시간을 기준으로 함.
                    time_steps_flat = time_steps.view(BW, T)
                    time_idx = time_steps_flat[update_idx].to(dtype=ts_embeddings.dtype)
                else:
                    time_idx = torch.arange(T, device=device, dtype=ts_embeddings.dtype).unsqueeze(0).expand(ts_kv.size(0), -1)

                seq_valid = seq_mask_flat[update_idx]

                # MultiModal Fusion
                updated_latent, seg_valid_batch = self.ts_centric_fusion(
                    ts_embeddings=ts_kv,                        # [Nwin, T, 512]
                    img_embeddings=img_kv,                      # [Nwin, T, 768]
                    text_embeddings=txt_kv,                     # [Nwin, T, 768]
                    time_indices=time_idx,                      # [Nwin, T] - 시간 정보
                    img_key_padding_mask=img_pad,               # [Nwin, T] - 이미지 없는 곳 표시
                    text_key_padding_mask=txt_pad,              # [Nwin, T] - 텍스트 없는 곳 표시
                    seq_valid_mask=seq_valid,                   # [Nwin, T] - 유효한 timestep
                    num_iterations=args.num_iterations          
                )

                fused_latent_flat[update_idx] = updated_latent  # [BW. L, 256] - 유효한 window 위치에만 fusion 결과 넣음.
                seg_valid_flat[update_idx] = seg_valid_batch

            fused_embeddings = fused_latent_flat.view(B, W, L, 256) # [B, W, L, 256] shape으로 복원함.
            seg_valid_out = seg_valid_flat.view(B, W, L)

        # ================ Attention Pooling ================
        # Pool fused latents to window-level embeddings
        fused_flat = fused_embeddings.reshape(BW, L, 256)       # [B, W, L, 256] → [B, W, 256]
        seg_valid_flat = seg_valid_out.reshape(BW, L)           # [B, W, L] → [BW, L]
        window_valid_mask = window_mask.reshape(BW).bool()      # [B, W] → [BW]

        # Pool only valid windows
        valid_fused = fused_flat[window_valid_mask]             # [Nwin, L, 256]
        valid_seg_valid = seg_valid_flat[window_valid_mask]     # [Nwin, L]
        pooled_emb = self.attention_pooling(valid_fused, seg_valid_mask=valid_seg_valid) # 유효한 window만 attention pooling에 사용함.

        # [B, W, 256]으로 복원함. (배치 간 Shape 맞춰주기)
        window_embeddings_flat = torch.zeros(BW, 256, device=device, dtype=pooled_emb.dtype)
        window_embeddings_flat[window_valid_mask] = pooled_emb
        window_embeddings = window_embeddings_flat.view(B, W, 256)

        return window_embeddings, window_mask


class MultiModalMultiTaskModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

        # Binary classifier head for edema detection
        self.edema_classifier = nn.Linear(256, 1)  # [B, W, 256] → [B, W, 1]

        # Hierarchical classifier for subtype classification
        self.subtype_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, 2)
        )

        # Regression head for score_diff prediction (only for Edema==1)
        self.regression_head = RegressionHead(input_dim=256)

    def forward(self, args, ts_series, cxr_data, text_data, prompt_data, has_cxr, has_text,
            window_mask, seq_valid_mask, time_steps=None,
        ):

        B, W = window_mask.shape

        # Encoder
        window_embeddings, _ = self.encoder(
            args, ts_series, cxr_data, text_data, prompt_data, has_cxr, has_text,
            window_mask, seq_valid_mask, time_steps
        )

        # Binary logits for edema detection
        edema_logits = self.edema_classifier(window_embeddings)

        # Subtype logits for valid windows
        subtype_logits = self.subtype_classifier(window_embeddings)

        # Regression predictions for raw score_diff (-7~11 range)
        regression_preds = self.regression_head(window_embeddings)

        # Extract valid windows
        """
        - 환자마다 ICU stay가 다르고, padding 된 window가 존재함.
        - padding된 window에 projection을 적용해서 낭비를 막기 위함임.
        """
        # Temporal indices
        BW = B * W
        window_embeddings_flat = window_embeddings.reshape(BW, 256)
        window_valid_mask = window_mask.reshape(BW).bool()
        valid_windows = window_embeddings_flat[window_valid_mask]  # [Nwin, 256]

        window_time_indices = torch.arange(W, device=window_embeddings.device).unsqueeze(0).expand(B, W)
        batch_indices = torch.arange(B, device=window_embeddings.device).unsqueeze(1).expand(B, W)
        window_time_indices_flat = window_time_indices.reshape(BW)[window_valid_mask]
        batch_indices_flat = batch_indices.reshape(BW)[window_valid_mask]

        return {
            'edema_logits': edema_logits,                     # [B, W, 1]
            'subtype_logits': subtype_logits,                   # [B, W, 2]
            'regression_preds': regression_preds,               # [B, W, 1] - raw score_diff predictions (-7~11)
            'window_embeddings': window_embeddings,             # [B, W, 256]
            'valid_embeddings': valid_windows,                  # for contrastive
            'window_time_indices': window_time_indices_flat,    # [Nwin]
            'batch_indices': batch_indices_flat,                # [Nwin]
        }


# Multi-layer AttentionPooling
class AttentionPooling(nn.Module):
    """
    L개의 latent를 하나의 window embedding으로 압축함.
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        # Multi-layer attention scoring
        self.attn_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent_emb, seg_valid_mask=None):
        attn_scores = self.attn_mlp(latent_emb).squeeze(-1)      # [N, L]

        if seg_valid_mask is not None:
            attn_scores = attn_scores.masked_fill(~seg_valid_mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=1)                      # [N, L]
        weighted_emb = (latent_emb * attn_weights.unsqueeze(-1)).sum(dim=1)   # [N, D]
        return weighted_emb


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


class ProjectionHead(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 2 layer MLP
        self.fc1 = nn.Linear(args.head_input_dim, args.head_hidden_dim1)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(args.head_hidden_dim1, args.head_hidden_dim2)

    def forward(self, x):
        original_shape = x.shape                # [B, W, D]
        x = x.view(-1, x.shape[-1])             # [B*W, D]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.view(*original_shape[:-1], -1)    # [B, W, proj_dim]
        return x


class RegressionHead(nn.Module):
    """
    Regression head for predicting raw score_diff (-7~11 range)
    Only applied to windows with Edema==1
    """
    def __init__(self, input_dim=256):
        super().__init__()
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.regressor(x)


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


# class TimeSeriesCentricCrossAttention_v5(nn.Module):
#     def __init__(self, args, d_model=256, num_heads=8,
#                 ts_input_dim=512, img_input_dim=768, txt_input_dim=768,
#                 disable_cxr=False, disable_txt=False, disable_prompt=False, dropout=0.1
#         ):
#         super().__init__()
#         self.d_model = d_model                      # latent embedding dimension
#         self.num_heads = num_heads                  # Multi-head attention head 개수
#         self.num_latents = args.num_latents         # Latent array query 개수
#         self.disable_cxr = disable_cxr
#         self.disable_txt = disable_txt
#         self.disable_prompt = disable_prompt

#         # Latent embeddings
#         self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
#         nn.init.uniform_(self.latent_init, -0.02, 0.02)

#         self.ts_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=ts_input_dim, attn_dropout=dropout
#         )
#         self.img_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=img_input_dim, attn_dropout=dropout
#         )
#         self.text_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=dropout
#         )
#         self.ctx_cross_attn = TemporalMultiheadAttention_v2(
#             d_model, num_heads, key_input_dim=txt_input_dim, attn_dropout=dropout
#         )

#         # latent 간 정보 교환
#         self.tsmixer = TSMixerEncoder(
#             d_model=d_model,
#             max_seq_len=self.num_latents,
#             num_layers=2,
#         )

#         # Modality-specific Time2Vec for time encoding - 이미지와 텍스트에 시간 정보를 줘서 어느 시점에 촬영했는지에 대한 정보를 부여함.
#         self.time2vec_img = Time2Vec(img_input_dim) 
#         self.time2vec_txt = Time2Vec(txt_input_dim) 

#         self.ln_time_img = nn.LayerNorm(img_input_dim)
#         self.ln_time_txt = nn.LayerNorm(txt_input_dim)
#         self.ln_latent = nn.LayerNorm(d_model)

#         self.debug_ts_attn = None

#     def forward(
#             self, ctx_embeddings, ctx_key_padding_masks, 
#             ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
#             img_key_padding_mask=None, text_key_padding_mask=None, seq_valid_mask=None,
#             num_iterations=2
#         ):

#         B, T, _ = ts_embeddings.shape
#         L = self.num_latents

#         # ================ Time emb add to Img, Text modality after projection ================
#         time_emb_img = self.ln_time_img(self.time2vec_img(time_indices.unsqueeze(-1)))
#         time_emb_txt = self.ln_time_txt(self.time2vec_txt(time_indices.unsqueeze(-1)))

#         # Latent matrix 초기화
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
#             self.ts_cross_attn.save_attn = (iter == 0)

#             # ==================== Clinical Prompt -> Latent ====================
#             if not self.disable_prompt:
#                 ctx_updates = []
#                 all_ctx_attn_weights = []

#                 for i in range(L):
#                     q_i = latent[:, i:i+1, :]                # [B, 1, 256]
#                     ctx_i = ctx_embeddings[i]                # [B, seq_len, 768]
#                     ctx_mask_i = ctx_key_padding_masks[i]    # [B, seq_len]

#                     ctx_out_i = self.ctx_cross_attn(
#                         query=q_i,
#                         key=ctx_i,
#                         value=ctx_i,
#                         key_padding_mask=ctx_mask_i
#                     )
#                     ctx_updates.append(ctx_out_i)

#                     if iter == 0 and self.ctx_cross_attn.last_attn is not None:
#                         all_ctx_attn_weights.append(
#                             self.ctx_cross_attn.last_attn.squeeze(1).detach()  # [B, seq_len]
#                         )

#                 ctx_out = torch.cat(ctx_updates, dim=1)       # [B, L, 256]
#                 latent = latent + ctx_out

#                 if len(all_ctx_attn_weights) > 0:
#                     self.debug_ctx_attn = torch.stack(all_ctx_attn_weights, dim=1)

#             # ==================== TS -> Latent ====================
#             ts_updates = []
#             all_ts_attention_weights = []

#             # 각 segment 별 독립적으로 cross-attention 수행함.
#             for i, (s, e) in enumerate(segments):
#                 q_i = latent[:, i:i+1, :] # [B, 1, D] - i번째 latent query
#                 k_i = ts_embeddings[:, s:e, :] # [B, seg, D] - i번째 구간의 TS
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
#                     all_ts_attention_weights.append(attn_full)

#                 ts_updates.append(out_i)

#             ts_out = torch.cat(ts_updates, dim=1) # [B, L, 256]
#             latent = latent + ts_out

#             if len(all_ts_attention_weights) > 0: # For debugging
#                 self.debug_ts_attn = torch.stack(all_ts_attention_weights, dim=1)

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

#             # ==================== Temporal Mixing ====================
#             seg_padding_mask = ~seg_valid
#             latent = self.tsmixer(latent, src_key_padding_mask=seg_padding_mask) # [B, L, 256]
#             latent = self.ln_latent(latent)

#         return latent, seg_valid


class TemporalMultiheadAttention_v2(nn.Module):
    """
    Modality-speicifc input을 받아 projection 후 MHA를 수행함.
    Latent Query는 256차원으로 고정함.
    """
    def __init__(self, d_model, num_heads, key_input_dim=None, value_input_dim=None, attn_dropout=0.1):
        super().__init__()
        self.d_model = d_model                          # Query dim
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attn_dropout = attn_dropout                # Attention dropout
        self.q_proj = nn.Linear(d_model, d_model)       # Query Projection

        k_in = key_input_dim if key_input_dim is not None else d_model
        v_in = value_input_dim if value_input_dim is not None else k_in

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


class TimeSeriesCentricCrossAttention_v4(nn.Module):
    def __init__(self, args, d_model=256, num_heads=8,
                ts_input_dim=512, img_input_dim=768, txt_input_dim=768,
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
        nn.init.uniform_(self.latent_init, -0.02, 0.02)

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
        time_emb_ts_raw = self.time2vec_ts(time_indices.unsqueeze(-1))  # [B, T, 768]
        time_emb_ts = self.ln_time_ts(time_emb_ts_raw)

        time_emb_img_raw = self.time2vec_img(time_indices.unsqueeze(-1))  # [B, T, 768]
        time_emb_img = self.ln_time_img(time_emb_img_raw)

        time_emb_txt_raw = self.time2vec_txt(time_indices.unsqueeze(-1))  # [B, T, 768]
        time_emb_txt = self.ln_time_txt(time_emb_txt_raw)

        latent = self.latent_init.expand(B, -1, -1)

        # 유효하지 않은 time step 마스킹.
        ts_key_padding_mask = None
        if seq_valid_mask is not None:
            ts_key_padding_mask = ~seq_valid_mask.bool()

        # T개 time step을 L개 구간으로 나눔.
        segments = build_hard_segments(T, L)

        # 각 segment가 유효한 데이터를 포함하는지 확인함.
        seg_valid = torch.zeros(B, L, device=ts_embeddings.device, dtype=torch.bool)
        if seq_valid_mask is not None:
            seq_mask_bool = seq_valid_mask.bool()
            for i, (s, e) in enumerate(segments):
                seg_valid[:, i] = seq_mask_bool[:, s:e].any(dim=1)
        else:
            seg_valid[:, :] = True

        # ================ Iterative Fusion ================
        for iter in range(num_iterations):
            self.ts_cross_attn.save_attn = (iter == 0) # 첫 iteration만 attention 저장함. (첫 에포크 첫 배치 시각화용)

            # ==================== TS -> Latent ====================
            latent_updates = []
            all_attention_weights = []
            ts_with_time = ts_embeddings + time_emb_ts

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
                img_with_time = img_embeddings + time_emb_img

                img_out = self.img_cross_attn(
                    query=latent,
                    key=img_with_time,
                    value=img_with_time,
                    key_padding_mask=img_key_padding_mask
                )
                latent = latent + img_out

            # ==================== Text -> Latent ====================
            if not self.disable_txt and text_embeddings is not None and text_embeddings.size(1) > 0:
                text_with_time = text_embeddings + time_emb_txt

                text_out = self.text_cross_attn(
                    query=latent,
                    key=text_with_time,
                    value=text_with_time,
                    key_padding_mask=text_key_padding_mask
                )
                latent = latent + text_out

            # ==================== Temporal Mixing ====================
            seg_padding_mask = ~seg_valid
            latent = self.tsmixer(latent, src_key_padding_mask=seg_padding_mask) # [B, L, 256]
            latent = self.ln_latent(latent)

        return latent, seg_valid


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