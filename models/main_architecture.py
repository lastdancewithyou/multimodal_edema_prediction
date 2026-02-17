import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from transformers import AutoModel

from models.encoder import TransformerTSEncoder, DemographicEncoder, TSMixerEncoder
from utils import timer


class MultiModalEncoder(nn.Module):
    def __init__(self, args, disable_cxr=False, disable_txt=False):
        super().__init__()

        self.disable_cxr = disable_cxr
        self.disable_txt = disable_txt

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

        # Image Encoder: DenseNet121 from torchxrayvision (pretrained on MIMIC-CXR)
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
        _img_trainable = sum(p.numel() for p in self.img_encoder.parameters() if p.requires_grad)
        _img_total     = sum(p.numel() for p in self.img_encoder.parameters())
        print(f"[MultiModalEncoder] DenseNet121: {_img_trainable:,} / {_img_total:,} params trainable")

        # Text Encoder: Bio_ClinicalBERT
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.language_model = AutoModel.from_pretrained(model_name)
        for param in self.language_model.parameters():
            param.requires_grad = False

        _bert_unfreeze = [
            "encoder.layer.10",
            "encoder.layer.11",
            "pooler",
        ]
        for name, param in self.language_model.named_parameters():
            if any(name.startswith(prefix) for prefix in _bert_unfreeze):
                param.requires_grad = True
        _txt_trainable = sum(p.numel() for p in self.language_model.parameters() if p.requires_grad)
        _txt_total     = sum(p.numel() for p in self.language_model.parameters())
        print(f"[MultiModalEncoder] BioClinicalBERT: {_txt_trainable:,} / {_txt_total:,} params trainable")

        # Demographic Encoder
        num_demo_features = args.num_demo_features
        self.demo_encoder = DemographicEncoder(
            input_dim=num_demo_features,
            output_dim=256,
            dropout_rate=0.1
        )
        print(f"[MultiModalEncoder] DemographicEncoder initialized with {num_demo_features} input features")

        # TS-Centric Fusion Module
        self.ts_centric_fusion = TimeSeriesCentricCrossAttention_v4(
            args=args,
            d_model=256,
            num_heads=8,
            ts_input_dim=512,       # TS encoder output dim
            img_input_dim=1024,     # DenseNet121 feature dim
            txt_input_dim=768,      # BioClinicalBERT hidden size
            cxr_dropout=0.1,
            text_dropout=0.1,
        )

        # Attention Pooling
        self.attention_pooling = AttentionPooling(input_dim=256)

    def forward(self, args, ts_series, cxr_data, text_data, has_cxr, has_text,
                window_mask, seq_valid_mask, demo_features=None, time_steps=None):
        """
        Forward pass through all modality encoders, fusion, and attention pooling to get embedding for contrastive learning & CLassification.

        Returns:
            window_embeddings: [B, W, 256] - Pooled window-level embeddings
            window_mask: [B, W] - Window validity mask
        """
        device = ts_series.device
        B, W, T, D = ts_series.shape

        # Modality-specific encoding (Concept of Projection)
        # ================ Time-series Encoding ================
        with timer("TS Encoder", None):
            ts_series_flat = ts_series.view(B * W, T, D)
            seq_valid_mask_flat = seq_valid_mask.view(B * W, T)         # 윈도우 내 유효한 time step을 masking
            seq_valid_lengths_flat = seq_valid_mask_flat.sum(dim=-1)   
            window_mask_flat = window_mask.reshape(B * W)               # 패딩되지 않은 유효한 window 선별

            valid_indices = window_mask_flat.nonzero(as_tuple=False).squeeze(1)     # 유효한 window 인덱스 추출
            valid_ts_series = ts_series_flat[valid_indices]                         # 유효한 window만 선택
            valid_seq_lengths = seq_valid_lengths_flat[valid_indices]

            valid_ts_encoded = self.ts_encoder(valid_ts_series, valid_seq_lengths)

            # 고정된 zero matrix를 만든 후, 유효한 window만 처리할 수 있도록 처리함.
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
        """
        if not self.disable_cxr:
            with timer("IMG Encoder", None):
                img_tensor = torch.zeros(B, W, T, 1024, device=device, dtype=ts_embeddings.dtype) # 처음에는 모두 0으로 초기화
                has_img = torch.zeros(B, W, T, device=device, dtype=torch.bool)

                unique_images = cxr_data['unique_images']       # unique한 이미지만
                unique_indices = cxr_data['unique_indices']     # 각 위치가 어떤 unique image인지
                pos = cxr_data['positions']                     # (batch, window, timestep)

                if unique_images.numel() > 0:
                    unique_features = self.img_encoder.features(unique_images)  # [N_unique, 1024, 7, 7]
                    unique_features = F.adaptive_avg_pool2d(unique_features, (1, 1)).flatten(1)  # [N_unique, 1024]
                    scattered = unique_features[unique_indices]
                    scattered = scattered.to(dtype=ts_embeddings.dtype)

                    b, w, t = pos[:, 0].long(), pos[:, 1].long(), pos[:, 2].long()
                    img_tensor[b, w, t] = scattered     # 원래 위치 (b, w, t)에 이미지 임베딩 넣기
                    has_img[b, w, t] = True             # 이미지 존재 여부 마스킹

                img_embeddings = img_tensor

        # Turn off Image modality (For ablation study)
        else:
            img_embeddings = torch.zeros(B, W, T, 1024, device=device, dtype=ts_embeddings.dtype)
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
                    outputs = self.language_model(unique_input_ids, attention_mask=unique_attention_mask)
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
            img_embeddings: [B, W, T, 1024] - sparse
            text_embeddings: [B, W, T, 768] - sparse
            """
            BW = B * W
            ts_flat = ts_embeddings.reshape(BW, T, 512)
            seq_mask_flat = seq_valid_mask.reshape(BW, T).float()       # 실제 데이터가 들어있는 time step을 뽑아냄.
            ts_flat_masked = ts_flat * seq_mask_flat.unsqueeze(-1)

            img_flat = img_embeddings.reshape(BW, T, 1024)
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
                    img_embeddings=img_kv,                      # [Nwin, T, 1024]
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


##################################################################################################
# Stage 1: Representation Learning Model
##################################################################################################
class MultiModalContrastiveModel(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.encoder = encoder
        self.pooled_augmenter = AugmentationModule(
            noise_type=args.aug_noise_type,
            epsilon=args.aug_epsilon,
            num_views=2,
        )

        # 일단 단일 projection head를 사용함.
        self.projection_head = ProjectionHead(args)

        print(f"[MultiModalContrastiveModel] ✅ Stage 1 model initialized")
        print(f"   - Augmentation: {args.aug_noise_type}, epsilon={args.aug_epsilon}")

    def forward(self, args, ts_series, cxr_data, text_data, has_cxr, has_text,
                window_mask, seq_valid_mask, demo_features=None, time_steps=None):
        """
        Forward pass for Stage 1 contrastive(Representation) learning.

        Flow:
            1. Encoder returns [B, W, 256] embeddings.
            2. Augment pooled embeddings → [Nwin, 2, 256] # temporary position
            3. Project each view to different projection head → [Nwin, 2, proj_dim]

        Returns:
            projected_embeddings_multiview: [Nwin, 2, proj_dim=128] - For Contrastive Loss
        """
        B, W = window_mask.shape

        # ================ Extracting embeddings using a pre-made encoder ================
        window_embeddings, _ = self.encoder(
            args, ts_series, cxr_data, text_data, has_cxr, has_text,
            window_mask, seq_valid_mask, demo_features, time_steps
        )

        # ================ Extract valid windows ================
        """
        - 환자마다 ICU stay가 다르고, padding 된 window가 존재함.
        - padding된 window에 augmentation과 projection을 적용해서 낭비를 막기 위함임.
        """
        BW = B * W
        window_embeddings_flat = window_embeddings.reshape(BW, 256)
        window_valid_mask = window_mask.reshape(BW).bool()
        pooled_base = window_embeddings_flat[window_valid_mask]  # [Nwin, 256]

        # ================ Augment pooled embeddings (2 views) ================
        """
        - 적절한 증강 위치를 찾다가 일시적으로 이 곳에 증강을 하였는데, 일단 modality-specific embedding에 적용하는 것을 최우선으로 고려함.
        """
        augmented_pooled = self.pooled_augmenter(pooled_base) # [Nwin, 256] → [Nwin, 2, 256]

        # ================ Project each view ================
        # Use separate projection heads for each view
        view_0 = augmented_pooled[:, 0, :]  # [Nwin, 256]
        view_1 = augmented_pooled[:, 1, :]  # [Nwin, 256]

        proj_0 = self.projection_head(view_0)  # [Nwin, proj_dim]
        proj_1 = self.projection_head(view_1)  # [Nwin, proj_dim]

        projected_embeddings_multiview = torch.stack([proj_0, proj_1], dim=1)

        return projected_embeddings_multiview


##################################################################################################
# Stage 2: Classification Model
##################################################################################################

class MultiModalClassificationModel(nn.Module):
    """
    Stage 2 Model: Pretrained Encoder + Linear Classifier

    - Used for classification with cross-entropy loss.
    - Freeze the encoder to perform linear probing, or apply cross-entropy-based fine-tuning.

    Architecture:
        Input → Encoder (frozen/fine-tuned) → [B, W, 256]
              → Linear Classifier → [B, W, num_classes]
              → CE Loss

    Returns:
        - window_embeddings: [B, W, 256] Pooled embeddings
        - logits: [B, W, num_classes] Classification logits
    """

    def __init__(self, encoder, args, freeze_encoder=False):
        super().__init__()

        # Pretrained encoder from Stage 1
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[MultiModalClassificationModel] 🧊 Encoder frozen (Linear probing mode)")
        else:
            print("[MultiModalClassificationModel] 🔥 Encoder trainable (Fine-tuning mode)")

        # Linear classifier
        self.classifier = SingleClassifier(input_dim=256, num_classes=args.num_classes)
        for param in self.classifier.parameters():
            param.requires_grad = True # Classifier는 두 mode에서 모두 학습 대상임.

        print(f"[MultiModalClassificationModel] ✅ Stage 2 model initialized")
        print(f"   - Encoder frozen: {freeze_encoder}")

    def forward(self, args, ts_series, cxr_data, text_data, has_cxr, has_text,
                window_mask, seq_valid_mask, demo_features=None, time_steps=None):
        """
        Forward pass for Stage 2 classification.

        Flow:
            1. Encoder → [B, W, 256]
            2. Linear Classifier → [B, W, num_classes]

        Returns:
            window_embeddings: [B, W, 256] - Pooled embeddings
            logits: [B, W, num_classes] - Classification logits
        """
        window_embeddings, _ = self.encoder(
            args, ts_series, cxr_data, text_data, has_cxr, has_text,
            window_mask, seq_valid_mask, demo_features, time_steps
        )

        # ================ Classification ================
        logits = self.classifier(window_embeddings)  # [B, W, num_classes]

        return window_embeddings, logits


class AttentionPooling(nn.Module):
    """
    L개의 latent를 하나의 window embedding으로 압축함.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attn_fc = nn.Linear(input_dim, 1) # attention score 계산 (=중요도 개념)

    def forward(self, latent_emb, seg_valid_mask=None):
        attn_scores = self.attn_fc(latent_emb).squeeze(-1)      # [N, L]

        # 유효하지 않은 latent는 attention에서 제외함.
        if seg_valid_mask is not None:
            attn_scores = attn_scores.masked_fill(~seg_valid_mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=1)                      # [N, L]
        weighted_emb = (latent_emb * attn_weights.unsqueeze(-1)).sum(dim=1)   # [N, D]
        return weighted_emb


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


class SingleClassifier(nn.Module):
    """
    Simple Linear classifier
    """
    def __init__(self, input_dim, num_classes, 
        ):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.linear(x)


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


class TimeSeriesCentricCrossAttention_v4(nn.Module):
    def __init__(self, args, d_model=256, num_heads=8,
                ts_input_dim=512, img_input_dim=1024, txt_input_dim=768,
                cxr_dropout=0.1, text_dropout=0.1
        ):
        super().__init__()
        self.d_model = d_model                      # latent embedding dimension
        self.num_heads = num_heads                  # Multi-head attention head 개수
        self.num_latents = args.num_latents         # Latent array query 개수
        # dropout = max(cxr_dropout, text_dropout)    # 현재는 dropout 일괄 적용

        # Latent embeddings
        self.latent_init = nn.Parameter(torch.empty(1, self.num_latents, d_model))
        nn.init.uniform_(self.latent_init, -0.02, 0.02)

        # Cross-attention modules with modality-specific input dimensions
        self.ts_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=ts_input_dim,
        )
        self.img_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=img_input_dim,
        )
        self.text_cross_attn = TemporalMultiheadAttention_v2(
            d_model, num_heads, key_input_dim=txt_input_dim,
        )

        # latent 간 정보 교환
        self.tsmixer = TSMixerEncoder(
            d_model=d_model,
            max_seq_len=self.num_latents,
            num_layers=2,
            # dropout=dropout
        )

        # Modality-specific Time2Vec for time encoding (시계열에는 넣어야 할지 말지 확정하지 못함)
        self.time2vec_ts = Time2Vec(ts_input_dim)
        self.time2vec_img = Time2Vec(img_input_dim) 
        self.time2vec_txt = Time2Vec(txt_input_dim) 

        self.ln_time_ts = nn.LayerNorm(ts_input_dim)
        self.ln_time_img = nn.LayerNorm(img_input_dim)
        self.ln_time_txt = nn.LayerNorm(txt_input_dim)
        # self.ln_latent = nn.LayerNorm(d_model)

        self.debug_ts_attn = None
        # self.residual_dropout = nn.Dropout(dropout)

    def forward(
            self, ts_embeddings, img_embeddings=None, text_embeddings=None, time_indices=None,
            img_key_padding_mask=None, text_key_padding_mask=None, seq_valid_mask=None,
            num_iterations=2
        ):

        B, T, _ = ts_embeddings.shape
        L = self.num_latents

        # ================ Time emb add to TS, img, Text modality after projection ================
        time_emb_ts_raw = self.time2vec_ts(time_indices.unsqueeze(-1))
        time_emb_ts = self.ln_time_ts(time_emb_ts_raw)

        time_emb_img_raw = self.time2vec_img(time_indices.unsqueeze(-1))  # [B, T, 1024]
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
            # ts_with_time = ts_embeddings # time emb 사용하지 않는 경우

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
            # latent = latent + self.residual_dropout(ts_out)

            if len(all_attention_weights) > 0: # For debugging
                self.debug_ts_attn = torch.stack(all_attention_weights, dim=1)

            # ==================== IMG -> Latent ====================
            if img_embeddings is not None and img_embeddings.size(1) > 0:
                img_with_time = img_embeddings + time_emb_img
                # img_with_time = img_embeddings # time emb 사용하지 않는 경우

                img_out = self.img_cross_attn(
                    query=latent,
                    key=img_with_time,
                    value=img_with_time,
                    key_padding_mask=img_key_padding_mask
                )
                latent = latent + img_out
                # latent = latent + self.residual_dropout(img_out)

            # ==================== Text -> Latent ====================
            if text_embeddings is not None and text_embeddings.size(1) > 0:
                text_with_time = text_embeddings + time_emb_txt
                # text_with_time = text_embeddings # time emb 사용하지 않는 경우

                text_out = self.text_cross_attn(
                    query=latent,
                    key=text_with_time,
                    value=text_with_time,
                    key_padding_mask=text_key_padding_mask
                )
                latent = latent + text_out
                # latent = latent + self.residual_dropout(text_out)

            # Temporal Mixing
            seg_padding_mask = ~seg_valid
            latent = self.tsmixer(latent, src_key_padding_mask=seg_padding_mask) # [B, L, 256]
            # latent = self.ln_latent(latent)

        return latent, seg_valid


class TemporalMultiheadAttention_v2(nn.Module):
    """
    Modality-speicifc input을 받아 projection 후 MHA를 수행함.
    Latent Query는 256차원으로 고정함.
    """
    def __init__(self, d_model, num_heads, key_input_dim=None, value_input_dim=None):
        super().__init__()
        self.d_model = d_model                          # Query dim
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
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
            is_causal=False
        )

        # MHA 연산 합치기 [B, H, T_q, d_k] → [B, T_q, D]
        out = out.transpose(1, 2).reshape(B, T_q, D)
        out = self.out_proj(out)
        return out


class AugmentationModule(nn.Module):
    def __init__(self, noise_type='gaussian', epsilon=0.1, num_views=2):
        super().__init__()
        self.noise_type = noise_type
        self.epsilon = epsilon
        self.num_views = num_views

        valid_noise_types = ['gaussian', 'uniform', 'poisson', 'laplace']
        if noise_type not in valid_noise_types:
            raise ValueError(f"증강 방법 선택 과정에서 오타 발생함.")
        print(f"[ModalityAugmentation] Initialized with noise_type={noise_type}, epsilon={epsilon}, num_views={num_views}")

    def _sample_noise(self, shape, device):
        # [N, 256] → [N, 2, 256]
        if self.noise_type == 'gaussian':
            omega = torch.randn(shape, device=device)

        elif self.noise_type == 'uniform':
            omega = 2 * torch.rand(shape, device=device) - 1

        elif self.noise_type == 'poisson':
            omega = torch.poisson(torch.ones(shape, device=device)) - 1.0

        elif self.noise_type == 'laplace':
            u = torch.rand(shape, device=device) - 0.5
            omega = torch.sign(u) * torch.log(1 - 2 * u.abs())
        return omega

    def forward(self, embeddings):
        """
        1. Modality embedding에 증강을 적용할 것이냐. - 환자의 원래 상태를 가장 잘 보존할 것이기 때문에 적절한 방법론일 것이라고 고려 중임.
        2. Latent level에 증강을 적용할 것이냐. - 이 경우 attention pooling 전에 적용해야 하기 때문에 모델 구조 변경 필요함.
        3. Window level에 증강을 적용할 것이냐. - window level에 적용하는 것이 가장 간단해서 현재 시범 적용 중임.
        """
        device = embeddings.device

        # Latent level augmentation
        if embeddings.dim() == 3:
            N, L, D = embeddings.shape

            embeddings_flat = embeddings.reshape(N * L, D) # [N, L, D] → [N*L, D]
            embeddings_norm = F.normalize(embeddings_flat, p=2, dim=-1)

            # Generate independent augmented views
            augmented_list = []
            for _ in range(self.num_views):
                omega = self._sample_noise(embeddings_norm.shape, device)
                delta = omega * torch.sign(embeddings_norm)
                delta_norm = F.normalize(delta, p=2, dim=-1) * self.epsilon
                z_aug = embeddings_norm + delta_norm
                z_aug = F.normalize(z_aug, p=2, dim=-1)
                augmented_list.append(z_aug)

            augmented_views = torch.stack(augmented_list, dim=1)  # [N*L, V, D]
            augmented_views = augmented_views.reshape(N, L, self.num_views, D)  # [N, L, V, D]

        # Window level augmentation
        elif embeddings.dim() == 2:
            N, D = embeddings.shape

            embeddings_norm = F.normalize(embeddings, p=2, dim=-1)  # [N, D]

            augmented_list = []
            for _ in range(self.num_views):
                omega = self._sample_noise(embeddings_norm.shape, device)  # [N, D]
                delta = omega * torch.sign(embeddings_norm)
                delta_norm = F.normalize(delta, p=2, dim=-1) * self.epsilon
                z_aug = embeddings_norm + delta_norm
                z_aug = F.normalize(z_aug, p=2, dim=-1)
                augmented_list.append(z_aug)

            augmented_views = torch.stack(augmented_list, dim=1)  # [N, V, D]

        else:
            raise ValueError(f"Expected 2D or 3D input, but embedding shape was {embeddings.shape}.")

        return augmented_views


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