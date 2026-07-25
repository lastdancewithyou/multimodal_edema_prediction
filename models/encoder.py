import math

import torch
import torch.nn as nn
import torch.functional as F

from training.run import parse_arguments
from utils.utils import timer

from models.PatchTST_self_supervised.src.models.patchTST import PatchTSTEncoder
from models.PatchTST_self_supervised.src.callback.patch_mask import create_patch
from models.PatchTST_self_supervised.src.callback.patch_mask_icu import build_patch_attn_mask


# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x

# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean

# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         ) 

#     def forward(self, x):
#         return self.proj(x)


# # Temporal Mixing Block
# class TSMixerBlock(nn.Module):
#     """
#     - Time-mixing only block for temporal dependencies in embedded features.
#     - This uses only the time-mixing component from TS-Mixer to learn temporal dependencies across latents in cross-attention outputs.
#     """
#     def __init__(self, d_model=256, max_seq_len=25, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.max_seq_len = max_seq_len

#         # Time-mixing MLP (across timesteps)
#         self.time_mixing = nn.Sequential(
#             nn.LayerNorm(d_model),
#             Transpose(1, 2),  # [B, T, D] -> [B, D, T]
#             nn.Linear(max_seq_len, max_seq_len),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(max_seq_len, max_seq_len),
#             nn.Dropout(dropout),
#             Transpose(1, 2)   # [B, D, T] -> [B, T, D]
#         )

#     def forward(self, x, key_padding_mask=None):
#         """
#         Args:
#             x: [B, T, D] - Input embeddings from cross-attention
#             key_padding_mask: [B, T] - True for padding positions

#         Returns:
#             x: [B, T, D] - Output with temporal mixing applied
#         """
#         # Mask padding positions before time mixing
#         if key_padding_mask is not None:
#             # Set padding positions to 0
#             mask = ~key_padding_mask  # Invert: True for valid positions
#             x_masked = x * mask.unsqueeze(-1).float()
#         else:
#             x_masked = x

#         # Time mixing only
#         x = x + self.time_mixing(x_masked)

#         if key_padding_mask is not None:
#             x = x * mask.unsqueeze(-1).float()
#         return x

# class Transpose(nn.Module):
#     def __init__(self, dim1, dim2):
#         super().__init__()
#         self.dim1 = dim1
#         self.dim2 = dim2

#     def forward(self, x):
#         return x.transpose(self.dim1, self.dim2)

# class TSMixerEncoder(nn.Module):
#     def __init__(self, d_model=256, max_seq_len=25, num_layers=2, dropout=0.1):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             TSMixerBlock(d_model=d_model, max_seq_len=max_seq_len, dropout=dropout)
#             for _ in range(num_layers)
#         ])
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, x, src_key_padding_mask=None):
#         for layer in self.layers:
#             x = layer(x, key_padding_mask=src_key_padding_mask)

#         x = self.norm(x)
#         return x


# # Transformer
# class TransformerTSEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, window_size, num_layers=3, num_heads=8, dropout=0.1):
#         super().__init__()

#         self.input_projection = nn.Linear(input_size, hidden_size)
#         self.ln_output = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(dropout)

#         self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_len=window_size)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size,
#             nhead=num_heads,
#             dim_feedforward=hidden_size * 4,
#             dropout=dropout,
#             activation='gelu',
#             batch_first=True,
#             norm_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         x_proj = self.input_projection(x)
#         x_proj = self.pos_encoder(x_proj)
#         output = self.transformer_encoder(x_proj, src_key_padding_mask=None)
#         output = self.ln_output(output)
#         return self.dropout(output)


# class PositionalEncoding(nn.Module):
#     """
#     Positional encoding for transformer.
#     Adds position information to the input embeddings.
#     """
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         # Create positional encoding matrix
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)


class PatchTSTTSEncoder(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                window_size,
                patch_len=8,
                stride=4,
                d_model=128,
                n_heads=16,
                n_layers=3,
                d_ff=512,
                dropout=0.1,
                shared_embedding=True,
                pretrained_path=None,
                freeze_backbone=True,
                unfreeze_last_n=0,
                pad_threshold=0.5,
                var_pool_type='mlp'):
        super().__init__()

        self.input_size    = input_size
        self.window_size   = window_size
        self.patch_len     = patch_len
        self.stride        = stride
        self.d_model       = d_model
        self.pad_threshold = pad_threshold
        self.num_patch     = (max(window_size, patch_len) - patch_len) // stride + 1

        # PatchTST backbone (encoder only; PretrainHead is excluded)
        self.backbone = PatchTSTEncoder(
            c_in=input_size,
            num_patch=self.num_patch,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            shared_embedding=shared_embedding,
            d_ff=d_ff,
            attn_dropout=0.,
            dropout=dropout,
            act='relu',
            res_attention=False,
            pre_norm=False,
            store_attn=False,
        )

        # Load pretrained weights
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            if isinstance(sd, dict) and 'model' in sd:
                sd = sd['model']
            sd = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}

            # W_pos shape mismatch handling: multi-length 사전학습 ckpt는 max num_patch로 저장됨.
            # 현재 모델의 num_patch가 더 작아도 ckpt의 W_pos를 그대로 보관 후 forward에서 슬라이싱.
            if 'W_pos' in sd and sd['W_pos'].shape != self.backbone.W_pos.shape:
                ckpt_np = sd['W_pos'].shape[0]
                print(f'[PatchTST pretrained] W_pos resize: '
                      f'{tuple(self.backbone.W_pos.shape)} → {tuple(sd["W_pos"].shape)} '
                      f'(forward에서 num_patch={self.num_patch}로 슬라이싱)')
                self.backbone.W_pos = nn.Parameter(torch.zeros_like(sd['W_pos']))

            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f'[PatchTST pretrained] loaded {len(sd)} keys from {pretrained_path} '
                f'(missing={len(missing)}, unexpected={len(unexpected)})')

        # var_pool 제거: channel-mixing은 downstream의 ts_pre_proj가 담당.
        # 출력 dim = input_size * d_model (예: 21 * 128 = 2688)
        self.output_dim = input_size * d_model
        self.ln_output  = nn.LayerNorm(self.output_dim)
        self.dropout    = nn.Dropout(dropout)
        print(f"TS Encoder Output_dim={self.output_dim}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Partial unfreeze: 마지막 N개 TSTEncoderLayer만 학습 가능하게
            if unfreeze_last_n > 0:
                encoder_layers = self.backbone.encoder.layers
                n_total = len(encoder_layers)
                n_unfreeze = min(unfreeze_last_n, n_total)
                for layer in encoder_layers[n_total - n_unfreeze:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
                total     = sum(p.numel() for p in self.backbone.parameters())
                print(f"[PatchTSTTSEncoder] partial unfreeze: last {n_unfreeze}/{n_total} layers "
                      f"trainable ({trainable:,}/{total:,} params)")

    def forward(self, x, ts_valid_mask=None):
        """
        x             : [B, T, n_vars]
        ts_valid_mask : [B, T] (1 = real timestep, 0 = padded). None → all-valid.

        Returns:
            features  : [B, num_patch, hidden_size]
            patch_kpm : [B, num_patch] bool (True = padded patch) or None
                        — for downstream fusion to skip padded patches in attention.
        """
        B, _T, V = x.shape

        # Patching: [B, T, V] -> [B, P, V, L]
        z, num_patch = create_patch(x, self.patch_len, self.stride)

        # Slot-level mask → patch-level key_padding_mask
        patch_kpm = None
        if ts_valid_mask is not None:
            attn_mask_patch = build_patch_attn_mask(
                ts_valid_mask, self.patch_len, self.stride,
                num_patch, self.pad_threshold,
            )
            patch_kpm = ~attn_mask_patch.bool()

        # Backbone: [B, P, V, L] -> [B, V, d_model, P]
        z = self.backbone(z, key_padding_mask=patch_kpm)

        z = z.permute(0, 3, 1, 2).contiguous()
        z = z.reshape(B, num_patch, V * self.d_model)
        # var_pool 제거 — z는 [B, P, V*d_model] (예: [B, P, 2688]) 그대로

        return self.dropout(self.ln_output(z)), patch_kpm