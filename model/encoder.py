import math

import torch
import torch.nn as nn
import torch.functional as F
# from torch.nn.utils import weight_norm
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.utils.checkpoint import checkpoint

from training.run import parse_arguments
from utils import timer


# Demographic modality encoder
class DemographicEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=256, dropout_rate=0.1):
        super().__init__()
        self.demo_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )

    def forward(self, demo_features):
        return self.demo_proj(demo_features.float())


# Temporal Mixing Block
class TSMixerBlock(nn.Module):
    """
    - Time-mixing only block for temporal dependencies in embedded features.
    - This uses only the time-mixing component from TS-Mixer to learn temporal dependencies across latents in cross-attention outputs.
    """
    def __init__(self, d_model=256, max_seq_len=25, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Time-mixing MLP (across timesteps)
        self.time_mixing = nn.Sequential(
            nn.LayerNorm(d_model),
            Transpose(1, 2),  # [B, T, D] -> [B, D, T]
            nn.Linear(max_seq_len, max_seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max_seq_len, max_seq_len),
            nn.Dropout(dropout),
            Transpose(1, 2)   # [B, D, T] -> [B, T, D]
        )

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [B, T, D] - Input embeddings from cross-attention
            key_padding_mask: [B, T] - True for padding positions

        Returns:
            x: [B, T, D] - Output with temporal mixing applied
        """
        # Mask padding positions before time mixing
        if key_padding_mask is not None:
            # Set padding positions to 0
            mask = ~key_padding_mask  # Invert: True for valid positions
            x_masked = x * mask.unsqueeze(-1).float()
        else:
            x_masked = x

        # Time mixing only
        x = x + self.time_mixing(x_masked)

        if key_padding_mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class TSMixerEncoder(nn.Module):
    def __init__(self, d_model=256, max_seq_len=25, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TSMixerBlock(d_model=d_model, max_seq_len=max_seq_len, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)

        x = self.norm(x)
        return x

# Transformer
class TransformerTSEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()

        self.window_size = window_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.ln_output = nn.LayerNorm(hidden_size)

        self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_len=window_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seq_valid_lengths):
        B, T, _ = x.shape # [B, T, D]        
        x_proj = self.input_projection(x) # Input projection
        x_proj = self.pos_encoder(x_proj) # Positional encoding

        # Create padding mask
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for i, valid_len in enumerate(seq_valid_lengths):
            if valid_len < T:
                mask[i, valid_len:] = True  # Mask positions after valid length

        output = self.transformer_encoder(x_proj, src_key_padding_mask=mask)  # Transformer encoder - [B, T, hidden_size]
        output = self.ln_output(output)
        output = self.dropout(output)
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    Adds position information to the input embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

# Grave of codes
# =============== Time-series Model comparison ===============
# GRU (commented out - now using Transformer)
# class GRU(nn.Module) :
#     def __init__(self, input_size, hidden_size, window_size, num_layers=2, dropout=0.1, bidirectional=False) :
#         super().__init__()
#
#         self.window_size = window_size
#         self.num_layers = num_layers
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bidirectional = bidirectional

#         self.dropout = nn.Dropout(dropout)
#         self.gru = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout,
#             bidirectional=bidirectional
#         )
#         self.layernorm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))

#     def forward(self, x, seq_valid_lengths):
#         x = x.float()
#         packed_x = pack_padded_sequence(x, seq_valid_lengths.cpu(), batch_first=True, enforce_sorted=False) # 시퀀스 내 패딩 처리
#         packed_output, _ = self.gru(packed_x)
#         output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=self.window_size) # total_length=self.window_size가 추가된 코드임임
#         output = self.layernorm(output)
#         output = self.dropout(output)
#         return output


# class VariableGatedAttention(nn.Module):
#     """
#     Variable-wise gating with observed mask soft weighting.

#     Args:
#         x: [B, T, D] - 원래 시계열 입력 (variable-wise feature)
#         observed_mask: [B, T, D] - 1=관측, 0=보간/결측

#     역할:
#     - x를 LayerNorm으로 안정화한 뒤,
#     - 값(x_norm)에 따라 변수별 gate를 만들고,
#     - observed_mask 기반 soft mask를 곱해 줌
#     (관측값: 1.0, 보간/결측값: missing_strength)
#     """

#     def __init__(self, num_vars, d_model, init_missing_strength=0.5):
#         super().__init__()
#         self.num_vars = num_vars
#         self.d_model = d_model

#         self.ln = nn.LayerNorm(d_model)
#         self.gate_linear = nn.Linear(d_model, d_model)

#         self.missing_strength = nn.Parameter(torch.ones(num_vars) * init_missing_strength)
#         self.missing_strength_act = nn.Sigmoid()

#         self.proj = nn.Linear(d_model, d_model)

#     def forward(self, x, observed_mask):
#         """
#         Args:
#             x: [B, T, D]
#             observed_mask: [B, T, D], 1=observed, 0=imputed
#         Returns:
#             x_out: [B, T, D]
#             gate_value: [B, T, D] (값 기반 gate)
#             gate_missing: [1, 1, D] (변수별 missing gate)
#         """
#         x_norm = self.ln(x)                     # [B, T, D]
#         gate_logits = self.gate_linear(x_norm)  # [B, T, D]
#         gate_value = torch.sigmoid(gate_logits)  # [B, T, D], 0~1

#         missing_strength = self.missing_strength_act(self.missing_strength)
#         gate_missing_base = missing_strength.view(1, 1, -1)
#         gate_missing = (gate_missing_base + (1.0 - gate_missing_base) * observed_mask)

#         gate_final = gate_value * gate_missing

#         x_gated = x * gate_final

#         x_out = self.proj(x_gated)
#         return x_out, gate_value, gate_missing_base

# Transformer-based Time Series Encoder
# class TransformerTSEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, window_size, num_layers=2, num_heads=8, dropout=0.1):
#         super().__init__()

#         self.window_size = window_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size

#         self.input_projection = nn.Linear(input_size, hidden_size)
#         self.ln_proj = nn.LayerNorm(hidden_size)
#         self.ln_output = nn.LayerNorm(hidden_size)

#         # Replace PositionalEncoding with Time2Vec for absolute time encoding
#         self.time_encoder = Time2VecEncoder(hidden_size)
#         self.ln_time = nn.LayerNorm(hidden_size)

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
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, seq_valid_lengths, time_indices=None):
#         """
#         Args:
#             x: [B, T, D] input time series
#             seq_valid_lengths: [B] valid sequence lengths
#             time_indices: [B, T] absolute time indices (ICU hours)
#         """
#         B, T, _ = x.shape # [B, T, D]

#         x_proj = self.input_projection(x)
#         x_proj = self.ln_proj(x_proj)

#         # Add absolute time encoding using Time2Vec
#         time_emb = self.time_encoder(time_indices.unsqueeze(-1))  # [B, T, hidden_size]
#         time_emb = self.ln_time(time_emb)
#         x_proj = x_proj + time_emb

#         # print(f"[TS Encoder] Time encoding applied!")
#         # print(f"  time_indices shape: {time_indices.shape}")
#         # print(f"  time_emb shape: {time_emb.shape}")
#         # print(f"  time_emb: mean={time_emb.mean().item():.4f}, std={time_emb.std().item():.4f}")
#         # print(f"  time_proj: mean={x_proj.mean().item():.4f}, std={x_proj.std().item():.4f}")

#         mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
#         for i, valid_len in enumerate(seq_valid_lengths):
#             if valid_len < T:
#                 mask[i, valid_len:] = True  # Mask positions after valid length

#         output = self.transformer_encoder(x_proj, src_key_padding_mask=mask)  # [B, T, hidden_size]

#         output = self.ln_output(output)
#         output = self.dropout(output)
#         return output


# class Time2VecEncoder(nn.Module):
#     """
#     Time2Vec encoding for absolute time representation.
#     Learns both linear and periodic patterns in time.
#     """
#     def __init__(self, d_model):
#         super().__init__()
#         self.d_model = d_model

#         # Linear component
#         self.linear = nn.Linear(1, d_model)

#         # Periodic components
#         self.w = nn.Parameter(torch.randn(1, d_model))
#         self.b = nn.Parameter(torch.randn(1, d_model))

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
#         nn.init.constant_(self.linear.bias, 0.0)
#         nn.init.uniform_(self.w, -0.1, 0.1)
#         nn.init.uniform_(self.b, -0.1, 0.1)

#     def forward(self, t):
#         """
#         Args:
#             t: [B, T, 1] - Absolute time indices
#         Returns:
#             time_emb: [B, T, d_model]
#         """
#         t_lin = self.linear(t)  # Linear trend
#         t_periodic = torch.sin(t * self.w + self.b)  # Periodic patterns
#         time_emb = t_lin + t_periodic
#         return time_emb


# class PositionalEncoding(nn.Module):
#     """
#     Positional encoding for transformer (DEPRECATED - use Time2VecEncoder instead).
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


# QKV Adapter for Cross-attention
# class QKVAdapter(nn.Module):
#     def __init__(self, input_size, d_model=256):
#         super().__init__()
#         self.proj = nn.Linear(input_size, d_model)
#         self.ln = nn.LayerNorm(d_model)

#     def forward(self, x):
#         x = self.proj(x)
#         x = self.ln(x)
#         return x