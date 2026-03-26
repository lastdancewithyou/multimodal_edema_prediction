import math

import torch
import torch.nn as nn
import torch.functional as F
# from torch.nn.utils import weight_norm
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.utils.checkpoint import checkpoint

from training.run import parse_arguments
from utils.utils import timer


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
    """
    - Clinical prompt projection layer 추가: nn.Linear(768, hidden_size)
    - Prefix token으로 clinical prompt CLS embedding 추가
    - Positional encoding을 T+1 길이로 확장
    - Padding mask를 prompt prefix를 고려하여 조정
    - Output에서 prompt token 제거하여 원래 shape 유지
    ┌─────────────────────────────────────────────────┐
    │ [PROMPT] | HR_0 | BP_0 | ... | HR_23 | BP_23  │ ← Transformer Input
    └─────────────────────────────────────────────────┘
    """
    def __init__(self, input_size, hidden_size, window_size, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()

        self.window_size = window_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.ln_output = nn.LayerNorm(hidden_size)

        # Clinical prompt projection (768 -> hidden_size)
        self.prompt_projection = nn.Linear(768, hidden_size)
        self.ln_prompt = nn.LayerNorm(hidden_size)

        # Positional encoding (max_len = window_size + 1 for prompt prefix)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_len=window_size + 1)

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

    def forward(self, x, seq_valid_lengths, clinical_prompt=None):
        """
        Args:
            x: [B, T, D] - Time-series input
            seq_valid_lengths: [B] - Valid sequence lengths
            clinical_prompt: [B, 768] - Clinical prompt CLS embeddings

        Returns:
            output: [B, T, hidden_size] - Encoded time-series (prompt prefix removed)
        """
        B, T, _ = x.shape
        x_proj = self.input_projection(x) 

        if clinical_prompt is not None:
            prompt_token = self.prompt_projection(clinical_prompt)  
            prompt_token = self.ln_prompt(prompt_token)
            prompt_token = prompt_token.unsqueeze(1)

            # Concatenate prompt prefix: [prompt | TS_0, TS_1, ..., TS_T-1]
            x_proj = torch.cat([prompt_token, x_proj], dim=1)  # [B, T+1, hidden_size]

            # Positional encoding for T+1 tokens
            x_proj = self.pos_encoder(x_proj)

            # Update padding mask to account for prompt prefix
            mask = torch.zeros(B, T + 1, dtype=torch.bool, device=x.device)
            mask[:, 0] = False
            for i, valid_len in enumerate(seq_valid_lengths):
                if valid_len < T:
                    mask[i, valid_len + 1:] = True  # Shift by 1 for prompt prefix
        else:
            # no prompt
            x_proj = self.pos_encoder(x_proj)

            mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            for i, valid_len in enumerate(seq_valid_lengths):
                if valid_len < T:
                    mask[i, valid_len:] = True

        output = self.transformer_encoder(x_proj, src_key_padding_mask=mask) 

        # Remove prompt prefix from output
        if clinical_prompt is not None:
            output = output[:, 1:, :]  # remove first prompt token

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