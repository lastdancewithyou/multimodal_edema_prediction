import math

import torch
import torch.nn as nn
import torch.functional as F
# from torch.nn.utils import weight_norm
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.utils.checkpoint import checkpoint

from training.run import parse_arguments
from utils.utils import timer


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        ) 

    def forward(self, x):
        return self.proj(x)

class DLinear(nn.Module):
    """
    DLinear
    """
    def __init__(self, seq_len=24, enc_in=58, hidden_size=512, individual=False):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.individual = individual

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size=3)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.enc_in):
                # 입력 24시간 -> 출력 24시간으로 유지
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.seq_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.seq_len))
        else:
            # 모든 채널이 공유하는 선형 계층 (24 -> 24)
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.seq_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.seq_len)

        self.projection = nn.Linear(self.enc_in, hidden_size)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)

        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros_like(seasonal_init)
            trend_output = torch.zeros_like(trend_init)
            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x_temporal = (seasonal_output + trend_output).permute(0, 2, 1) # [Batch, seq_len, channels]
        return self.projection(x_temporal)


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
    def __init__(self, input_size, hidden_size, window_size, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.ln_output = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, x):
        x_proj = self.input_projection(x)
        x_proj = self.pos_encoder(x_proj)
        output = self.transformer_encoder(x_proj, src_key_padding_mask=None)
        output = self.ln_output(output)
        return self.dropout(output)


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