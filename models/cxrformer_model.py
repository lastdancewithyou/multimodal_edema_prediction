import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import trunc_normal_
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# LayerScale
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, C, H, W) -> (B, N, D)."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)          # B, D, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, D
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Self-Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Cross-Attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Cross-attention: queries from image tokens (x), keys/values from text tokens (context).

    Args:
        dim:         image token dimension
        context_dim: text token dimension (defaults to dim)
        num_heads:   number of attention heads
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        context_dim = context_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x:       image tokens  [B, N, dim]
            context: text tokens   [B, M, context_dim]
        Returns:
            [B, N, dim]
        """
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)       # [B, h, N, d]
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                                   # [B, h, M, d]

        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)  # [B, h, N, M]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Stochastic depth helper
# ---------------------------------------------------------------------------

def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]
    x_subset = x[brange]

    residual = residual_func(x_subset).flatten(1)
    residual_scale_factor = b / sample_subset_size

    x_plus_residual = torch.index_add(
        x.flatten(1), 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


# ---------------------------------------------------------------------------
# Block (self-attention only)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(x, attn_residual_func, self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(x, ffn_residual_func, self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


# ---------------------------------------------------------------------------
# CrossAttentionBlock (self-attention + cross-attention + FFN)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(Block):
    """
    Block with an additional cross-attention sublayer between self-attention and FFN:

        Self-Attn → Cross-Attn → FFN

    All self-attention and FFN weights are identical in structure to Block,
    so pretrained weights load cleanly with strict=False.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.norm_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls_cross = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path_cross = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:       image tokens  [B, N, dim]
            context: text tokens   [B, M, context_dim] (optional)
                     If None, only self-attention and FFN are performed (cross-attention is skipped)
        Returns:
            [B, N, dim]
        """
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        # Self-attention + FFN
        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(x, attn_residual_func, self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
        else:
            x = x + attn_residual_func(x)

        # Cross-attention (only if context is provided)
        if context is not None:
            def cross_attn_residual_func(x: Tensor) -> Tensor:
                return self.ls_cross(self.cross_attn(self.norm_cross(x), context))

            if self.training and self.sample_drop_ratio > 0.0:
                x = x + self.drop_path_cross(cross_attn_residual_func(x))
            else:
                x = x + cross_attn_residual_func(x)

        # FFN
        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(x, ffn_residual_func, self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + ffn_residual_func(x)

        return x


# ---------------------------------------------------------------------------
# CXformer: ViT-Base backbone + cross-attention in the last 6 blocks
# ---------------------------------------------------------------------------

class CXformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        context_dim: int = 512,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 1.0,
        num_register_tokens: int = 4,
        n_cross_attn_blocks: int = 6,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_cross_attn_blocks = n_cross_attn_blocks
        self.num_register_tokens = num_register_tokens
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Register tokens
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))

        # Stochastic depth decay (linear schedule)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Block 공통 kwargs
        block_kwargs = dict(
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            norm_layer=norm_layer,
        )

        n_self_attn = depth - n_cross_attn_blocks  # 6
        blocks = []
        for i in range(depth):
            dp = dpr[i]
            if i < n_self_attn:
                blocks.append(Block(dim=embed_dim, drop_path=dp, **block_kwargs))
            else:
                blocks.append(CrossAttentionBlock(dim=embed_dim, context_dim=context_dim, drop_path=dp, **block_kwargs))
        self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(embed_dim)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        self._init_weights()

    def _init_weights(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w // self.patch_size, h // self.patch_size
        M = int(math.sqrt(N))
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            size=(w0, h0),
            mode="bicubic",
            antialias=True,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(x.dtype)

    def prepare_tokens(self, x: Tensor) -> Tensor:
        B, _, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        # register tokens을 CLS 바로 뒤, patch tokens 앞에 삽입
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((x[:, :1], reg_tokens, x[:, 1:]), dim=1)
        return x

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> dict:
        """
        Args:
            x:       images   [B, C, H, W]
            context: text embeddings  [B, M, context_dim] (optional)
                     If None, only self-attention is performed (cross-attention is skipped)

        Returns:
            dict with:
                x_norm_clstoken:   [B, embed_dim]   CLS token (for classification)
                x_norm_patchtokens:[B, N, embed_dim] patch tokens (register tokens 제외)
        """
        x = self.prepare_tokens(x)

        n_self_attn = self.depth - self.n_cross_attn_blocks

        if context is None:
            # context가 None이면 self-attention만 수행
            for blk in self.blocks:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        else:
            # context가 있으면 원래대로 self-attn + cross-attn 수행
            for i, blk in enumerate(self.blocks):
                if i < n_self_attn:
                    if self.gradient_checkpointing and self.training:
                        x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                    else:
                        x = blk(x)
                else:
                    if self.gradient_checkpointing and self.training:
                        x = torch.utils.checkpoint.checkpoint(blk, x, context, use_reentrant=False)
                    else:
                        x = blk(x, context)

        x = self.norm(x)
        return {
            "x_norm_clstoken": x[:, 0],
            # register tokens(1 ~ 1+num_register_tokens)은 제외
            "x_norm_patchtokens": x[:, 1 + self.num_register_tokens:],
        }

    @staticmethod
    def _remap_checkpoint(sd: dict) -> dict:
        new_sd = {}
        # Q, K, V 를 블록별로 모아서 나중에 cat
        qkv_buf: dict = {}  # {i: {"query": {"weight": t, "bias": t}, ...}}

        for k, v in sd.items():
            # mask_token 은 우리 모델에 없으므로 제거
            if k == "embeddings.mask_token":
                continue

            # ── embeddings ────────────────────────────────────────────────
            if k == "embeddings.cls_token":
                new_sd["cls_token"] = v
            elif k == "embeddings.position_embeddings":
                # 체크포인트 pos_embed(518px 기준 37×37+1=1370)를
                # 현재 모델(224px 기준 16×16+1=257)로 bicubic interpolation
                cls_pe = v[:, :1, :]       # [1, 1, 768]  CLS 위치 임베딩
                patch_pe = v[:, 1:, :]     # [1, 1369, 768]  패치 위치 임베딩
                src_size = int(patch_pe.shape[1] ** 0.5)   # 37
                tgt_size = 16                               # 224 // 14
                patch_pe = (
                    patch_pe
                    .reshape(1, src_size, src_size, -1)
                    .permute(0, 3, 1, 2)                   # [1, 768, 37, 37]
                )
                patch_pe = torch.nn.functional.interpolate(
                    patch_pe.float(),
                    size=(tgt_size, tgt_size),
                    mode="bicubic",
                    antialias=True,
                ).to(v.dtype)
                patch_pe = (
                    patch_pe
                    .permute(0, 2, 3, 1)                   # [1, 16, 16, 768]
                    .reshape(1, tgt_size * tgt_size, -1)   # [1, 256, 768]
                )
                new_sd["pos_embed"] = torch.cat([cls_pe, patch_pe], dim=1)  # [1, 257, 768]
            elif k == "embeddings.register_tokens":
                new_sd["register_tokens"] = v
            elif k.startswith("embeddings.patch_embeddings.projection."):
                suffix = k[len("embeddings.patch_embeddings.projection."):]
                new_sd[f"patch_embed.proj.{suffix}"] = v

            # ── final layernorm ───────────────────────────────────────────
            elif k.startswith("layernorm."):
                suffix = k[len("layernorm."):]
                new_sd[f"norm.{suffix}"] = v

            # ── encoder blocks ────────────────────────────────────────────
            elif k.startswith("encoder.layer."):
                # k 예시: "encoder.layer.0.norm1.weight"
                rest = k[len("encoder.layer."):]           # "0.norm1.weight"
                idx_str, _, tail = rest.partition(".")     # "0", "norm1.weight"
                i = int(idx_str)

                if tail.startswith("norm1.") or tail.startswith("norm2."):
                    new_sd[f"blocks.{i}.{tail}"] = v
                elif tail == "layer_scale1.lambda1":
                    new_sd[f"blocks.{i}.ls1.gamma"] = v
                elif tail == "layer_scale2.lambda1":
                    new_sd[f"blocks.{i}.ls2.gamma"] = v
                elif tail.startswith("mlp."):
                    new_sd[f"blocks.{i}.{tail}"] = v
                elif tail.startswith("attention.output.dense."):
                    suffix = tail[len("attention.output.dense."):]
                    new_sd[f"blocks.{i}.attn.proj.{suffix}"] = v
                elif tail.startswith("attention.attention."):
                    # "attention.attention.query.weight" → qkv_type="query", param="weight"
                    attn_rest = tail[len("attention.attention."):]
                    qkv_type, _, param = attn_rest.partition(".")
                    qkv_buf.setdefault(i, {}).setdefault(qkv_type, {})[param] = v

        # Q, K, V weight/bias 를 순서대로 cat → qkv
        for i, parts in qkv_buf.items():
            for param in ("weight", "bias"):
                tensors = [
                    parts[t][param]
                    for t in ("query", "key", "value")
                    if param in parts.get(t, {})
                ]
                if tensors:
                    new_sd[f"blocks.{i}.attn.qkv.{param}"] = torch.cat(tensors, dim=0)

        return new_sd

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "m42-health/CXformer-base",
        local_path: Optional[str] = None,
        **kwargs,
    ) -> "CXformer":
        """
        Instantiate CXformer and load matching weights from HuggingFace or a local file.

        Cross-attention layers (norm_cross, cross_attn, ls_cross, drop_path_cross)
        are NOT present in the pretrained checkpoint and will remain randomly
        initialised — they are the parameters to be fine-tuned.

        Args:
            model_id:   HuggingFace repo ID (used when local_path is None).
                        Weights are cached at ~/.cache/huggingface/ after the first
                        download, so subsequent calls load from cache without
                        re-downloading.
            local_path: Optional path to a local .bin or .safetensors file.
                        When provided, model_id is ignored entirely — no network
                        access occurs.
            **kwargs:   CXformer.__init__ arguments (e.g. context_dim=512).

        Examples:
            # HuggingFace (cached after first call):
            model = CXformer.from_pretrained("m42-health/CXformer-base", context_dim=512)

            # Local file (no network):
            model = CXformer.from_pretrained(local_path="/path/to/model.bin", context_dim=512)
        """
        model = cls(**kwargs)

        if local_path is not None:
            # ── Local file ────────────────────────────────────────────────
            if local_path.endswith(".safetensors"):
                state_dict = load_file(local_path)
            else:
                state_dict = torch.load(local_path, map_location="cpu")
            source = local_path
        else:
            # ── HuggingFace Hub (cached) ───────────────────────────────────
            # hf_hub_download stores files in ~/.cache/huggingface/hub/.
            # On repeated calls it returns the cached path immediately without
            # making a network request (unless the remote file has changed).
            ckpt_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
            state_dict = load_file(ckpt_path)
            source = model_id

        state_dict = CXformer._remap_checkpoint(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        cross_attn_keys = [k for k in missing if any(
            tag in k for tag in ["cross_attn", "norm_cross", "ls_cross", "drop_path_cross"]
        )]
        other_missing = [k for k in missing if k not in cross_attn_keys]

        print(f"[CXformer] Loaded pretrained weights from '{source}'")
        print(f"  Cross-attn keys (will be fine-tuned): {len(cross_attn_keys)}")
        if other_missing:
            print(f"  Other missing keys: {other_missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        return model


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = alpha / r

        in_dim = linear.in_features
        out_dim = linear.out_features

        self.lora_A = nn.Parameter(torch.empty(in_dim, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_dim))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = (self.dropout(x @ self.lora_A)) @ self.lora_B
        return base + lora * self.scaling

    def extra_repr(self) -> str:
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"r={self.r}, scaling={self.scaling:.3f}")


def apply_lora_to_cxformer(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_keys: tuple = ("attn.qkv", "attn.proj"),  # Cross-attention 제외 (랜덤 초기화 상태이므로 full fine-tuning)
) -> nn.Module:

    # Step 1: 전체 freeze
    for param in model.parameters():
        param.requires_grad_(False)

    # Step 2: 교체 대상을 먼저 수집 (순회 중 교체 방지)
    lora_targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(k in name for k in target_keys):
            lora_targets.append(name)

    # Step 3: LoRA 적용 (Self-attention만)
    replaced = set()
    for name in lora_targets:
        if name in replaced:
            continue

        parent = model
        *path, last = name.split('.')
        for p in path:
            parent = getattr(parent, p)

        module = getattr(parent, last)

        # 이미 LoRALinear인 경우 스킵 (혹시 모를 중복 방지)
        if isinstance(module, LoRALinear):
            continue

        setattr(parent, last, LoRALinear(module, r, alpha, dropout))
        replaced.add(name)

    # Step 4: Cross-attention layers는 full fine-tuning (unfreeze)
    cross_attn_params = 0
    cross_attn_layers = []
    for name, param in model.named_parameters():
        if "cross_attn" in name:
            param.requires_grad_(True)
            cross_attn_params += param.numel()
            cross_attn_layers.append(name)

    ####################################################################################
    # # Step 5: Unfreeze last 2 layers' FFN and LayerNorms for task-specific adaptation
    # last2_params = 0
    # last2_layers = []
    # for name, param in model.named_parameters():
    #     # Check if parameter belongs to blocks.10 or blocks.11 (last 2 of 12 layers)
    #     if any(f"blocks.{i}." in name for i in [10, 11]):
    #         # Unfreeze FFN components (MLP layers and norm2)
    #         if any(key in name for key in ["mlp", "norm2"]):
    #             param.requires_grad_(True)
    #             last2_params += param.numel()
    #             last2_layers.append(name)
    #         # Also unfreeze cross-attention layer norms (even though unused with context=None)
    #         if "norm_cross" in name:
    #             param.requires_grad_(True)
    #             last2_params += param.numel()
    #             last2_layers.append(name)

    # print(f"  Last 2 layers FFN/LayerNorm unfrozen: {last2_params:,} parameters")
    # print(f"    - Blocks 10-11 FFN and LayerNorms will be fully fine-tuned")
    ####################################################################################

    return model