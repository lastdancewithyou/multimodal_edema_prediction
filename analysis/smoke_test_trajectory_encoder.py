"""CPU smoke test for LocalTrajectoryEncoder + pathology readout."""
from __future__ import annotations

import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.main_architecture_duett import (  # noqa: E402
    LocalTrajectoryEncoder,
    PatchDualPathologyPerceiver,
)


def main() -> None:
    torch.manual_seed(7)
    B, T, V, D, K = 3, 24, 34, 64, 4
    x = torch.zeros(B, T, 2 * V)
    observed = torch.rand(B, T, V) < 0.20
    x[:, :, :V] = torch.randn(B, T, V) * observed
    x[:, :, V:] = observed.float() * torch.randint(1, 4, (B, T, V)).float()

    encoder = LocalTrajectoryEncoder(
        n_vars=V, n_timesteps=T, d_model=D, recency_windows=(6, 12, 24)
    )
    tokens, padding_mask = encoder(
        tuple(x[i] for i in range(B)), return_padding_mask=True
    )
    expected = (B, V * 3 + 1, D)
    assert tokens.shape == expected, (tokens.shape, expected)
    assert padding_mask.shape == expected[:2], (padding_mask.shape, expected[:2])
    assert not padding_mask[:, -1].any(), "REP must never be masked"
    assert torch.isfinite(tokens).all()

    perceiver = PatchDualPathologyPerceiver(
        n_pathologies=K, d_ts=D, d_latent=D, n_heads=4, dropout=0.0
    )
    image_tokens = torch.randn(B, 49, D)
    out = perceiver(tokens, image_tokens, ts_padding_mask=padding_mask)
    loss = out["ts_logits"].square().mean() + out["fusion_logits"].square().mean()
    loss.backward()
    grad_norm = sum(
        float(p.grad.norm()) for p in encoder.parameters() if p.grad is not None
    )
    assert grad_norm > 0.0
    print(f"PASS tokens={tuple(tokens.shape)} logits={tuple(out['fusion_logits'].shape)} ")
    print(f"PASS finite=True encoder_grad_norm_sum={grad_norm:.6f}")


if __name__ == "__main__":
    main()
