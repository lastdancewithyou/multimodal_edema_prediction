from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaKLKD(nn.Module):
    """Binary KL(P_teacher || P_student) with temperature T, scaled by T^2.

    z_s, z_t are logits [B]. Teacher is already detached upstream, but we
    detach again defensively.
    """

    def __init__(self, T: float = 4.0, eps: float = 1e-7):
        super().__init__()
        self.T = T
        self.eps = eps

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        z_t = z_t.detach()
        p_t = torch.sigmoid(z_t / self.T).clamp(self.eps, 1 - self.eps)
        p_s = torch.sigmoid(z_s / self.T).clamp(self.eps, 1 - self.eps)
        kl = p_t * (p_t.log() - p_s.log()) + (1 - p_t) * ((1 - p_t).log() - (1 - p_s).log())
        return (self.T ** 2) * kl.mean()


KD_LOSSES = {
    "vanilla_kl": VanillaKLKD,
}


def build_kd_loss(name: str, **kwargs) -> nn.Module:
    if name not in KD_LOSSES:
        raise ValueError(f"unknown KD loss: {name!r}. available: {list(KD_LOSSES)}")
    return KD_LOSSES[name](**kwargs)


class StudentKDLoss(nn.Module):
    """Combined hard-label BCE + soft KD loss for the student.

    total = alpha * BCE(z_s, y) + (1 - alpha) * L_kd(z_s, z_t)
    """

    def __init__(self, kd_name: str = "vanilla_kl", kd_T: float = 4.0,
                 kd_alpha: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        self.alpha = kd_alpha
        self.kd = build_kd_loss(kd_name, T=kd_T)
        pw = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, y: torch.Tensor) -> dict:
        loss_kd = self.kd(z_s, z_t)
        loss_bce = self.bce(z_s, y.float())
        total = self.alpha * loss_bce + (1.0 - self.alpha) * loss_kd
        return {"total": total, "bce": loss_bce.detach(), "kd": loss_kd.detach()}


# =============================================================================
# Multi-label pathology loss for PathologyPerceiver
# =============================================================================
class PathologyMultiLabelLoss(nn.Module):
    """Stage 2 (image-only) + Stage 4 (multimodal) 각각에 per-pathology BCE.

    per_pathology_loss[k] = mean( BCE(logit[:, k], y[:, k]) * mask[:, k] ) / mean(mask[:, k])
    stage_loss           = Σ_k  label_weight[k] × per_pathology_loss[k]
    total                = α_stage2 × stage2_loss + α_stage4 × stage4_loss

    Args:
        label_weights:      [K]  per-pathology loss 가중치 (e.g., [1.0, 0.2, 0.2, 0.2])
        pos_weight:         [K]  per-pathology positive-class weight (capped 권장)
        alpha_stage2:       image-only aux loss 전체 가중치
        alpha_stage4:       multimodal aux loss 전체 가중치
    """

    def __init__(self,
                 label_weights: torch.Tensor,
                 pos_weight: torch.Tensor | None = None,
                 alpha_stage2: float = 0.5,
                 alpha_stage4: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.register_buffer("label_weights", label_weights.float())
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None
        self.alpha_stage2 = float(alpha_stage2)
        self.alpha_stage4 = float(alpha_stage4)
        self.eps = eps
        self.n_pathologies = int(label_weights.numel())

    def _per_pathology_bce(self, logits: torch.Tensor,
                            y: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        losses = []
        for k in range(self.n_pathologies):
            pw = None if self.pos_weight is None else self.pos_weight[k:k+1]
            l = F.binary_cross_entropy_with_logits(
                logits[:, k], y[:, k], reduction="none", pos_weight=pw)
            m = mask[:, k]
            l_masked = (l * m).sum() / (m.sum() + self.eps)
            losses.append(l_masked)
        return torch.stack(losses)  # [K]

    def forward(self,
                stage2_logits: torch.Tensor,
                stage4_logits: torch.Tensor,
                y_multi: torch.Tensor,
                y_multi_mask: torch.Tensor) -> dict:
        s2_per = self._per_pathology_bce(stage2_logits, y_multi, y_multi_mask)  # [K]
        s4_per = self._per_pathology_bce(stage4_logits, y_multi, y_multi_mask)  # [K]

        s2_total = (self.label_weights * s2_per).sum()
        s4_total = (self.label_weights * s4_per).sum()
        total = self.alpha_stage2 * s2_total + self.alpha_stage4 * s4_total

        return {
            "total":         total,
            "stage2_total":  s2_total.detach(),
            "stage4_total":  s4_total.detach(),
            "stage2_per":    s2_per.detach(),      # [K]
            "stage4_per":    s4_per.detach(),      # [K]
        }


# =============================================================================
# 3-way multi-label loss for DualPathologyPerceiver (image / ts / fusion branches)
# =============================================================================
class DualPathologyLoss(nn.Module):

    def __init__(self,
                 label_weights: torch.Tensor,
                 pos_weight: torch.Tensor | None = None,
                 alpha_img: float = 0.5,
                 alpha_ts:  float = 0.5,
                 alpha_fus: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.register_buffer("label_weights", label_weights.float())
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None
        self.alpha_img = float(alpha_img)
        self.alpha_ts  = float(alpha_ts)
        self.alpha_fus = float(alpha_fus)
        self.eps = eps
        self.n_pathologies = int(label_weights.numel())

    def _per_pathology_bce(self, logits: torch.Tensor,
                            y: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """logits/y/mask: [B, K]. Returns [K] per-pathology mean BCE."""
        losses = []
        for k in range(self.n_pathologies):
            pw = None if self.pos_weight is None else self.pos_weight[k:k+1]
            l = F.binary_cross_entropy_with_logits(
                logits[:, k], y[:, k], reduction="none", pos_weight=pw)
            m = mask[:, k]
            l_masked = (l * m).sum() / (m.sum() + self.eps)
            losses.append(l_masked)
        return torch.stack(losses)  # [K]

    def forward(self,
                img_logits: torch.Tensor,
                ts_logits: torch.Tensor,
                fusion_logits: torch.Tensor,
                y_multi: torch.Tensor,
                y_multi_mask: torch.Tensor) -> dict:
        img_per = self._per_pathology_bce(img_logits,    y_multi, y_multi_mask)   # [K]
        ts_per  = self._per_pathology_bce(ts_logits,     y_multi, y_multi_mask)   # [K]
        fus_per = self._per_pathology_bce(fusion_logits, y_multi, y_multi_mask)   # [K]

        # args.label_weights
        img_total = (self.label_weights * img_per).sum()
        ts_total  = (self.label_weights * ts_per).sum()
        fus_total = (self.label_weights * fus_per).sum()

        # label weight로 질병별 가중치 합산을 하고, 이후 모달리티별 가중치 alpha를 곱함.
        total = (self.alpha_img * img_total
                 + self.alpha_ts  * ts_total
                 + self.alpha_fus * fus_total)

        return {
            "total":     total,
            "img_total": img_total.detach(),
            "ts_total":  ts_total.detach(),
            "fus_total": fus_total.detach(),
            "img_per":   img_per.detach(),   # [K]
            "ts_per":    ts_per.detach(),    # [K]
            "fus_per":   fus_per.detach(),   # [K]
        }
