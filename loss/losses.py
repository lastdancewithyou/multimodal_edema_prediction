import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import timer


OUTPUT_DIR = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/output/"


class MaskedSoftCrossEntropyLoss(nn.Module):
    """Soft-label cross-entropy with per-sample masking.

    Args:
        logits:        [B, C] raw class logits.
        target_probs:  [B, C] soft target distribution (rows should sum to 1 where mask=1).
        mask:          [B] indicator (1 = include in loss, 0 = skip).

    Returns:
        (loss, valid_count) where loss = -sum_i mask_i * sum_c p_ic * log_softmax(logits_ic) / sum(mask).
        If sum(mask) == 0, loss is a zero tensor with grad connection to logits.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, target_probs, mask):
        logits = logits.float()
        target_probs = target_probs.float()
        mask = mask.float()

        valid_count = mask.sum()
        if valid_count.item() == 0:
            return 0.0 * logits.sum(), 0

        log_probs = F.log_softmax(logits, dim=-1)                       # [B, C]
        per_sample_ce = -(target_probs * log_probs).sum(dim=-1)         # [B]
        masked_ce = per_sample_ce * mask                                # [B]
        loss = masked_ce.sum() / valid_count.clamp(min=1.0)
        return loss, int(valid_count.item())


class DualStreamDistillationLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.subtype_ce = MaskedSoftCrossEntropyLoss()
        self.lupi_fd_weight  = args.lupi_fd_weight    # feature distillation (fused level)
        self.lupi_rd_weight  = args.lupi_rd_weight    # readout distillation (post-readout pre-classifier)
        self.lupi_kd_weight  = args.lupi_kd_weight    # logit distillation
        self.lupi_cov_weight = args.lupi_cov_weight   # covariance regularization
        self.subtype_weight  = args.subtype_weight    # masked soft-CE on subtype head

    def binary_cross_entropy(self, logit, soft_labels):
        logit  = logit.squeeze(-1).float()
        labels = soft_labels.float()
        valid_mask = ~torch.isnan(labels)
        num_samples = int(valid_mask.sum().item())
        if num_samples == 0:
            return 0.0 * logit.sum(), 0
        loss_per_sample = self.bce_loss(logit[valid_mask], labels[valid_mask])
        return loss_per_sample.mean(), num_samples
    
    def covariance_loss(self, z):
        B, D = z.shape
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        z_centered = z - z.mean(dim=0)
        cov_matrix = (z_centered.T @ z_centered) / (B - 1)
        off_diagonal_mask = ~torch.eye(D, device=z.device, dtype=torch.bool)
        return (cov_matrix[off_diagonal_mask] ** 2).sum() / D

    def forward(self, logit_priv, logit_deploy, fused_priv, fused_deploy,
                feat_priv, feat_deploy,
                edema_soft_labels,
                subtype_logits_priv=None, subtype_logits_deploy=None,
                subtype_target_probs=None, subtype_mask=None,
                device=None, accelerator=None):

        # BCE on both paths
        with timer("BCE Priv", accelerator):
            bce_priv, bce_count = self.binary_cross_entropy(logit_priv, edema_soft_labels)
        with timer("BCE Deploy", accelerator):
            bce_deploy, _ = self.binary_cross_entropy(logit_deploy, edema_soft_labels)

        # Feature distillation (FD)
        with timer("Feature KD", accelerator):
            if fused_priv.shape == fused_deploy.shape and fused_priv.numel() > 0:
                fd_loss = (1.0 - F.cosine_similarity(
                    fused_deploy, fused_priv.detach(), dim=-1
                )).mean()

            else:
                fd_loss = torch.tensor(0.0, device=device, requires_grad=False)

        # Readout distillation (RD) — Cosine distance (검증된 baseline)
        with timer("Readout KD", accelerator):
            if feat_priv is not None and feat_deploy is not None and feat_priv.numel() > 0:
                # 방향 일치
                rd_cos_loss = (1.0 - F.cosine_similarity(
                    feat_deploy, feat_priv.detach(), dim=-1
                )).mean()

                # 크기 일치
                rd_mse_loss = F.smooth_l1_loss(feat_deploy, feat_priv.detach())

                rd_alpha = 1.0
                rd_beta = 1.0

                rd_loss = rd_alpha * rd_cos_loss + rd_beta * rd_mse_loss

                rd_cos_val = float(rd_cos_loss.detach().item())
                rd_mse_val = float(rd_mse_loss.detach().item())
            else:
                rd_loss = torch.tensor(0.0, device=device, requires_grad=False)
                rd_cos_val = 0.0
                rd_mse_val = 0.0

        # Logit distillation (KD)
        with timer("Logit KD", accelerator):
            lp = logit_priv.squeeze(-1).float()
            ld = logit_deploy.squeeze(-1).float()
            y  = edema_soft_labels.float()
            valid_mask = ~torch.isnan(y)

            if int(valid_mask.sum().item()) > 0:
                # teacher_probs = torch.sigmoid(lp[valid_mask].detach())
                # kd_loss = F.binary_cross_entropy_with_logits(
                #     ld[valid_mask], teacher_probs, reduction='mean'
                # )
                
                T = 2.0
                teacher_probs = torch.sigmoid(lp[valid_mask].detach() / T)
                kd_loss = F.binary_cross_entropy_with_logits(
                    ld[valid_mask] / T, teacher_probs, reduction='mean'
                ) * (T ** 2)   # gradient scale 보정
            else:
                kd_loss = torch.tensor(0.0, device=device, requires_grad=False)

        # ── Covariance regularization ──
        with timer("Covariance Reg", accelerator):
            fd_pooled = fused_deploy.mean(dim=1)
            cov_loss  = self.covariance_loss(fd_pooled)

            # # Normalized
            # fd_pooled_norm = F.normalize(fd_pooled, p=2, dim=-1)
            # cov_loss  = self.covariance_loss(fd_pooled_norm)

        # ── Subtype masked soft-CE (auxiliary head, both paths — BCE 패턴과 동일하게 합산) ──
        with timer("Subtype CE", accelerator):
            has_target = (subtype_target_probs is not None) and (subtype_mask is not None)
            if has_target and subtype_logits_priv is not None:
                subtype_loss_priv, subtype_count = self.subtype_ce(
                    subtype_logits_priv, subtype_target_probs, subtype_mask
                )
            else:
                subtype_loss_priv = torch.tensor(0.0, device=device, requires_grad=False)
                subtype_count = 0
            if has_target and subtype_logits_deploy is not None:
                subtype_loss_deploy, sc_d = self.subtype_ce(
                    subtype_logits_deploy, subtype_target_probs, subtype_mask
                )
                subtype_count = sc_d if subtype_count == 0 else subtype_count
            else:
                subtype_loss_deploy = torch.tensor(0.0, device=device, requires_grad=False)
            subtype_loss = subtype_loss_priv + subtype_loss_deploy

        total_loss = (
            bce_priv
            + bce_deploy
            + self.lupi_fd_weight  * fd_loss
            + self.lupi_rd_weight  * rd_loss
            + self.lupi_kd_weight  * kd_loss
            + self.lupi_cov_weight * cov_loss
            + self.subtype_weight  * subtype_loss
        )

        loss_counts = {
            'bce_count': bce_count,
            'rd_cos': rd_cos_val,
            'rd_mse': rd_mse_val,
            'subtype_count':       subtype_count,
            'subtype_loss_priv':   float(subtype_loss_priv.detach().item()),
            'subtype_loss_deploy': float(subtype_loss_deploy.detach().item()),
        }
        return total_loss, bce_priv, bce_deploy, fd_loss, rd_loss, kd_loss, cov_loss, subtype_loss, loss_counts

    # validation & test에서 사용 (deploy path만 들어오는 케이스)
    def deploy_bce(self, logit_deploy, edema_soft_labels):
        return self.binary_cross_entropy(logit_deploy, edema_soft_labels)
