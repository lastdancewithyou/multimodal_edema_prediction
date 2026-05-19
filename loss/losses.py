import os
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import timer


OUTPUT_DIR = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/output/"


class MultiModalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.use_label_smooth = args.use_label_smooth
        self.label_smoothing = args.label_smoothing

        # ==================== Binary Cross-Entropy Loss (Edema Detection) ====================
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        # ==================== Contrastive Loss ====================
        # init_temp = args.contrast_temperature
        # self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.temperature = args.contrast_temperature 

    def info_nce_loss(self, clinical_emb, text_emb):
        clinical_emb = F.normalize(clinical_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        logits = torch.matmul(clinical_emb, text_emb.T) / self.temperature
        
        labels = torch.arange(logits.size(0), device=logits.device)
        
        loss_c2t = F.cross_entropy(logits, labels)
        loss_t2c = F.cross_entropy(logits.T, labels)
        
        return (loss_c2t + loss_t2c) / 2

    # Edema Detection Head
    def binary_cross_entropy(self, edema_logits, edema_soft_labels):
        logits = edema_logits.squeeze(-1).float()
        labels = edema_soft_labels.float()

        valid_mask = ~torch.isnan(labels)
        num_samples = valid_mask.sum().item()

        if num_samples == 0:
            return 0.0 * edema_logits.sum(), 0

        loss_per_sample = self.bce_loss(logits[valid_mask], labels[valid_mask])
        return loss_per_sample.mean(), int(num_samples)

    # Sub-task BCE (Cardiomegaly/Pneumonia 공용 — NaN 라벨 마스킹)
    def _subtask_bce(self, logits, labels):
        logits = logits.squeeze(-1).float()
        labels = labels.float()

        valid_mask = ~torch.isnan(labels)
        num_samples = valid_mask.sum().item()

        if num_samples == 0:
            return 0.0 * logits.sum(), 0

        loss_per_sample = self.bce_loss(logits[valid_mask], labels[valid_mask])
        return loss_per_sample.mean(), int(num_samples)

    # # Subtype Classification Head — 주석 보존 (재활성 대비)
    # def cross_entropy(self, subtype_logits, subtype_labels, edema_labels):
    #     logits = subtype_logits.float()
    #     edema_labels_long = edema_labels.long()
    #     subtype_labels_long = subtype_labels.long()
    #     valid_mask = ((edema_labels_long == 1) & ((subtype_labels_long == 0) | (subtype_labels_long == 1) | (subtype_labels_long == 2)))
    #     num_samples = valid_mask.sum().item()
    #     if num_samples == 0:
    #         return 0.0 * subtype_logits.sum(), 0
    #     logits_valid = logits[valid_mask]
    #     labels_valid = subtype_labels_long[valid_mask]
    #     ce_loss = self.ce_loss(logits_valid, labels_valid)
    #     return ce_loss, num_samples

    def forward(
            self,
            edema_logits,
            edema_soft_labels,
            cardiomegaly_logits=None, cardiomegaly_labels=None, cardio_weight=0.0,
            pneumonia_logits=None, pneumonia_labels=None, pneumo_weight=0.0,
            # subtype_logits=None, subtype_labels=None, ce_weight=0.0,
            bce_weight=0.0, 
            info_loss_t=None, info_weight=0.0,
            # contrast_weight=0.0, proj_view1=None, proj_view2=None,
            device=None, accelerator=None,
        ):
        # -------------------- BCE Loss (Edema Soft) --------------------
        if bce_weight > 0.0 and edema_logits is not None and edema_soft_labels is not None:
            with timer("BCE Loss", accelerator):
                bce_loss, bce_count = self.binary_cross_entropy(edema_logits, edema_soft_labels)
        else:
            bce_loss = torch.tensor(0.0, device=device, requires_grad=False)
            bce_count = 0

        # -------------------- Sub-task --------------------
        if cardio_weight > 0.0 and cardiomegaly_logits is not None and cardiomegaly_labels is not None:
            with timer("Cardio BCE", accelerator):
                cardio_loss, cardio_count = self._subtask_bce(cardiomegaly_logits, cardiomegaly_labels)
        else:
            cardio_loss = torch.tensor(0.0, device=device, requires_grad=False)
            cardio_count = 0

        if pneumo_weight > 0.0 and pneumonia_logits is not None and pneumonia_labels is not None:
            with timer("Pneumo BCE", accelerator):
                pneumo_loss, pneumo_count = self._subtask_bce(pneumonia_logits, pneumonia_labels)
        else:
            pneumo_loss = torch.tensor(0.0, device=device, requires_grad=False)
            pneumo_count = 0

        # # -------------------- CE Loss (Subtype) — 주석 --------------------
        # if ce_weight > 0.0:
        #     with timer("CE Loss", accelerator):
        #         ce_loss, ce_count = self.cross_entropy(subtype_logits, subtype_labels, edema_labels)
        # else:
        #     ce_loss = torch.tensor(0.0, device=device, requires_grad=False)
        #     ce_count = 0
        ce_loss = torch.tensor(0.0, device=device, requires_grad=False)
        ce_count = 0

        # -------------------- NTXent (시계열 two-view contrastive) --------------------
        # if contrast_weight > 0.0 and proj_view1 is not None and proj_view2 is not None and proj_view1.size(0) >= 2:
        #     with timer("NTXent Loss", accelerator):
        #         ntxent_loss = self.ntxent_loss(proj_view1, proj_view2)
        #     ntxent_count = int(proj_view1.size(0))
        # else:
        #     ntxent_loss = torch.tensor(0.0, device=device, requires_grad=False)
        #     ntxent_count = 0

        if info_loss_t is None:
            info_loss_t = torch.tensor(0.0, device=device, requires_grad=False)


        total_loss = (
            bce_weight * bce_loss
            + cardio_weight * cardio_loss
            + pneumo_weight * pneumo_loss
            # + contrast_weight * ntxent_loss
            + info_weight * info_loss_t 
            # + ce_weight * ce_loss
        )

        loss_counts = {
            'bce_count': bce_count,
            'ce_count': ce_count,
            'cardio_count': cardio_count,
            'pneumo_count': pneumo_count,
            'info_count': 1 if info_loss_t.item() > 0 else 0,
        }

        return total_loss, bce_loss, ce_loss, cardio_loss, pneumo_loss, info_loss_t, loss_counts

    # validation & test
    def inference(self, classification_input, logits, labels):
        return {
            'logits': logits,
            'labels': labels,
            'window_embeddings': classification_input
        }


# class NTXentLoss(torch.nn.Module):
#     def __init__(self, temperature=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.criterion = torch.nn.CrossEntropyLoss()

#     def forward(self, z_i, z_j):
#         """
#         z_i: [B, Dim] - View 1의 잠재 임베딩 (Latent)
#         z_j: [B, Dim] - View 2의 잠재 임베딩 (Latent)
#         """
#         B = z_i.size(0)

#         # 1. L2 정규화 (코사인 유사도를 구하기 위해)
#         z_i = F.normalize(z_i, dim=1)
#         z_j = F.normalize(z_j, dim=1)

#         # 2. 모든 임베딩을 하나로 합침: [2B, Dim]
#         z = torch.cat([z_i, z_j], dim=0)

#         # 3. 유사도 행렬 계산: [2B, 2B]
#         sim_matrix = torch.matmul(z, z.T) / self.temperature

#         # 4. 자기 자신과의 유사도(대각 원소)는 정답이 될 수 없으므로 아주 작은 값으로 마스킹
#         sim_matrix.fill_diagonal_(-1e9)

#         # 5. 정답 라벨 생성 (i의 짝은 i+B, i+B의 짝은 i)
#         labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)], dim=0).to(z.device)

#         # 6. 크로스 엔트로피로 Loss 계산
#         loss = self.criterion(sim_matrix, labels)
        
#         return loss    


############################################################################################################
############################################################################################################
# class ConstrainttimeLoss(nn.Module):
#     """
#     Temporal Neighbor Contrastive Loss (Single-View, No Augmentation)

#     - 같은 환자의 시간적으로 가까운 window끼리 embedding을 가까이 당김 (pull)
#     - 시간 거리에 반비례하는 가중치: w(i,j) = 1 / (beta + |t_i - t_j|)
#     - Weighted InfoNCE loss 사용

#     Args:
#         beta: Time distance weight decay factor (default: 1.0)
#     """
#     def __init__(self, ucl_beta=None):
#         super().__init__()
#         self.beta = ucl_beta
#         self.very_neg = -1e9

#     def forward(self, embeddings, time_indices, batch_indices, temperature, accelerator=None):
#         """
#         Args:
#             embeddings: [N_valid, D] - valid window embeddings (already filtered by model)
#             time_indices: [N_valid] - temporal indices for each window
#             batch_indices: [N_valid] - batch (patient) indices for each window
#             window_mask: [B, W] - not used (already filtered in model)
#             temperature: temperature for InfoNCE
#             accelerator: accelerator object for logging
#         """
#         device = embeddings.device
#         N = embeddings.size(0)

#         # Early exit if not enough samples
#         if N < 2:
#             return torch.tensor(0.0, device=device, requires_grad=False), 0

#         # Normalize embeddings (if not already normalized)
#         z = F.normalize(embeddings, p=2, dim=-1)  # [N, D]

#         # ------------------------------------------------------------------
#         # Time-aware Pull Loss (Weighted InfoNCE)
#         # ------------------------------------------------------------------
#         # Compute similarity matrix: [N, N]
#         sim = torch.matmul(z, z.T) / temperature

#         # Mask diagonal (self-contrast)
#         mask_self = torch.eye(N, dtype=torch.bool, device=device)
#         sim = sim.masked_fill(mask_self, self.very_neg)

#         # Compute time-aware positive mask and weights
#         same_patient = batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)  # [N, N]
#         time_dist = (time_indices.unsqueeze(1) - time_indices.unsqueeze(0)).abs()  # [N, N]
#         mask_pos = same_patient & (time_dist > 0)  # Exclude self, same patient only

#         # Time-aware weights: closer in time = higher weight
#         w = torch.zeros_like(sim)
#         w[mask_pos] = (1.0 / (self.beta + time_dist[mask_pos].float())).to(w.dtype)
#         w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # Normalize weights

#         # Weighted InfoNCE loss
#         pull_loss = -(w * F.log_softmax(sim, dim=1)).sum(dim=1).mean()

#         if accelerator is not None and accelerator.is_main_process:
#             num_pos_pairs = mask_pos.sum().item()
#             total_pairs = N * (N - 1)
#             pos_ratio = 100.0 * num_pos_pairs / total_pairs if total_pairs > 0 else 0.0

#             # print(f"[Temporal UCL] Loss={pull_loss.item():.4f} | "
#             #     f"Positive pairs: {num_pos_pairs}/{total_pairs} ({pos_ratio:.1f}%) | "
#             #     f"Samples: {N}")

#         return pull_loss, N


# ## Supervised Contrastive Loss (Single-View)
# class SupConLoss(nn.Module):
#     def __init__(self):
#         super(SupConLoss, self).__init__()

#     def forward(self, embeddings, labels, window_mask, temperature):
#         """
#         Supervised Contrastive Loss

#         Args:
#             embeddings: [N_valid, D] - already filtered valid embeddings from model
#             labels: [B, W] - edema labels
#             window_mask: [B, W] - window mask
#             temperature: temperature scaling
#         """
#         device = embeddings.device

#         # Flatten labels and mask to match embeddings
#         lab_flat = labels.view(-1)                      # [B*W]
#         mask_flat = window_mask.view(-1).bool()         # [B*W]

#         # Extract labels only for valid windows (matching embeddings order)
#         labels_valid = lab_flat[mask_flat]  # [N_valid]

#         # Filter out unlabeled samples (-1)
#         labeled_mask = (labels_valid != -1)
#         num_samples = labeled_mask.sum().item()

#         if num_samples == 0:
#             return torch.tensor(0.0, device=device, requires_grad=True), 0

#         features = embeddings[labeled_mask]     # [N_labeled, D]
#         labels_final = labels_valid[labeled_mask]  # [N_labeled]

#         # L2 normalize
#         features = F.normalize(features, p=2, dim=-1)

#         N = features.shape[0]
#         if N < 2:
#             return torch.tensor(0.0, device=device, requires_grad=True), N

#         # Compute similarity matrix
#         logits = torch.div(features @ features.T, temperature)  # [N, N]

#         # Create positive mask (same label)
#         labels_row = labels_final.view(-1, 1)
#         labels_col = labels_final.view(1, -1)
#         pos_mask = (labels_row == labels_col).float()  # [N, N]

#         # Remove self-contrast (diagonal)
#         logits_mask = torch.ones_like(pos_mask)
#         diag_idx = torch.arange(N, device=device)
#         logits_mask[diag_idx, diag_idx] = 0
#         pos_mask = pos_mask * logits_mask  # Remove diagonal from positive mask

#         # Mask out invalid positions in logits
#         logits_masked = logits + (1 - logits_mask) * -1e9

#         # Compute log probability
#         log_prob = logits - torch.logsumexp(logits_masked, dim=1, keepdim=True)

#         # Compute mean of log-likelihood over positive pairs
#         pos_per_anchor = pos_mask.sum(1).clamp(min=1e-6)
#         mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_per_anchor
#         loss = -mean_log_prob_pos.mean()

#         return loss, N


# class InfoNCELoss(nn.Module):
#     def __init__(self):
#         super(InfoNCELoss, self).__init__()

#     def forward(self, embeddings, temperature):
#         device = embeddings.device
#         N = embeddings.shape[0]

#         if N < 2:
#             return torch.tensor(0.0, device=device, requires_grad=True), N

#         # L2 normalize
#         features = F.normalize(embeddings, p=2, dim=-1)

#         sim_matrix = torch.matmul(features, features.T) / temperature  # [N, N]

#         # Create mask to exclude diagonal
#         mask_self = torch.eye(N, dtype=torch.bool, device=device)

#         # High similarity closer embeddings should be pulled together
#         sim_matrix_masked = sim_matrix.masked_fill(mask_self, -1e9) # Mask out diagonal for denominator

#         log_prob = F.log_softmax(sim_matrix_masked, dim=1)  # [N, N]

#         loss = -log_prob.sum(dim=1).mean() / (N - 1)

#         return loss, N