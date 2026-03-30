import os
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# from training.run import parse_arguments
from utils.utils import timer


OUTPUT_DIR = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/output/"


class MultiModalLoss(nn.Module):
    def __init__(self, args, class_weights=None):
        super().__init__()

        self.num_classes = args.num_classes
        self.use_label_smooth = args.use_label_smooth
        self.label_smoothing = args.label_smoothing
        amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # ==================== CE Loss ====================
        if class_weights is not None:
            print("[Loss] Using class weights for CE Loss")
            self.class_weights = class_weights.to(dtype=amp_dtype)

            # label smoothing
            if self.use_label_smooth:
                self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights.float(), ignore_index=-1, label_smoothing=self.label_smoothing)
                print(f"[Loss] CE Loss with label smoothing (factor={self.label_smoothing})")
            else:
                self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights.float(), ignore_index=-1)
                print(f"[Loss] Standard CE Loss with class weights")
        
        # CE loss without class weights
        else:
            print("[Loss] CE Loss without class weights")
            if self.use_label_smooth:
                self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=self.label_smoothing)
                print(f"[Loss] CE Loss with label smoothing (factor={self.label_smoothing})")
            else:
                self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
                print(f"[Loss] Standard CE Loss")

        # ==================== Binary Cross-Entropy Loss (Edema Detection) ====================
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # reduction='none' for manual filtering
        print(f"[Loss] BCE Loss initialized for edema detection")

        # ==================== MSE Loss (Score Diff Regression) ====================
        self.mse_loss = nn.MSELoss(reduction='mean')
        print(f"[Loss] MSE Loss initialized for raw score_diff regression (range: -7~11)")

    ###########################################################################
    def cross_entropy(self, subtype_logits, subtype_labels, edema_labels, window_mask):
        """
        Subtype classification loss (Cross-Entropy)
        Only applies to windows where:
        - window_mask == 1 (valid, non-padded)
        - edema_labels == 1 (edema-positive)
        - subtype_labels in {0, 1} (valid subtype labels)

        Args:
            subtype_logits: [B, W, 2] - subtype classifier output
            subtype_labels: [B, W] - subtype labels {0: non-cardiogenic, 1: cardiogenic, -1: unlabeled}
            edema_labels: [B, W] - edema labels {0, 1}
            window_mask: [B, W] - valid window mask
        """
        # Flatten tensors
        logits_flat = subtype_logits.view(-1, 2).float()      # [B*W, C=2]
        subtype_labels_flat = subtype_labels.view(-1).long()  # [B*W]
        edema_labels_flat = edema_labels.view(-1).long()      # [B*W]
        window_mask_flat = window_mask.view(-1).bool()        # [B*W]

        # Filter (Valid window + Edema-positive + Valid subtype label)
        valid_mask = (window_mask_flat &                                            # Valid windows
                    (edema_labels_flat == 1) &                                      # Edema-positive samples only
                    ((subtype_labels_flat == 0) | (subtype_labels_flat == 1)))      # For subtype classification

        num_samples = valid_mask.sum().item()

        # 보호장치
        if num_samples == 0:
            return torch.tensor(0.0, device=subtype_logits.device, requires_grad=False), 0

        # Extract valid samples
        logits_valid = logits_flat[valid_mask]           # [N_valid, 2]
        labels_valid = subtype_labels_flat[valid_mask]   # [N_valid] in {0, 1}

        ce_loss = self.ce_loss(logits_valid, labels_valid)
        return ce_loss, num_samples

    def binary_cross_entropy(self, edema_logits, edema_labels, window_mask):
        edema_logits_flat = edema_logits.view(-1).float()   # [B*W]
        edema_labels_flat = edema_labels.view(-1).float()     # [B*W]
        window_mask_flat = window_mask.view(-1).bool()        # [B*W]

        valid_mask = window_mask_flat & (edema_labels_flat != -1)
        num_samples = valid_mask.sum().item()

        if num_samples == 0:
            return torch.tensor(0.0, device=edema_logits.device, requires_grad=False), 0

        logits_valid = edema_logits_flat[valid_mask]
        labels_valid = edema_labels_flat[valid_mask]

        loss_per_sample = self.bce_loss(logits_valid, labels_valid)
        bce_loss = loss_per_sample.mean()

        return bce_loss, num_samples

    def regression_mse(self, regression_preds, score_diff_targets, edema_labels, window_mask):
        """
        MSE loss for score_diff prediction
        Only applies to windows where:
        - window_mask == 1 (valid, non-padded)
        - edema_labels == 1 (edema-positive only)
        - score_diff_targets is not NaN
        """
        preds_flat = regression_preds.view(-1).float()    
        targets_flat = score_diff_targets.view(-1).float()
        edema_flat = edema_labels.view(-1).long()         
        window_mask_flat = window_mask.view(-1).bool()    

        # Filter: valid window + edema-positive + non-NaN target
        valid_mask = (window_mask_flat & (edema_flat == 1) & ~torch.isnan(targets_flat))

        num_samples = valid_mask.sum().item()

        if num_samples == 0:
            return torch.tensor(0.0, device=regression_preds.device, requires_grad=False), 0

        preds_valid = preds_flat[valid_mask]      
        targets_valid = targets_flat[valid_mask]  

        mse_loss = self.mse_loss(preds_valid, targets_valid)

        return mse_loss, num_samples

    def forward(self,
                # Model outputs
                edema_logits, subtype_logits, valid_embeddings, window_time_indices, batch_indices, regression_preds,
                # Labels
                edema_labels, subtype_labels, window_mask, score_diff_targets=None,
                # Loss weights
                bce_weight=0.0, ce_weight=0.0, mse_weight=0.0, device=None, accelerator=None
        ):
        if device is None:
            device = edema_logits.device

        # -------------------- (0) Binary CE Loss (Edema Detection) --------------------
        if bce_weight > 0.0 and edema_logits is not None and edema_labels is not None:
            with timer("BCE Loss", accelerator):
                bce_loss, bce_count = self.binary_cross_entropy(edema_logits, edema_labels, window_mask)
        else:
            # Use same device as edema_logits if available, else window_mask
            ref_device = edema_logits.device if edema_logits is not None else window_mask.device
            bce_loss = torch.tensor(0.0, device=ref_device, requires_grad=False)
            bce_count = 0

        # -------------------- (1) CE Loss (Subtype Classification) --------------------
        if ce_weight > 0.0:
            with timer("CE Loss", accelerator):
                ce_loss, ce_count = self.cross_entropy(subtype_logits, subtype_labels, edema_labels, window_mask)
        else:
            # Use same device as subtype_logits if available, else window_mask
            ref_device = subtype_logits.device if subtype_logits is not None else window_mask.device
            ce_loss = torch.tensor(0.0, device=ref_device, requires_grad=False)
            ce_count = 0

        # -------------------- (2) MSE Loss (Score Diff Regression) --------------------
        if mse_weight > 0.0:
            with timer("MSE Loss", accelerator):
                mse_loss, mse_count = self.regression_mse(regression_preds, score_diff_targets, edema_labels, window_mask)
        else:
            # Use same device as regression_preds if available, else window_mask
            ref_device = regression_preds.device if regression_preds is not None else window_mask.device
            mse_loss = torch.tensor(0.0, device=ref_device, requires_grad=False)
            mse_count = 0

        # -------------------- NaN Detection --------------------
        if torch.isnan(bce_loss) or torch.isinf(bce_loss):
            print(f"[WARNING] BCE Loss is NaN/Inf: {bce_loss.item()}")

        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            print(f"[WARNING] CE Loss is NaN/Inf: {ce_loss.item()}")

        if torch.isnan(mse_loss) or torch.isinf(mse_loss):
            print(f"[WARNING] MSE Loss is NaN/Inf: {mse_loss.item()}")

        # -------------------- Total Loss --------------------
        total_loss = (
            bce_weight * bce_loss +
            ce_weight * ce_loss +
            mse_weight * mse_loss
        )

        # -------------------- Sample Counts --------------------
        loss_counts = {
            'bce_count': bce_count,
            'ce_count': ce_count,
            'mse_count': mse_count
        }

        return total_loss, bce_loss, ce_loss, mse_loss, loss_counts
    
    # validation & test
    def inference(self, classification_input, logits, labels, window_mask):
        return {
            'logits': logits,
            'labels': labels,
            'window_mask': window_mask,
            'window_embeddings': classification_input
        }


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