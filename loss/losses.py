import os
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# from training.run import parse_arguments
from utils import timer


OUTPUT_DIR = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/output/"


class MultiModalLoss(nn.Module):
    """
    - Loss들을 관리하는 최상위 모듈
    - 현재는 라벨 없는 데이터를 사용하는 Loss는 포함되어 있지 않음.
    """
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
        

        # ==================== Temporal Neighbor based Contrastive Learning (For Generalization) ====================
        self.use_ucl = args.use_ucl
        if self.use_ucl:
            self.ucl_loss_fn = ConstrainttimeLoss(ucl_beta=args.ucl_beta)
            self.ucl_temperature = args.ucl_temperature
            print(f"[Loss] Temporal UCL enabled (temperature={self.ucl_temperature})")
        else: 
            self.ucl_loss_fn = None
            print(f"[Loss] Temporal UCL disabled")

        # ==================== Supervised Contrastive Loss ====================
        # self.use_supcon = args.use_supcon
        # if self.use_supcon:
        #     self.supcon_loss_fn = SupConLoss()
        #     self.scl_temperature = args.scl_temperature
        #     print(f"[Loss] Supervised Contrastive Loss enabled (temperature={self.scl_temperature})")
        # else:
        #     self.supcon_loss_fn = None
        #     print(f"[Loss] Supervised Contrastive Loss disabled")

        # # ==================== Target_SupCon Loss (TSC + KCL) ====================
        # self.use_target_supcon = args.use_target_supcon
        # if self.use_target_supcon:
        #     self.target_supcon_loss_fn = TSCwithQueue(
        #         args=args,
        #         embedding_dim=args.head_hidden_dim2,
        #         queue_size=args.target_supcon_queue_size,
        #         n_cls=args.num_classes,
        #         T=args.target_supcon_temperature,
        #         num_positive=args.target_supcon_K,
        #         targeted=True,
        #         tr=args.target_supcon_tr,
        #         sep_t=True,
        #         tw=args.target_supcon_tw
        #     )
        #     print(f"[Loss] TSCwithQueue Loss enabled")
        #     print(f"       - Queue Size: {args.target_supcon_queue_size}")
        #     print(f"       - Temperature: {args.target_supcon_temperature}")
        #     print(f"       - K (max positives): {args.target_supcon_K}")
        #     print(f"       - TSC Weight (tw): {args.target_supcon_tw}")
        #     print(f"       - Embedding Dim: {args.head_hidden_dim2}")
        # else:
        #     self.target_supcon_loss_fn = None
        #     print(f"[Loss] TSCwithQueue Loss disabled")

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

        # 보호장치
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=subtype_logits.device, requires_grad=False)

        # Extract valid samples
        logits_valid = logits_flat[valid_mask]           # [N_valid, 2]
        labels_valid = subtype_labels_flat[valid_mask]   # [N_valid] in {0, 1}

        ce_loss = self.ce_loss(logits_valid, labels_valid)
        return ce_loss

    def binary_cross_entropy(self, edema_logits, edema_labels, window_mask):
        edema_logits_flat = edema_logits.view(-1).float()   # [B*W]
        edema_labels_flat = edema_labels.view(-1).float()     # [B*W]
        window_mask_flat = window_mask.view(-1).bool()        # [B*W]

        valid_mask = window_mask_flat & (edema_labels_flat != -1)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=edema_logits.device, requires_grad=False)

        logits_valid = edema_logits_flat[valid_mask]
        labels_valid = edema_labels_flat[valid_mask]

        loss_per_sample = self.bce_loss(logits_valid, labels_valid)
        bce_loss = loss_per_sample.mean()

        return bce_loss

    def forward(self,
                # Model outputs
                edema_logits, subtype_logits, valid_embeddings,
                window_time_indices, batch_indices,
                # Labels
                edema_labels, subtype_labels, window_mask,
                # Loss weights
                bce_weight=0.0, ce_weight=0.0, ucl_weight=0.0,
                # Misc
                device=None, accelerator=None
        ):
        if device is None:
            device = edema_logits.device

        # -------------------- (0) Binary CE Loss (Edema Detection) --------------------
        if bce_weight > 0.0 and edema_logits is not None and edema_labels is not None:
            with timer("BCE Loss", accelerator):
                bce_loss = self.binary_cross_entropy(edema_logits, edema_labels, window_mask)
        else:
            bce_loss = torch.tensor(0.0, device=device, requires_grad=False)

        # -------------------- (1) CE Loss (Subtype Classification) --------------------
        if ce_weight > 0.0:
            with timer("CE Loss", accelerator):
                ce_loss = self.cross_entropy(subtype_logits, subtype_labels, edema_labels, window_mask)
        else:
            ce_loss = torch.tensor(0.0, device=device, requires_grad=False)

        # -------------------- (2) Temporal UCL Loss --------------------
        if self.use_ucl and ucl_weight > 0.0:
            with timer("Temporal UCL Loss", accelerator):
                ucl_loss = self.ucl_loss_fn(
                    embeddings=valid_embeddings,
                    time_indices=window_time_indices,
                    batch_indices=batch_indices,
                    temperature=self.ucl_temperature,
                    accelerator=accelerator
                )
        else:
            ucl_loss = torch.tensor(0.0, device=device, requires_grad=False)

        # unused code
        #####################################################################################################################################
        # # -------------------- (2) Supervised Contrastive Loss (Subtype Classification) --------------------
        # # NOTE: For multi-task learning, only apply contrastive loss to edema=1 samples with labeled subtypes
        # if self.use_supcon and scl_weight > 0.0:
        #     with timer("Supervised Contrastive Loss", accelerator):
        #         supcon_input = projected_embeddings_multiview if projected_embeddings_multiview is not None else classification_input

        #         # Multi-task filtering: only use edema=1 AND subtype IN {1,2} samples
        #         if edema_labels is not None:
        #             # Create filtered inputs for SupCon
        #             B, W = subtype_labels.shape
        #             subtype_labels_flat = subtype_labels.reshape(B * W)
        #             edema_labels_flat = edema_labels.reshape(B * W)
        #             mask_flat = window_mask.reshape(B * W).bool()

        #             # Get valid windows
        #             subtype_valid = subtype_labels_flat[mask_flat]  # [Nwin]
        #             edema_valid = edema_labels_flat[mask_flat]      # [Nwin]

        #             # Filter: edema==1 AND subtype IN {1, 2} (exclude NaN, -1, 0)
        #             # Subtype labels: 1 (non-cardiogenic), 2 (cardiogenic)
        #             has_edema = (edema_valid == 1)
        #             has_subtype_label = (subtype_valid == 1) | (subtype_valid == 2)
        #             contrastive_mask = has_edema & has_subtype_label

        #             if contrastive_mask.sum() > 0:
        #                 # Filter embeddings and labels (single-view, no augmentation)
        #                 # supcon_input: [Nwin, D] → filter → [N_contrastive, D]
        #                 supcon_input_filtered = supcon_input[contrastive_mask]
        #                 labels_filtered = subtype_valid[contrastive_mask]

        #                 # Create a temporary window mask (all 1s since we already filtered)
        #                 temp_window_mask = torch.ones(labels_filtered.shape[0], device=device, dtype=torch.bool)

        #                 # Reshape for SupConLoss: expects [B, W, D] format
        #                 # We treat filtered samples as a single "batch" with W windows
        #                 scl_loss = self.supcon_loss_fn(
        #                     embeddings=supcon_input_filtered.unsqueeze(0),  # [1, N_contrastive, D]
        #                     labels=labels_filtered.unsqueeze(0),  # [1, N_contrastive]
        #                     window_mask=temp_window_mask.unsqueeze(0),  # [1, N_contrastive]
        #                     temperature=self.scl_temperature
        #                 )
        #             else:
        #                 # No samples for contrastive learning
        #                 scl_loss = torch.tensor(0.0, device=device, requires_grad=False)
        #         else:
        #             # Legacy behavior: use all labeled samples
        #             scl_loss = self.supcon_loss_fn(
        #                 embeddings=supcon_input,
        #                 labels=labels,
        #                 window_mask=window_mask,
        #                 temperature=self.scl_temperature
        #             )
        # else:
        #     scl_loss = torch.tensor(0.0, device=device, requires_grad=False)

        # # -------------------- (2-1) Target_SupCon Loss (KCL + TSC) --------------------
        # if self.use_target_supcon and target_supcon_weight > 0.0 and projected_embeddings_multiview is not None:
        #     with timer("Target_SupCon Loss", accelerator):
        #         # Multi-view projections: [Nwin, 2, D]
        #         z_view0 = projected_embeddings_multiview[:, 0, :]  # [Nwin, D]
        #         z_view1 = projected_embeddings_multiview[:, 1, :]  # [Nwin, D]

        #         B, W = labels.shape
        #         labels_flat = labels.reshape(B * W)
        #         mask_flat = window_mask.reshape(B * W).bool()
        #         labels_valid = labels_flat[mask_flat]  # [Nwin]

        #         # Filter out -1 labels (unlabeled samples)
        #         labeled_mask = (labels_valid != -1)                 # [Nwin]
        #         z_view0_labeled = z_view0[labeled_mask]             # [N_labeled, D]
        #         z_view1_labeled = z_view1[labeled_mask]             # [N_labeled, D]
        #         labels_labeled = labels_valid[labeled_mask]         # [N_labeled] - no -1

        #         # TSCwithQueue forward: (v, v_tilde, y, update_queue)
        #         loss_dict = self.target_supcon_loss_fn(
        #             v=z_view0_labeled,
        #             v_tilde=z_view1_labeled,
        #             y=labels_labeled,
        #             update_queue=True,
        #             current_epoch=current_epoch,
        #             total_epochs=total_epochs,
        #         )

        #         target_supcon_loss = loss_dict["loss"]  # KCL + TSC combined
        #         loss_kcl = loss_dict["loss_kcl"]        # KCL only (for logging)
        #         loss_tsc = loss_dict["loss_tsc"]        # TSC only (for logging)

        # else:
        #     target_supcon_loss = torch.tensor(0.0, device=device, requires_grad=False)
        #     loss_kcl = torch.tensor(0.0, device=device, requires_grad=False)
        #     loss_tsc = torch.tensor(0.0, device=device, requires_grad=False)
        #####################################################################################################################################

        # -------------------- NaN Detection --------------------
        if torch.isnan(bce_loss) or torch.isinf(bce_loss):
            print(f"[WARNING] BCE Loss is NaN/Inf: {bce_loss.item()}")

        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            print(f"[WARNING] CE Loss is NaN/Inf: {ce_loss.item()}")

        if torch.isnan(ucl_loss) or torch.isinf(ucl_loss):
            print(f"[WARNING] Temporal UCL Loss is NaN/Inf: {ucl_loss.item()}")

        # if torch.isnan(scl_loss) or torch.isinf(scl_loss):
        #     print(f"[WARNING] SCL Loss is NaN/Inf: {scl_loss.item()}")

        # if torch.isnan(target_supcon_loss) or torch.isinf(target_supcon_loss):
        #     print(f"[WARNING] Target_SupCon Loss is NaN/Inf: {target_supcon_loss.item()}")

        # -------------------- Total Loss --------------------
        total_loss = (
            bce_weight * bce_loss +
            ce_weight * ce_loss +
            ucl_weight * ucl_loss
        )

        return total_loss, bce_loss, ce_loss, ucl_loss
    
    # validation & test
    def inference(self, classification_input, logits, labels, window_mask):
        return {
            'logits': logits,
            'labels': labels,
            'window_mask': window_mask,
            'window_embeddings': classification_input
        }


class ConstrainttimeLoss(nn.Module):
    """
    Temporal Neighbor Contrastive Loss (Single-View, No Augmentation)

    - 같은 환자의 시간적으로 가까운 window끼리 embedding을 가까이 당김 (pull)
    - 시간 거리에 반비례하는 가중치: w(i,j) = 1 / (beta + |t_i - t_j|)
    - Weighted InfoNCE loss 사용

    Args:
        beta: Time distance weight decay factor (default: 1.0)
    """
    def __init__(self, ucl_beta=None):
        super().__init__()
        self.beta = ucl_beta
        self.very_neg = -1e9

    def forward(self, embeddings, time_indices, batch_indices, temperature, accelerator=None):
        """
        Args:
            embeddings: [N_valid, D] - valid window embeddings (already filtered by model)
            time_indices: [N_valid] - temporal indices for each window
            batch_indices: [N_valid] - batch (patient) indices for each window
            window_mask: [B, W] - not used (already filtered in model)
            temperature: temperature for InfoNCE
            accelerator: accelerator object for logging
        """
        device = embeddings.device
        N = embeddings.size(0)

        # Early exit if not enough samples
        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=False)

        # Normalize embeddings (if not already normalized)
        z = F.normalize(embeddings, p=2, dim=-1)  # [N, D]

        # ------------------------------------------------------------------
        # Time-aware Pull Loss (Weighted InfoNCE)
        # ------------------------------------------------------------------
        # Compute similarity matrix: [N, N]
        sim = torch.matmul(z, z.T) / temperature

        # Mask diagonal (self-contrast)
        mask_self = torch.eye(N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, self.very_neg)

        # Compute time-aware positive mask and weights
        same_patient = batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)  # [N, N]
        time_dist = (time_indices.unsqueeze(1) - time_indices.unsqueeze(0)).abs()  # [N, N]
        mask_pos = same_patient & (time_dist > 0)  # Exclude self, same patient only

        # Time-aware weights: closer in time = higher weight
        w = torch.zeros_like(sim)
        w[mask_pos] = (1.0 / (self.beta + time_dist[mask_pos].float())).to(w.dtype)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # Normalize weights

        # Weighted InfoNCE loss
        pull_loss = -(w * F.log_softmax(sim, dim=1)).sum(dim=1).mean()

        if accelerator is not None and accelerator.is_main_process:
            num_pos_pairs = mask_pos.sum().item()
            total_pairs = N * (N - 1)
            pos_ratio = 100.0 * num_pos_pairs / total_pairs if total_pairs > 0 else 0.0

            # print(f"[Temporal UCL] Loss={pull_loss.item():.4f} | "
            #     f"Positive pairs: {num_pos_pairs}/{total_pairs} ({pos_ratio:.1f}%) | "
            #     f"Samples: {N}")

        return pull_loss



# class TSCwithQueue(nn.Module):
#     """
#     - Loss와 Queue를 관리하는 최상위 모듈
#     """
#     def __init__(self, args, embedding_dim=128, queue_size=8192, n_cls=3, T=0.07, num_positive=6, targeted=True, tr=1, sep_t=True, tw=1.0):
#         super().__init__()
        
#         self.queue = TSCQueue(
#             embedding_dim=embedding_dim,
#             queue_size=queue_size,
#         )

#         self.kcl = KCL(
#             dim=embedding_dim,
#             K=queue_size,
#             T=T,
#             num_positive=num_positive,
#             targeted=targeted,
#             tr=tr,
#             sep_t=sep_t,
#             tw=tw,
#         )

#     def forward(self, v, v_tilde, y, update_queue=True, current_epoch=None, total_epochs=None):
#         queue_v, queue_labels = self.queue.get()

#         logits, _, q, loss, loss_class, loss_target, softmax_self_prob = self.kcl(
#             v=v,
#             v_tilde=v_tilde,
#             y=y,
#             queue_v=queue_v,
#             queue_labels=queue_labels,
#             current_epoch=current_epoch,
#             total_epochs=total_epochs,
#         )

#         if update_queue:
#             with torch.no_grad():
#                 self.queue.enqueue(v, y)

#         return {
#             "loss": loss,
#             "loss_kcl": loss_class,
#             "loss_tsc": loss_target,
#             "logits": logits,
#             "embeddings": q,
#             "softmax_self_prob": softmax_self_prob,
#         }


# class TSCQueue(nn.Module):
#     """
#     TSC를 위한 FIFO Queue
#     - Momentum update 방식은 아님...
#     - 과거 배치의 임베딩과 라벨을 저장하여 대조학습의 negative samples 확보
#     """
#     def __init__(self, embedding_dim, queue_size, device=None):
#         super().__init__()
#         self.queue_size = queue_size
#         self.embedding_dim = embedding_dim

#         # [queue_size, Dim=128]
#         self.register_buffer("queue_embeds", F.normalize(torch.randn(queue_size, embedding_dim), dim=1))
#         self.register_buffer("queue_labels", torch.full((queue_size,), -100, dtype=torch.long)) # 결측 라벨도 아닌 -100을 사용함.
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

#     @torch.no_grad()
#     def enqueue(self, embeddings, labels):
#         """
#         embeddings: [B, D]
#         labels: [B]
#         - 과거 샘플들의 feature와 label을 저장해서 TSC Loss의 분모 V_i를 구성함.
#         1. 새 임베딩을 normalize하여 unit sphere에 위치시킴.
#         2. queue_ptr 위치부터 순서대로 덮어씀.
#         3. Queue 끝을 넘으면 앞으로 돌아옴 (circular buffer)
#         4. Pointer를 다음 위치로 업데이트함.
#         """
#         z = F.normalize(embeddings.detach(), dim=1) # no-grad
#         labels = labels.detach().long()

#         B = z.size(0)                     # 모델에 들어온 배치 크기
#         ptr = int(self.queue_ptr.item())  # 현재 queue에 쓰기 시작할 위치
#         Q = self.queue_size               # 전체 queue 크기

#         # B >= Q인 경우: 배치가 queue 전체보다 크므로 최신 Q개만 사용 (FIFO)
#         if B >= Q:
#             z = z[-Q:]
#             labels = labels[-Q:]
#             B = Q

#         # circular enqueue
#         if ptr + B <= Q:
#             # 한 번에 다 들어가는 경우
#             self.queue_embeds[ptr:ptr + B] = z
#             self.queue_labels[ptr:ptr + B] = labels

#         else:
#             # queue size를 초과하는 경우 (첫 번째 빼고는 전부 else구문을 통해 업데이트될 것임.)
#             first = Q - ptr
#             # 일단 잔여 queue 끝까지 채우기
#             self.queue_embeds[ptr:] = z[:first]
#             self.queue_labels[ptr:] = labels[:first]
#             # 남은 것은 앞부터 채우기
#             self.queue_embeds[:B - first] = z[first:]
#             self.queue_labels[:B - first] = labels[first:]

#         self.queue_ptr[0] = (ptr + B) % Q # FIFO / 포인터도 업데이트함.

#     def get(self):
#         return self.queue_embeds, self.queue_labels


# class KCL(nn.Module):
#     """
#     1. KCL: Queue와 K를 사용해서 샘플링했을 뿐 supervised contrastive learning과 같은 개념
#     - Anchor와 같은 클래스 샘플 = positive
#     - Anchor와 다른 클래스 샘플 = negative
#     - Queue에서 K개의 positive만 샘플링하며, 이를 통해 클래스별 imbalance를 해소함.
#     - 그런데 이미지와 다르게 클래스별 임베딩 간 차이가 크지 않은 우리 연구에는 보완이 필요해보임.
        
#     2. TSC: Target embedding을 사용하여 할당된 Target으로 클래스별 임베딩을 끌어당김.
#     - 각 클래스마다 optimal target embedding 사전 정의
#     - Class centroid ↔ Target embedding Hungarian algorithm based matching
#     - Target을 추가 positive로 사용하여 centroid로 수렴 유도 (전역적으로 밀어내는 효과도 있음.)
#     """
#     def __init__(self,
#         dim=128,            # projection head 후 임베딩 차원
#         K=8192,             # queue size
#         m=0.999,            # momentum encoder update ratio
#         T=0.07,             # temperature
#         num_positive=0,     # anchor 당 positive sampling 개수 (K)
#         targeted=False,     # target-based contrastive loss 사용 여부
#         tr=20, 
#         sep_t=True, 
#         tw=1                # target loss 가중치 (lambda)
#     ):
#         super(KCL, self).__init__()

#         self.K = K                          # queue size
#         self.m = m                          
#         self.T = T                          # temperature
#         self.num_positive = num_positive    # anchor 당 positive sampling 개수 (K)
#         self.n_cls = 3                      # The number of classes
#         self.targeted = targeted            # TSC 사용 여부
#         self.tr = tr                        # Target repeat 횟수
#         self.sep_t = sep_t
#         self.tw = tw                        # target-loss weight

#         # ========================== optimal target embedding ==========================
#         optimal_target = np.load(OUTPUT_DIR + 'targets/optimal_target_{}_{}.npy'.format(self.n_cls, dim)) # [n_cls, dim]
#         optimal_target_order = np.arange(self.n_cls)
#         target_repeat = tr * np.ones(self.n_cls)
#         """
#         분모가 이미 aug pos 1개, K (queue), T_n (targets)로 구성되어 있는데, 이때 queue 8192개에 target이 3개라면 target은 거의 무시될 수 밖에 없는 구조임.
#         - 따라서 target을 tr배하여 분모 내에서 target의 영향력을 높임.
#         - 즉, target이 softmax 분모에서 사라지지 않도록 분모 weight를 맞추는 역할.
#         - 코드에 문제가 없다면, tr에 따른 실험이 필요함 [10, 100, 1000]
#         - (추가) tr을 과하게 설정하지 않는 것이 중요할 것을 보임.
#         """
#         optimal_target = torch.Tensor(optimal_target).float()
#         target_repeat = torch.Tensor(target_repeat).long()
#         optimal_target = torch.cat([optimal_target[i:i + 1, :].repeat(target_repeat[i], 1) for i in range(len(target_repeat))], dim=0)
#         target_labels = torch.cat([torch.Tensor([optimal_target_order[i]]).repeat(target_repeat[i]) for i in range(len(target_repeat))], dim=0).long().unsqueeze(-1)

#         self.register_buffer("optimal_target", optimal_target)
#         self.register_buffer("optimal_target_unique", optimal_target[::self.tr, :].contiguous().transpose(0, 1))
#         self.register_buffer("target_labels", target_labels)

#         # Initialize class centroids (for EMA update during training)
#         self.register_buffer("class_centroid", torch.randn(self.n_cls, dim))

#         # Target Diagnosis
#         print(f"\n{'='*80}")
#         print(f"🎯 TSC Target Embeddings Loaded")
#         print(f"   File: optimal_target_{self.n_cls}_{dim}.npy")
#         print(f"   Shape after repeat (tr={tr}): {self.optimal_target.shape}")
#         print(f"   Expected: [{self.n_cls * tr}, {dim}] = [{self.n_cls}*{tr}, {dim}]")
#         print(f"   Target labels shape: {self.target_labels.shape}")
#         print(f"   Unique targets shape: {self.optimal_target_unique.shape}")
#         print(f"   Target embedding norms (first 3): {F.normalize(self.optimal_target[:3], dim=1).norm(dim=1).cpu().numpy()}")
#         print(f"   First target sample (class 0): {self.optimal_target[0, :5].cpu().numpy()}")
#         print(f"{'='*80}\n")

#     def forward(
#         self,
#         v,                      # projection head output
#         v_tilde,                # augmentation(positive pair)
#         y,                      # labels
#         queue_v,
#         queue_labels,
#         current_epoch=None,     # Current epoch (for staged activation)
#         total_epochs=None,      # Total epochs (for staged activation)
#     ):

#         device = v.device
#         B, D = v.shape
#         q = F.normalize(v, dim=1)         # [B, D] - 현재 배치의 anchor (query view)
#         k = F.normalize(v_tilde, dim=1)   # [B, D] - positive view (key view)

#         qneg = F.normalize(queue_v.detach(), dim=1)  # [Q, D] - queue는 학습 대상이 아니므로 loss가 queue를 통해 과거 임베딩으로 역전파되는 것을 차단함.
#         qlab = queue_labels.detach().long()          # [Q]    - queue에 저장된 임베딩의 클래스 라벨

#         # 사전에 계산해둔 128차원의 target load함.
#         use_target = self.targeted
#         if current_epoch is not None and total_epochs is not None:
#             prev_use_target = hasattr(self, '_prev_use_target') and self._prev_use_target
#             use_target = use_target and (current_epoch >= total_epochs // 2)

#             if use_target and not prev_use_target:
#                 valid_queue_labels = (queue_labels != -100).sum().item()
#                 queue_fill_ratio = valid_queue_labels / self.K * 100

#                 print(f"\n{'='*80}")
#                 print(f"🎯 TSC (Targeted Supervised Contrastive) ACTIVATED!")
#                 print(f"   Epoch: {current_epoch + 1}/{total_epochs}")
#                 print(f"   Queue size: {self.K}")
#                 print(f"   Queue filled: {valid_queue_labels}/{self.K} ({queue_fill_ratio:.1f}%)")
#                 print(f"   Temperature: {self.T}")
#                 print(f"   Target weight (tw): {self.tw}")

#                 # Centroids가 Assigned target에 얼만큼 잘 따라가는지 측정함.
#                 print(f"\n   📊 Centroid-Target Status:")
#                 tgt_unique = F.normalize(self.optimal_target_unique.t(), dim=1).cpu()  # [n_cls, D]
#                 cent = F.normalize(self.class_centroid, dim=1).cpu()  # [n_cls, D]
#                 for c in range(self.n_cls):
#                     dist = (cent[c] - tgt_unique[c]).norm().item()
#                     print(f"      Class {c}: centroid-target distance = {dist:.4f}")

#                 print(f"{'='*80}\n")

#             self._prev_use_target = use_target

#         # compute logits
#         l_pos = torch.einsum("bd,bd->b", q, k).unsqueeze(1) # [B, 1] - anchor와 self positive 간의 유사도 (logit 0번으로 고정함)
#         l_neg_queue = torch.matmul(q, qneg.t())             # [B, K] - anchor q_i와 queue에 저장된 모든 과거 임베딩과의 유사도

#         # ==================== KCL 진단 로그 (매 forward) ====================
#         if not hasattr(self, '_kcl_log_step'):
#             self._kcl_log_step = 0
#         self._kcl_log_step += 1

#         log_interval = 50  # 50 step마다 출력
#         if self._kcl_log_step % log_interval == 1:
#             with torch.no_grad():
#                 # (1) Augmentation 품질: self-positive similarity
#                 pos_sim = l_pos.squeeze(1)  # [B]

#                 # (2) Queue 내 같은/다른 클래스 similarity
#                 y_flat = y.view(-1)  # [B]
#                 qlab_flat = qlab     # [Q]
#                 valid_queue = (qlab_flat != -100)  # -100은 미초기화 슬롯
#                 queue_fill = valid_queue.sum().item()
#                 queue_total = qlab_flat.numel()

#                 # # 🔍 Queue embedding norm 체크 (normalize 제대로 되었는지)
#                 # if valid_queue.any():
#                 #     queue_norms = qneg[valid_queue].norm(dim=1)
#                 # else:
#                 #     queue_norms = None
#                 # anchor_norms = q.norm(dim=1)

#                 # 클래스별 positive 개수 (queue 기준)
#                 pos_counts = []
#                 neg_sims_list = []
#                 pos_sims_list = []
#                 for cls in torch.unique(y_flat):
#                     cls_int = int(cls.item())
#                     if cls_int < 0:
#                         continue
#                     anchor_sel = (y_flat == cls_int)
#                     queue_pos_sel = (qlab_flat == cls_int) & valid_queue
#                     queue_neg_sel = (qlab_flat != cls_int) & valid_queue

#                     n_pos = queue_pos_sel.sum().item()
#                     pos_counts.append((cls_int, anchor_sel.sum().item(), n_pos))

#                     if anchor_sel.any() and queue_pos_sel.any():
#                         sim_pos = l_neg_queue[anchor_sel][:, queue_pos_sel]
#                         pos_sims_list.append(sim_pos.mean().item())
#                     if anchor_sel.any() and queue_neg_sel.any():
#                         sim_neg = l_neg_queue[anchor_sel][:, queue_neg_sel]
#                         neg_sims_list.append(sim_neg.mean().item())

#                 pos_neg_gap = (
#                     (sum(pos_sims_list) / len(pos_sims_list)) -
#                     (sum(neg_sims_list) / len(neg_sims_list))
#                 ) if pos_sims_list and neg_sims_list else float('nan')

#                 print(f"\n{'='*70}")
#                 print(f"[KCL LOG | step {self._kcl_log_step}] use_target={use_target}")
#                 print(f"  Queue: {queue_fill}/{queue_total} filled ({100*queue_fill/queue_total:.1f}%)")
#                 # print(f"  🔍 Normalization Check:")
#                 # print(f"     Anchor norms:  mean={anchor_norms.mean():.4f}  min={anchor_norms.min():.4f}  max={anchor_norms.max():.4f}")
#                 # if queue_norms is not None:
#                 #     print(f"     Queue norms:   mean={queue_norms.mean():.4f}  min={queue_norms.min():.4f}  max={queue_norms.max():.4f}")
#                 # else:
#                 #     print(f"     Queue norms:   N/A (queue empty)")
#                 print(f"  Self-positive sim (aug quality):  mean={pos_sim.mean():.4f}  min={pos_sim.min():.4f}  max={pos_sim.max():.4f}")
#                 if pos_sims_list:
#                     print(f"  Queue same-class sim (pos):       mean={sum(pos_sims_list)/len(pos_sims_list):.4f}")
#                 if neg_sims_list:
#                     print(f"  Queue diff-class sim (neg):       mean={sum(neg_sims_list)/len(neg_sims_list):.4f}")
#                 print(f"  Pos-Neg gap:                       {pos_neg_gap:.4f}  (클수록 좋음)")
#                 print(f"  Class-wise positive count in queue:")
#                 for cls_int, n_anchor, n_pos in pos_counts:
#                     print(f"    Class {cls_int}: anchors={n_anchor}, queue_positives={n_pos}")
#                 print(f"{'='*70}\n")

#         if use_target:
#             tgt = F.normalize(self.optimal_target.to(device), dim=1)  # [Tn, D]    - [n_cls, D]짜리 원형 target을 각 클래스마다 tr번 repeat해서 만듦.
#             l_neg_tgt = torch.matmul(q, tgt.t())                      # [B, Tn]
#             l_neg = torch.cat([l_neg_queue, l_neg_tgt], dim=1)        # [B, K+Tn]
#         else:
#             l_neg = l_neg_queue                             # [B, K]

#         logits = torch.cat([l_pos, l_neg], dim=1) # [B, 1+K(+Tn)]
#         logits = logits / self.T
#         labels_ce = torch.zeros(B, dtype=torch.long, device=device)

#         # same-class positive mask 만들기
#         y = y.view(-1, 1).long()
#         qlab_row = qlab.view(1, -1)

#         if use_target:
#             target_labels = self.target_labels.to(device).t().contiguous()  # [1,Tn]

#             # queue 내부 같은 클래스 샘플을 1차 mask로 정의함
#             # - target은 아직 전부 0으로 봄.
#             mask_queue = (y == qlab_row).float()                                    # [B,K]
#             mask_target = torch.zeros((B, target_labels.size(1)), device=device)    # [B,Tn]
#             mask = torch.cat([mask_queue, mask_target], dim=1)                      # [B,K+Tn]

#             #  class centroid ↔ target assignment
#             """
#             - Class centroid가 어떤 target으로 수렴해야 하는지 결정함.
#             - Class centroid와 target 간의 similarity를 최대화하도록 매칭함.

#             Flow:
#                 1. Class centroid를 EMA 방식으로 업데이트함.
#                 2. Centroid-Target Similarity matrix 계산함 [n_cls, n_cls]
#                 3. 헝가리안 알고리즘으로 최적 매칭 찾기
#                 4. 매칭된 target을 해당 class의 positive로 추가함.
#             """
#             with torch.no_grad():
#                 feat_all = q.detach()

#                 # ==================== Centroid Update (EMA) ====================
#                 centroid_updates = {}
#                 for one_label in torch.unique(y):
#                     one_label_int = int(one_label.item())
                    
#                     if one_label_int < 0 or one_label_int >= self.n_cls: # Skip invalid labels (-1 or out of range)
#                         continue

#                     sel = (y[:, 0] == one_label_int)
#                     if sel.any():
#                         centroid_old = self.class_centroid[one_label_int].clone()
#                         centroid_batch = F.normalize(feat_all[sel].mean(dim=0), dim=0)  # [D]
#                         self.class_centroid[one_label_int] = F.normalize(0.9 * self.class_centroid[one_label_int].to(device) + 0.1 * centroid_batch, dim=0)

#                         # Track centroid movement
#                         centroid_new = self.class_centroid[one_label_int]
#                         movement = (centroid_new - centroid_old.to(device)).norm().item()
#                         centroid_updates[one_label_int] = (sel.sum().item(), movement)

#                 # DEBUG: Print centroid updates (first time TSC activates)
#                 if not hasattr(self, '_centroid_logged') and centroid_updates:
#                     print(f"\n📊 Centroid Update (first TSC batch):")
#                     for cls, (count, movement) in centroid_updates.items():
#                         print(f"   Class {cls}: {count} samples, centroid moved {movement:.6f}")
#                     self._centroid_logged = True

#                 # ==================== Hungarian Matching ====================
#                 cent = F.normalize(self.class_centroid.to(device), dim=1)      # [n_cls, D]
#                 otu = self.optimal_target_unique.to(device)                    # [D, n_cls]
#                 dist = torch.matmul(cent, otu).detach().float().cpu().numpy()  # [n_cls, n_cls] (similarity) - cast to float32 before numpy
#                 row_ind, col_ind = linear_sum_assignment(-dist)                # maximize similarity

#                 # 🔍 ALWAYS PRINT Hungarian Matching (디버깅용)
#                 print(f"\n{'='*80}")
#                 print(f"🔍 Hungarian Matching Result (Step {self._kcl_log_step}):")
#                 print(f"   Centroid-Target Similarity Matrix (값이 높을수록 유사):")
#                 for i in range(self.n_cls):
#                     print(f"   Centroid {i}: {dist[i]}")
#                 print(f"   Assignment: {list(zip(row_ind, col_ind))}")
#                 for cls, tgt_idx in zip(row_ind, col_ind):
#                     print(f"   Class {cls} → Target {tgt_idx} (similarity: {dist[cls, tgt_idx]:.4f})")
#                 print(f"   Centroid norms: {cent.norm(dim=1).cpu().numpy()}")

#                 # Target 벡터 자체 확인
#                 tgt_unique_cpu = otu.t().cpu()  # [n_cls, D]
#                 print(f"\n   Target vector preview (first 5 dims):")
#                 for i in range(self.n_cls):
#                     print(f"   Target {i}: {tgt_unique_cpu[i, :5].numpy()}")
#                 print(f"{'='*80}\n")

#                 # ==================== Matched Target을 Positive로 추가 ====================
#                 for cls, tgt_idx in zip(row_ind, col_ind):
#                     cls = int(cls)
#                     sel = (y[:, 0] == cls)

#                     if not sel.any():
#                         continue

#                     t_indices = torch.arange(tgt_idx * self.tr, tgt_idx * self.tr + self.tr, device=device)

#                     # target이 matching positivie로 이 시점부터 지정됨
#                     sel_indices = torch.where(sel)[0]  # [N_sel]
#                     mask[sel_indices[:, None], qlab.numel() + t_indices[None, :]] = 1.0

#             if self.sep_t:
#                 mask_target_only = mask.clone()
#                 mask_target_only[:, :qlab.numel()] = 0.0
#                 mask_class_only = mask.clone()
#                 mask_class_only[:, qlab.numel():] = 0.0
#             else:
#                 mask_class_only = mask
#                 mask_target_only = mask
#         else:
#             mask = (y == qlab_row).float()
#             mask_class_only = mask
#             mask_target_only = mask

#         mask_pos_view = torch.zeros_like(mask)          # [B,K(+Tn)]

#         # ==================== KCL의 K-positive 샘플링 ====================
#         """
#         - 일종의 hard negative mining 강화 방식임.
#         - 랜덤샘플링과 다르게 가장 효과적인 K 샘플링 방법은 없을까?
#         """
#         if self.num_positive > 0:
#             work_mask = mask.clone()
#             for iteration in range(self.num_positive):
#                 all_pos = work_mask.view(-1).nonzero().view(-1)

#                 # Early exit: positive가 하나도 없으면 종료 (보호장치인데 사실상 작동 안함.)
#                 if all_pos.numel() == 0:          
#                     break

#                 num_pos = work_mask.sum(1)                                   # 각 anchor의 positive 개수                   
#                 num_pos_cum = num_pos.cumsum(0).roll(1)
#                 num_pos_cum[0] = 0

#                 rand = torch.rand(B, device=device)                          # 각 anchor에서 랜덤하게 하나씩 샘플링
#                 idxs = ((rand * num_pos).floor() + num_pos_cum).long()
#                 valid = (num_pos > 0)                                        # Positive가 있는 anchor만 처리
#                 idxs = idxs[valid]

#                 if idxs.numel() > 0:
#                     idxs = idxs.clamp(0, all_pos.numel() - 1)
#                     sampled = all_pos[idxs]
#                     mask_pos_view.view(-1)[sampled] = 1.0
#                     work_mask.view(-1)[sampled] = 0.0
#         else:
#             # K-sampling 비활성화 - 모든 positive 사용
#             mask_pos_view = mask.clone()

#         # 최종 mask 구성 과정
#         if use_target and self.sep_t:
#             mask_pos_view_class = (mask_pos_view * (mask_class_only > 0).float())
#             mask_pos_view_target = (mask_target_only > 0).float()
#             denom_mask = mask_pos_view + (mask_target_only > 0).float()
#         else:
#             mask_pos_view_class = mask_pos_view.clone()
#             mask_pos_view_target = torch.zeros_like(mask_pos_view)
#             denom_mask = mask_pos_view

#         ones_pos = torch.ones((B, 1), device=device)
#         zeros_pos = torch.zeros((B, 1), device=device)

#         mask_all = torch.cat([ones_pos, denom_mask], dim=1)  # [B, 1+K(+Tn)]
#         mask_class = torch.cat([ones_pos, mask_pos_view_class], dim=1)
#         mask_target = torch.cat([zeros_pos, mask_pos_view_target], dim=1)

#         # ==================== Loss 계산 ====================
#         log_prob = F.log_softmax(logits, dim=1)
#         denom = mask_all.sum(1).clamp_min(1.0)

#         loss_class = -((mask_class * log_prob).sum(1) / denom).mean()
#         loss_target = -((mask_target * log_prob).sum(1) / denom).mean()

#         loss_target = loss_target * self.tw
#         loss = loss_class + loss_target

#         return logits, labels_ce, q, loss, loss_class, loss_target

################################################################################################################################
################################################################################################################################

# Global Loss
## Supervised Contrastive Loss (Single-View)
class SupConLoss(nn.Module):
    def __init__(self):
        super(SupConLoss, self).__init__()

    def forward(self, embeddings, labels, window_mask, temperature):
        device = embeddings.device
        D = embeddings.shape[-1]

        # 1) Flatten windows
        feat_flat = embeddings.reshape(-1, D)           # [B*W, D]
        lab_flat = labels.view(-1)                      # [B*W]
        mask_flat = window_mask.view(-1).bool()         # [B*W]

        # 2) Valid window filtering (exclude unlabeled and padded)
        valid = mask_flat & (lab_flat != -1)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        features = feat_flat[valid]     # [N, D]
        labels_valid = lab_flat[valid]  # [N]

        # 3) L2 normalize
        features = F.normalize(features, p=2, dim=-1)

        N = features.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 4) Compute similarity matrix
        logits = torch.div(features @ features.T, temperature)  # [N, N]

        # 5) Create positive mask (same label)
        labels_row = labels_valid.view(-1, 1)
        labels_col = labels_valid.view(1, -1)
        pos_mask = (labels_row == labels_col).float()  # [N, N]

        # 6) Remove self-contrast (diagonal)
        logits_mask = torch.ones_like(pos_mask)
        diag_idx = torch.arange(N, device=device)
        logits_mask[diag_idx, diag_idx] = 0
        pos_mask = pos_mask * logits_mask  # Remove diagonal from positive mask

        # 7) Mask out invalid positions in logits
        logits_masked = logits + (1 - logits_mask) * -1e9

        # 8) Compute log probability
        log_prob = logits - torch.logsumexp(logits_masked, dim=1, keepdim=True)

        # 9) Compute mean of log-likelihood over positive pairs
        pos_per_anchor = pos_mask.sum(1).clamp(min=1e-6)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_per_anchor
        loss = -mean_log_prob_pos.mean()

        return loss

#############################################################################################################
#############################################################################################################
# Grave of codes

# Time-aware pull loss
"""
- Anchor: Window itself
- Positive: 같은 환자의 다른 시간 window
- Pull strength = 시간 거리의 함수

z(t) --strong pull--> z(t+1)
z(t) --weak pull--> z(t+5)
- 시간 축에서의 국소 이웃을 학습하는 loss
"""
# def time_aware_pull_loss(
#     z,                 # [N, D]
#     t_local,           # [N]
#     stay_local,        # [N]
#     temperature=0.1,
#     beta=1.0,
#     very_neg=-1e9
# ):
#     """
#     Single-view time-aware pull loss.
#     Encourages temporally nearby windows of the same patient to cluster.
#     """
#     device = z.device
#     N = z.size(0)

#     z_norm = F.normalize(z, p=2, dim=-1) # L2 Normalize
#     sim = torch.matmul(z_norm, z_norm.T) / temperature  # [N, N]

#     # Remove self-contrast (Diagonal)
#     mask_self = torch.eye(N, dtype=torch.bool, device=device)
#     sim = sim.masked_fill(mask_self, very_neg)

#     # Same-patient & different-time mask
#     same_patient = stay_local.unsqueeze(1) == stay_local.unsqueeze(0)
#     time_dist = (t_local.unsqueeze(1) - t_local.unsqueeze(0)).abs()
#     mask_pos = same_patient & (time_dist > 0)

#     # Time-aware weights
#     w = torch.zeros_like(sim) 
#     w[mask_pos] = (1.0 / (beta + time_dist[mask_pos])).to(w.dtype)
#     w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

#     # Weighted InfoNCE
#     log_prob = F.log_softmax(sim, dim=1)
#     loss = -(w * log_prob).sum(dim=1).mean()

#     return loss


# # Local Loss
# ## Margin 나눠서 실험 수행[0.3, 0.5, 0.8]
# def local_temporal_loss(z, stay_ids, margin=0.5):
#     """
#     Local temporal smoothness loss: pull nearby time windows closer if they're within margin.
#     L = -Σ min(0, ||z_i - z_j||₂ - margin) = Σ max(0, margin - ||z_i - z_j||₂)
#     """
#     device = z.device

#     # Patient-level filter
#     mask = stay_ids[1:] == stay_ids[:-1]

#     if not mask.any():
#         return torch.tensor(0.0, device=device)

#     z_prev = z[:-1][mask]
#     z_next = z[1:][mask]

#     dist = torch.norm(z_prev - z_next, dim=1)

#     # -Σ min(0, dist - margin) = Σ max(0, margin - dist)
#     loss = torch.relu(margin - dist).mean()

#     return loss


# # Time-series embedding level Loss
# class TSReconstructionLoss(nn.Module):
#     def __init__(self, d_model=512, mask_ratio=0.15):
#         super().__init__()
#         self.mask_ratio = mask_ratio
#         self.d_model = d_model

#         self.reconstruction_head = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, d_model)
#         )

#         print(f"[TSReconstructionLoss] Initialized with d_model={d_model}, mask_ratio={mask_ratio}")

#     def forward(self, ts_encoded, seq_valid_mask):
#         """
#         Args:
#             ts_encoded: [B*W, T, 512] - TS encoder output (before QKV adapter)
#             seq_valid_mask: [B, W, T] - 1 for valid timesteps, 0 for padding
#         """
#         device = ts_encoded.device
#         BW, T, D = ts_encoded.shape

#         # Reshape seq_valid_mask from [B, W, T] to [B*W, T]
#         B = seq_valid_mask.size(0)
#         W = seq_valid_mask.size(1)
#         seq_valid_mask = seq_valid_mask.view(B * W, T)  # [B*W, T]

#         # Only process windows with valid timesteps
#         valid_windows = seq_valid_mask.sum(dim=1) > 0  # [B*W]
#         if valid_windows.sum() == 0:
#             return torch.tensor(0.0, device=device, requires_grad=True)

#         ts_valid = ts_encoded[valid_windows]        # [N_valid, T, D]
#         mask_valid = seq_valid_mask[valid_windows]  # [N_valid, T]

#         N_valid = ts_valid.size(0)

#         # Create random mask for valid timesteps only
#         # mask_prob: [N_valid, T], True=mask
#         rand = torch.rand(N_valid, T, device=device)
#         mask_prob = (rand < self.mask_ratio) & mask_valid.bool()  # Only mask valid positions

#         # Ensure at least 1 valid timestep is NOT masked per window
#         valid_count = mask_valid.sum(dim=1)         # [N_valid]
#         mask_count = mask_prob.sum(dim=1)           # [N_valid]
#         all_masked = mask_count >= valid_count      # [N_valid]

#         # Masking correction condition
#         if all_masked.any():
#             # Unmask the first valid timestep in windows that are fully masked
#             for i in all_masked.nonzero(as_tuple=False).squeeze(1):
#                 first_valid_idx = mask_valid[i].nonzero(as_tuple=False)[0, 0]
#                 mask_prob[i, first_valid_idx] = False

#         # Original embeddings - Targets for reconstruction (No gradient flow)
#         targets = ts_valid.clone().detach()  # [N_valid, T, D]

#         # Mask embeddings
#         masked_embeddings = ts_valid.clone()
#         masked_embeddings[mask_prob] = 0.0

#         # Reconstruct masked embeddings
#         reconstructed = self.reconstruction_head(masked_embeddings)                     # [N_valid, T, D]

#         # MSE loss on masked positions
#         loss_mask = mask_prob  # [N_valid, T]
#         if loss_mask.sum() == 0:
#             return torch.tensor(0.0, device=device, requires_grad=True)
#         diff = (reconstructed - targets) ** 2                                           # [N_valid, T, D]
#         loss = (diff * loss_mask.unsqueeze(-1)).sum() / (loss_mask.sum() * D + 1e-8)

#         return loss


##############################################################################################################
##############################################################################################################
# # Unsupervised Contrastive Learning (UCL)
# class UnsupervisedContrastiveLoss(nn.Module):
#     """
#     Modular unsupervised contrastive learning framework.
#     ucl_components (dict): loss component weights + hyperparameters
#     """
#     def __init__(self, ucl_components):
#         super().__init__()
#         self.very_neg = -1e9
#         self.params = ucl_components

#         print(f"[UCL] Initialized with params: {self.params}")

#     def forward(self, cl_embeddings, time_indices, stay_ids, window_mask, temperature, prototypes=None, density=None):
#         device = cl_embeddings.device
#         B, V, W, D = cl_embeddings.shape

#         cl_flat = cl_embeddings.permute(0, 2, 1, 3).reshape(-1, V, D)       # [B*W, V, D]
#         t_flat = time_indices.view(-1)                                      # [B*W]
#         stay_flat = stay_ids.unsqueeze(1).repeat(1, W).view(-1)             # [B*W]
#         m_flat = window_mask.view(-1).bool()                                # [B*W]

#         z_local = cl_flat[m_flat]                                           # [N_local, V, D]
#         t_local = t_flat[m_flat]                                            # [N_local]
#         stay_local = stay_flat[m_flat]                                      # [N_local]

#         N = z_local.size(0)
#         if N < 2:
#             return torch.tensor(0.0, device=device, requires_grad=True)

#         z_anchor = z_local[:, 0, :]                             # [N, D] - Original view
#         z_aug = z_local[:, 1, :] if V > 1 else z_anchor         # [N, D] - Augmented view

#         total_loss = torch.tensor(0.0, device=device)
#         weights_components = {'standard_infonce', 'time_aware_pull', 'prototypical'}

#         for key, value in self.params.items():
#             if key not in weights_components:
#                 continue

#             weight = value
#             if weight == 0.0:
#                 continue

#             if key == 'standard_infonce':
#                 loss = self._standard_infonce_loss(
#                     z_anchor, z_aug, temperature, device
#                 )
#             elif key == 'time_aware_pull':
#                 loss = self._time_aware_pull_loss(
#                     z_anchor, z_aug, t_local, stay_local, temperature, device
#                 )
#             elif key == 'prototypical':
#                 if prototypes is not None and density is not None:
#                     loss = self._prototypical_loss(
#                         z_anchor, prototypes, density, temperature, device
#                     )
#                 else:
#                     # Prototypes not initialized yet (warmup phase)
#                     loss = torch.tensor(0.0, device=device)

#             total_loss += weight * loss

#         return total_loss
    
#     def _standard_infonce_loss(self, z_anchor, z_aug, temperature, device):
#         """
#         Standard InfoNCE loss
#         Each anchor's positive is its augmented view; all others are negatives.
#         """
#         N = z_anchor.size(0)

#         sim_matrix = torch.matmul(z_anchor, z_aug.T) / temperature 
#         labels = torch.arange(N, device=device)

#         loss = F.cross_entropy(sim_matrix, labels)
#         return loss

    # def _time_aware_pull_loss(self, z_anchor, z_aug, t_local, stay_local, temperature, device):
    #     """
    #     Time-aware InfoNCE loss for medical time series.
    #     Positive pairs are weighted by temporal proximity within the same patient.
    #     """
    #     N = z_anchor.size(0)

    #     sim = torch.matmul(z_anchor, z_aug.T) / temperature  # [N, N]

    #     # Mask diagonal (self-contrast)
    #     mask_self = torch.eye(N, dtype=torch.bool, device=device)
    #     sim = sim.masked_fill(mask_self, self.very_neg)

    #     # Identify positive pairs (same patient, different time)
    #     same_patient = stay_local.unsqueeze(1) == stay_local.unsqueeze(0)  # [N, N]
    #     time_dist = (t_local.unsqueeze(1) - t_local.unsqueeze(0)).abs()  # [N, N]
    #     mask_pos = same_patient & (time_dist > 0)  # Exclude self

    #     # Compute time-aware weights
    #     beta = self.params['beta']
    #     w = torch.zeros_like(sim)
    #     w[mask_pos] = (1.0 / (beta + time_dist[mask_pos])).to(w.dtype)
    #     w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # Normalize

    #     # Weighted InfoNCE
    #     loss = -(w * F.log_softmax(sim, dim=1)).sum(dim=1).mean()
    #     return loss

#     def _prototypical_loss(self, z_anchor, prototypes, density, temperature, device):
#         """
#         Prototypical contrastive loss: Each sample is contrasted against prototype centroids with density-based temperature scaling.

#         z_anchor: query embeddings                              # [N, D]
#         prototypes: prototype vectors (K = num_prototypes)      # [K, D]
#         density: temperature scaling factor per prototype       # [K]
#         temperature: base temperature
#         """

#         # Compute similarity: z_anchor vs all prototypes [N, K]
#         sim = torch.matmul(z_anchor, prototypes.T) / temperature

#         # Assign each sample to nearest prototype (pseudo-label)
#         proto_labels = torch.argmax(sim, dim=1)

#         # Density-based dynamic temperature scaling
#         sim = sim / density.unsqueeze(0)
#         loss = F.cross_entropy(sim, proto_labels)

#         return loss


# class PrototypeManager:
#     """
#     Manages prototypes for prototypical contrastive learning using faiss clustering.
#     Based on PCL (Prototypical Contrastive Learning) paper.
#     """
#     def __init__(self, num_prototypes=100, embedding_dim=256, temperature=0.1, device='cuda'):
#         self.num_prototypes = num_prototypes
#         self.embedding_dim = embedding_dim
#         self.temperature = temperature
#         self.device = device

#         # Prototypes (centroids)
#         self.prototypes = None
#         self.density = None  # Concentration estimation for each prototype
#         self.initialized = False
#         print(f"[PrototypeManager] Initialized with {num_prototypes} prototypes, dim={embedding_dim}")

#     @torch.no_grad()
#     def update_prototypes(self, features):
#         """
#         Update prototypes using K-means clustering on features.

#         Args:
#             features: [N, D] normalized embeddings from momentum encoder
#         """

#         features = features.cpu().numpy().astype('float32')
#         N, D = features.shape

#         if N < self.num_prototypes:
#             print(f"[PrototypeManager] Warning: {N} samples < {self.num_prototypes} prototypes. Skipping update.")
#             return

#         # Initialize faiss clustering
#         k = self.num_prototypes
#         clus = faiss.Clustering(D, k)
#         clus.verbose = False
#         clus.niter = 20
#         clus.nredo = 5
#         clus.max_points_per_centroid = 3000
#         clus.min_points_per_centroid = 50

#         # Use CPU for clustering to avoid CUBLAS errors
#         # CPU K-means is stable and fast enough for our use case (runs once per epoch)
#         index = faiss.IndexFlatL2(D)

#         print(f"[PrototypeManager] Running K-means on CPU with {N} samples, {k} clusters...")

#         # Run clustering
#         clus.train(features, index)

#         # Get cluster assignments and distances
#         D_dist, I = index.search(features, 1)  # [N, 1]
#         im2cluster = I.squeeze(1)  # [N]

#         # Get centroids
#         centroids = faiss.vector_to_array(clus.centroids).reshape(k, D)

#         # Compute density (concentration) for each cluster
#         Dcluster = [[] for _ in range(k)]
#         for idx, cluster_id in enumerate(im2cluster):
#             Dcluster[cluster_id].append(D_dist[idx][0])

#         density = np.zeros(k)
#         for i, dist_list in enumerate(Dcluster):
#             if len(dist_list) > 1:
#                 d = (np.asarray(dist_list) ** 0.5).mean() / np.log(len(dist_list) + 10)
#                 density[i] = d

#         # Handle clusters with single points
#         dmax = density.max() if density.max() > 0 else 1.0
#         for i, dist_list in enumerate(Dcluster):
#             if len(dist_list) <= 1:
#                 density[i] = dmax

#         # Clamp extreme values for stability
#         density = np.clip(density, np.percentile(density, 10), np.percentile(density, 90))
#         # Scale the mean to temperature
#         density = self.temperature * density / (density.mean() + 1e-8)

#         # Convert to torch tensors
#         self.prototypes = torch.tensor(centroids, dtype=torch.float32, device=self.device)
#         self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
#         self.density = torch.tensor(density, dtype=torch.float32, device=self.device)

#         self.initialized = True
#         print(f"[PrototypeManager] Updated prototypes: {k} clusters, density range [{density.min():.4f}, {density.max():.4f}]")

#     def get_prototypes(self):
#         """Return current prototypes and density."""
#         if not self.initialized:
#             return None, None
#         return self.prototypes, self.density



##############################################################################################################
# temporal based push-pull mechanism
# class ConstrainttimeLoss(nn.Module):
#     def __init__(self, beta, margin, tau, pull_weight, push_weight, use_tau_cut=True):
#         super().__init__()
#         self.beta = beta
#         self.margin = margin
#         self.tau = tau
#         self.pull_w = pull_weight
#         self.push_w = push_weight
#         self.use_tau_cut = use_tau_cut

#     def forward(self, cl_embeddings, time_indices, batch_indices, window_mask, temperature, accelerator=None):
#         # if False:
#         device = cl_embeddings.device
#         B, V, W, D = cl_embeddings.shape
        
#         # All-gather from all GPUs for better contrastive learning
#         with timer("All-gather for UCL"):

#             cl_flat = cl_embeddings.permute(0, 2, 1, 3).reshape(-1, V, D)
#             t_flat  = time_indices.view(-1)                                # [B*W]
#             b_flat  = batch_indices.view(-1)                               # [B*W]
#             m_flat  = window_mask.view(-1).bool()                          # [B*W]

#             z_local = cl_flat[m_flat]                                      # [N_local, V, D]
#             t_local = t_flat[m_flat]                                       # [N_local]
#             b_local = b_flat[m_flat]                                       # [N_local]

#             if accelerator.num_processes > 1:
#                 world_size = torch.distributed.get_world_size()

#                 z_lists = [None] * world_size
#                 t_lists = [None] * world_size
#                 b_lists = [None] * world_size

#                 torch.distributed.all_gather_object(z_lists, z_local)
#                 torch.distributed.all_gather_object(t_lists, t_local)
#                 torch.distributed.all_gather_object(b_lists, b_local)

#                 z_all = torch.cat([z.to(cl_embeddings.device) for z in z_lists], dim=0)  # [ΣN, V, D]
#                 t_all = torch.cat([t.to(cl_embeddings.device) for t in t_lists], dim=0)  # [ΣN]
#                 b_all = torch.cat([b.to(cl_embeddings.device) for b in b_lists], dim=0)  # [ΣN]

#             else: 
#                 z_all, t_all, b_all = z_local, t_local, b_local

#         # Local anchor, Global contrast 분리
#         # Normalization already done in ProjectionHead, no need to repeat here
#         z_O_local = z_local[:, 0, :]                          # local original view (anchor)
#         z_A_all = z_all[:, 1, :]                              # global augmented view (contrast)
#         # z_A_local = z_local[:, 1, :]                        # local augmented view (anchor)
#         # z_O_all = z_all[:, 0, :]                            # global original view (contrast)

#         if z_O_local.numel() == 0:
#             print("[UCL DEBUG] No valid local features.")
#             print("cl_embeddings:", cl_embeddings)
#             return torch.tensor(0.0, device=device, requires_grad=False)
        
#         t_local = t_local
#         b_local = b_local
#         t_all = t_all
#         b_all = b_all

#         # ---------------------------------------------------------
#         # 1) Time‑aware Pull (InfoNCE + 가중치) - Local anchor vs Global contrast
#         # ---------------------------------------------------------
#         sim_OA = torch.matmul(z_O_local, z_A_all.T) / temperature  # [N_local, N_all]

#         time_dist = (t_local.unsqueeze(1) - t_all.unsqueeze(0)).abs()          # [N_local, N_all]
#         same_seq  = b_local.unsqueeze(1) == b_all.unsqueeze(0)                 # [N_local, N_all] (같은 환자 여부)

#         w = torch.zeros_like(sim_OA)
#         w_dtype = w.dtype
#         w[same_seq] = (1.0 / (self.beta + time_dist[same_seq].float())).to(w_dtype) # 시간 거리에 반비례하는 가중치
#         w = w / (w.sum(dim=1, keepdim=True) + 1e-8)              # normalize
#         pull_loss = - (w * F.log_softmax(sim_OA, dim=1)).sum(dim=1).mean()

#         # ---------------------------------------------------------
#         # 2) Centroid‑based Push (sequence ranking) - Local anchor 기준
#         # ---------------------------------------------------------
#         seq_ids = torch.unique(b_local)

#         centroids = []
#         intra_d = []

#         for sid in seq_ids:
#             idx = (b_local == sid).nonzero(as_tuple=False).squeeze(1)
#             z_seq = z_O_local[idx]           # [n_i, D] - local anchor만 사용
#             c = z_seq.mean(dim=0)            # centroid
#             centroids.append(c)
#             intra_d.append(torch.norm(z_seq - c, dim=1).mean())

#         C = torch.stack(centroids)           # [S,D]
#         intra = torch.stack(intra_d)         # [S]
#         inter = torch.cdist(C, C, p=2)       # [S,S]

#         # τ‑cut: 지나치게 비슷한 다른‑환자 쌍 drop
#         if self.use_tau_cut:
#             # convert dist→cos for unit vec: cosθ = 1 - d^2 / 2
#             sim = 1.0 - inter.pow(2) / 2.0
#             keep_mask = sim < self.tau
#             inter = inter * keep_mask + 1e9 * (~keep_mask)  # 1e9 ensures no ranking loss

#         # ranking loss vectors
#         S = C.size(0)
#         if S < 2:
#             push_loss = cl_embeddings.new_tensor(0.0)
#             total = self.pull_w * pull_loss + self.push_w * push_loss
#             return total

#         else:
#             x1 = intra.repeat_interleave(S - 1)                 # [S*(S-1)]
#             # exclude diagonal for inter distances
#             off_diag_mask = ~torch.eye(S, dtype=torch.bool, device=device)
#             x2 = inter[off_diag_mask]
#             target = -torch.ones_like(x1)

#             push_loss = F.margin_ranking_loss(x1, x2, target, margin=self.margin)

#             total = self.pull_w * pull_loss + self.push_w * push_loss
#         return total