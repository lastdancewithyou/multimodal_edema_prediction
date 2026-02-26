import os
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore', message='Spectral initialisation failed')

import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils import timer

# 단일 배치 학습 함수
def train_batch(args, model, batch, loss_module, device, accelerator, dataset, disable_cxr=False, disable_txt=False, max_length=256,
                bce_weight=None, ce_weight=None, ucl_weight=None
    ):
    
    model.train()

    # ==================== 1. 배치 데이터 GPU 전송 ====================
    with timer("Batch Data preparation", accelerator):
        # New multi-task format
        for k in ['edema_labels', 'subtype_labels', 'window_mask', 'valid_seq_mask']:
            batch[k] = batch[k].to(device, non_blocking=True)
        edema_labels = batch['edema_labels']
        subtype_labels = batch['subtype_labels']

        demo_features = batch.get('demo_features')
        if demo_features is not None:
            demo_features = demo_features.to(device, non_blocking=True)

        img_index_tensor = batch['img_index_tensor']
        txt_index_tensor = batch['text_index_tensor']
        has_cxr = (img_index_tensor != -1).long().to(device, non_blocking=True)   # [B, W, T]
        has_text = (txt_index_tensor != -1).long().to(device, non_blocking=True)  # [B, W, T]

        window_mask = batch['window_mask']
        seq_valid_mask = batch['valid_seq_mask']
        stay_ids = torch.tensor(batch['stay_ids'], dtype=torch.long, device=device)  # [B]

    # ==================== 2. 모델 입력 데이터 전처리 ====================
    with timer("데이터 전처리 작업", accelerator):
        ts_series, cxr_views, text_series, has_cxr, has_text = prepare_multiview_inputs_v2(
            batch, device, has_cxr, has_text,
            dataset=dataset,
            disable_cxr=disable_cxr,
            disable_txt=disable_txt,
            max_length=max_length,
        )

    # ==================== 3. Forward Pass 및 Loss 계산 ====================
    with timer("Batch별 Embedding 추출 및 Loss 연산 총", accelerator):
        with accelerator.autocast():
            time_steps = batch.get('time_steps', None)
            if time_steps is not None:
                time_steps = time_steps.to(device, non_blocking=True)

            model_type = type(model).__name__ # 모델 타입에 따라 forward 분리 수행.

            if "Contrastive" in model_type:  # Stage 1: MultiModalContrastiveModel
                model_outputs = model(
                    args, ts_series, cxr_views, text_series, has_cxr, has_text,
                    window_mask, seq_valid_mask, demo_features, time_steps=time_steps
                )

                # Unpack model outputs
                edema_logits = model_outputs['edema_logits']          # [B, W, 1]
                subtype_logits = model_outputs['subtype_logits']       # [B, W, 2]
                valid_embeddings = model_outputs['valid_embeddings']   # [Nwin, 256]
                window_time_indices = model_outputs['window_time_indices']  # [Nwin]
                batch_indices = model_outputs['batch_indices']         # [Nwin]

            with timer("Main loss 연산", accelerator):
                total_batch_loss, bce_loss_t, ce_loss_t, ucl_loss_t = loss_module(
                    edema_logits = edema_logits,
                    subtype_logits = subtype_logits,
                    valid_embeddings = valid_embeddings,
                    window_time_indices = window_time_indices,
                    batch_indices = batch_indices,
                    edema_labels=edema_labels,
                    subtype_labels=subtype_labels,
                    window_mask=window_mask,
                    bce_weight=bce_weight,
                    ce_weight=ce_weight,
                    ucl_weight=ucl_weight,
                    device=device,
                    accelerator=accelerator
                )

    # ==================== 5. Metrics 수집 ====================
    window_count = window_mask.sum().item()
    batch_bce = float(bce_loss_t.detach().item())
    batch_ce = float(ce_loss_t.detach().item())
    batch_ucl = float(ucl_loss_t.detach().item())

    batch_outputs = {
        'edema_labels': edema_labels,
        'subtype_labels': subtype_labels,
        'edema_logits': edema_logits,
        'subtype_logits': subtype_logits
    }

    batch_counts = {
        'window_count': window_count,
    }

    return total_batch_loss, batch_bce, batch_ce, batch_ucl, batch_outputs, batch_counts


def prepare_multiview_inputs_v2(batch, device, has_cxr, has_text, dataset, disable_cxr=False, disable_txt=False, max_length=256):
    """
    Flow:
        1. Time-series: 단순 GPU 전송 [B, W, T, D]
        2. Text: unique_txt_keys → input_ids, attention_mask 구성
        3. Image: unique_img_paths → 캐시에서 이미지 로드
        4. Positions: 각 unique 데이터가 배치의 어디에 위치하는지 저장
    """
    ts = batch['ts_tensor']
    img_index_tensor = batch['img_index_tensor']
    txt_index_tensor = batch['text_index_tensor']
    unique_img_paths = batch['unique_img_paths']
    unique_txt_keys = batch['unique_txt_keys']

    # ==================== TEXT PREPARATION ====================
    if not disable_txt:
        text_ids_list, text_masks_list = [], []
        for stay_id, hour in unique_txt_keys:
            token = dataset.text_map[stay_id][hour]
            ids = torch.tensor(token['input_ids'], dtype=torch.long)
            mask = torch.tensor(token['attention_mask'], dtype=torch.long)

            current_len = len(ids)
            if current_len > max_length:
                ids = ids[:max_length]
                mask = mask[:max_length]
            elif current_len < max_length:
                pad_len = max_length - current_len
                ids = F.pad(ids, (0, pad_len), value=0)
                mask = F.pad(mask, (0, pad_len), value=0)

            text_ids_list.append(ids)
            text_masks_list.append(mask)

        if len(unique_txt_keys) > 0:
            unique_text_ids = torch.stack(text_ids_list, dim=0).to(device, non_blocking=True)
            unique_text_masks = torch.stack(text_masks_list, dim=0).to(device, non_blocking=True)
        else:
            unique_text_ids = torch.empty(0, max_length, dtype=torch.long, device=device)
            unique_text_masks = torch.empty(0, max_length, dtype=torch.long, device=device)
    else:
        unique_text_ids = torch.empty(0, max_length, dtype=torch.long, device=device)
        unique_text_masks = torch.empty(0, max_length, dtype=torch.long, device=device)

    # ==================== IMG PREPARATION ====================
    if not disable_cxr:
        if len(unique_img_paths) > 0:
            unique_imgs = torch.stack(
                [dataset.load_image_cached(path) for path in unique_img_paths],
                dim=0
            ).to(device, non_blocking=True)
        else:
            num_channels = 3 if dataset.to_3ch else 1
            unique_imgs = torch.empty(0, num_channels, 224, 224, device=device)
    else:
        num_channels = 3 if dataset.to_3ch else 1
        unique_imgs = torch.empty(0, num_channels, 224, 224, device=device)

    ts_series = ts.to(device, non_blocking=True)  # [B, W, T, D]

    # Text data structure
    if not disable_txt:
        valid_positions = (txt_index_tensor != -1).nonzero(as_tuple=False)
        if len(valid_positions) > 0:
            unique_indices = txt_index_tensor[valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]]
            text_data = {
                'unique_input_ids': unique_text_ids,
                'unique_attention_mask': unique_text_masks,
                'unique_indices': unique_indices,
                'positions': valid_positions
            }
        else:
            text_data = {
                'unique_input_ids': torch.empty(0, max_length, dtype=torch.long, device=device),
                'unique_attention_mask': torch.empty(0, max_length, dtype=torch.long, device=device),
                'unique_indices': torch.empty(0, dtype=torch.long, device=device),
                'positions': torch.empty(0, 3, dtype=torch.long, device=device)
            }
    else:
        text_data = {
            'unique_input_ids': torch.empty(0, max_length, dtype=torch.long, device=device),
            'unique_attention_mask': torch.empty(0, max_length, dtype=torch.long, device=device),
            'unique_indices': torch.empty(0, dtype=torch.long, device=device),
            'positions': torch.empty(0, 3, dtype=torch.long, device=device)
        }

    # CXR data structure
    if not disable_cxr:
        valid_positions = (img_index_tensor != -1).nonzero(as_tuple=False)
        if len(valid_positions) > 0:
            unique_indices = img_index_tensor[valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]]
            cxr_data = {
                'unique_images': unique_imgs,
                'unique_indices': unique_indices,
                'positions': valid_positions
            }
        else:
            num_channels = 3 if dataset.to_3ch else 1
            cxr_data = {
                'unique_images': torch.empty(0, num_channels, 224, 224, device=device),
                'unique_indices': torch.empty(0, dtype=torch.long, device=device),
                'positions': torch.empty(0, 3, dtype=torch.long, device=device)
            }
    else:
        num_channels = 3 if dataset.to_3ch else 1
        cxr_data = {
            'unique_images': torch.empty(0, num_channels, 224, 224, device=device),
            'unique_indices': torch.empty(0, dtype=torch.long, device=device),
            'positions': torch.empty(0, 3, dtype=torch.long, device=device)
        }

    return ts_series, cxr_data, text_data, has_cxr, has_text


############################################################################################
############################################################################################
# Grave of codes

# 3D
# def plot_umap_3d(args, window_embeddings_list, window_labels_list, window_pred_list=None, epoch=None, prefix=None, pca_model=None):
#     window_embeddings = torch.cat(window_embeddings_list, dim=0)
#     window_labels = torch.cat(window_labels_list, dim=0)
    
#     window_pred = None
#     if window_pred_list is not None:
#         window_pred = torch.cat(window_pred_list, dim=0)
#         window_pred = window_pred.cpu().numpy()

#     window_embeddings = window_embeddings.cpu().numpy()
#     window_labels = window_labels.cpu().numpy()

#     # PCA 좌표계 고정
#     if pca_model is None:
#         pca_model = PCA(n_components=args.pca_components, random_state=args.random_seed)
#         pca_model.fit(window_embeddings)

#     window_embeddings = pca_model.transform(window_embeddings)

#     # UMAP -> 3D 변환
#     umap_model = UMAP(
#         n_components=3,
#         n_neighbors=args.umap_n_neighbors,
#         min_dist=args.umap_min_dist,
#         metric=args.umap_metric,
#         random_state=args.random_seed
#     )

#     embeddings_3d = umap_model.fit_transform(window_embeddings)

#     umap_df = pd.DataFrame({
#         'x': embeddings_3d[:, 0],
#         'y': embeddings_3d[:, 1],
#         'z': embeddings_3d[:, 2],
#         'label': window_labels
#     })

#     if window_pred is not None:
#         umap_df['pred'] = window_pred

#     label_colors = {
#         0: 'lightcoral',
#         1: 'turquoise', 
#         2: 'blueviolet',
#         -1: 'black'  # Unlabeled
#     }

#     label_mapping = {
#         0: 'Negative',
#         1: 'Noncardiogenic Edema',
#         2: 'Cardiogenic Edema',
#         -1: 'Unlabeled'
#     }

#     # Plotly 3D visualization
#     fig = px.scatter_3d(
#         umap_df,
#         x='x', y='y', z='z',
#         color=umap_df['label'].map(label_mapping),
#         color_discrete_map={
#             label_mapping[k]: v for k, v in label_colors.items()
#         },
#         opacity=0.65,
#         title=f"{prefix.title()} 3D UMAP @ Epoch {epoch}" if epoch is not None else f"{prefix.title()} 3D UMAP"
#     )

#     fig.update_traces(marker=dict(size=2, line=dict(width=0)))
#     fig.update_layout(
#         scene=dict(
#             xaxis_title="component 0",
#             yaxis_title="component 1",
#             zaxis_title="component 2"
#         ),
#         legend_title="Classes"
#     )

#     # 저장
#     save_dir = args.umap_save_dir
#     os.makedirs(save_dir, exist_ok=True)
#     file_base = f"{prefix}_umap_epoch{epoch}" if epoch is not None else f"{prefix}_umap"

#     # HTML (interactive)
#     html_path = os.path.join(save_dir, f"{file_base}.html")
#     fig.write_html(html_path)
#     return pca_model


# def compute_contrastive_metrics(embeddings, labels):
#     """
#     대조학습 품질 직접 측정: Intra-class vs Inter-class distance

#     Args:
#         embeddings: torch.Tensor [N, D] - Embedding vectors
#         labels: torch.Tensor [N] - Class labels

#     Returns:
#         dict with:
#             - intra_dist: 같은 클래스 내 평균 거리 (작을수록 좋음)
#             - inter_dist: 다른 클래스 간 평균 거리 (클수록 좋음)
#             - separation_ratio: inter/intra (클수록 좋음, >2.0 목표)
#     """
#     embeddings_np = embeddings.cpu().numpy()
#     labels_np = labels.cpu().numpy()

#     # Cosine distance matrix
#     dist_matrix = cosine_distances(embeddings_np, embeddings_np)

#     intra_dists = []
#     inter_dists = []

#     unique_labels = np.unique(labels_np)

#     for label in unique_labels:
#         # 같은 클래스 마스크
#         same_class_mask = labels_np == label
#         same_class_indices = np.where(same_class_mask)[0]

#         if len(same_class_indices) < 2:
#             continue

#         # Intra-class distance (같은 클래스 내)
#         for i in range(len(same_class_indices)):
#             for j in range(i+1, len(same_class_indices)):
#                 idx_i, idx_j = same_class_indices[i], same_class_indices[j]
#                 intra_dists.append(dist_matrix[idx_i, idx_j])

#         # Inter-class distance (다른 클래스와)
#         diff_class_indices = np.where(~same_class_mask)[0]
#         if len(diff_class_indices) == 0:
#             continue

#         # 샘플링으로 계산량 감소 (최대 1000개 페어만)
#         max_pairs = min(1000, len(same_class_indices) * len(diff_class_indices))
#         sample_size = min(len(same_class_indices), int(np.sqrt(max_pairs)))

#         sampled_same = np.random.choice(same_class_indices, size=min(sample_size, len(same_class_indices)), replace=False)
#         sampled_diff = np.random.choice(diff_class_indices, size=min(sample_size, len(diff_class_indices)), replace=False)

#         for idx_i in sampled_same:
#             for idx_j in sampled_diff:
#                 inter_dists.append(dist_matrix[idx_i, idx_j])

#     intra_mean = np.mean(intra_dists) if len(intra_dists) > 0 else 0.0
#     inter_mean = np.mean(inter_dists) if len(inter_dists) > 0 else 0.0
#     ratio = inter_mean / intra_mean if intra_mean > 1e-6 else 0.0

#     return {
#         'intra_dist': intra_mean,
#         'inter_dist': inter_mean,
#         'separation_ratio': ratio
#     }

# def compute_class_compactness(embeddings, labels):
#     """
#     각 클래스의 응집도(compactness) 측정

#     Args:
#         embeddings: torch.Tensor [N, D]
#         labels: torch.Tensor [N]

#     Returns:
#         list of dict with class-wise compactness metrics
#     """
#     embeddings_np = embeddings.cpu().numpy()
#     labels_np = labels.cpu().numpy()

#     unique_labels = np.unique(labels_np)
#     class_variances = []

#     for label in unique_labels:
#         class_mask = labels_np == label
#         class_embeddings = embeddings_np[class_mask]

#         if len(class_embeddings) < 2:
#             continue

#         # 클래스 중심
#         centroid = class_embeddings.mean(axis=0)

#         # 중심으로부터 평균 거리
#         distances = np.linalg.norm(class_embeddings - centroid, axis=1)
#         variance = distances.mean()

#         class_variances.append({
#             'class': int(label),
#             'compactness': float(variance),
#             'n_samples': len(class_embeddings)
#         })

#     return class_variances

# def evaluate_contrastive_quality(embeddings, labels, epoch, accelerator):
#     """
#     대조학습 품질 종합 평가

#     Args:
#         embeddings: torch.Tensor [N, D] - Window-level embeddings
#         labels: torch.Tensor [N] - Window-level labels
#         epoch: int - Current epoch number
#         accelerator: Accelerator object

#     Returns:
#         dict with contrastive learning quality metrics
#     """
#     embeddings_np = embeddings.cpu().numpy()
#     labels_np = labels.cpu().numpy()

#     # 1. Silhouette Score (전체적인 클러스터 품질)
#     try:
#         sil_score = silhouette_score(embeddings_np, labels_np, metric='cosine')
#     except Exception as e:
#         print(f"[Warning] Silhouette score computation failed: {e}")
#         sil_score = float('nan')

#     # 2. Intra/Inter distance ratio (대조학습 특화)
#     try:
#         contrastive_metrics = compute_contrastive_metrics(embeddings, labels)
#     except Exception as e:
#         print(f"[Warning] Contrastive metrics computation failed: {e}")
#         contrastive_metrics = {
#             'intra_dist': float('nan'),
#             'inter_dist': float('nan'),
#             'separation_ratio': float('nan')
#         }

#     # 3. Class compactness (클래스별 상세 분석)
#     try:
#         class_stats = compute_class_compactness(embeddings, labels)
#     except Exception as e:
#         print(f"[Warning] Class compactness computation failed: {e}")
#         class_stats = []

#     if accelerator.is_main_process:
#         print(f"\n{'='*60}")
#         print(f"Epoch {epoch} - Contrastive Learning Quality Metrics")
#         print(f"{'='*60}")

#         # Silhouette Score
#         print(f"\n📊 Silhouette Score: {sil_score:.4f}")
#         if sil_score > 0.5:
#             status = "✅ 우수 (Excellent separation)"
#         elif sil_score > 0.3:
#             status = "✅ 양호 (Good separation)"
#         elif sil_score > 0.2:
#             status = "⚠️  보통 (Weak structure)"
#         else:
#             status = "❌ 불량 (Poor separation)"
#         print(f"   Status: {status}")

#         # Contrastive Metrics
#         print(f"\n📏 Distance Metrics:")
#         print(f"   Intra-class distance: {contrastive_metrics['intra_dist']:.4f} (낮을수록 좋음)")
#         print(f"   Inter-class distance: {contrastive_metrics['inter_dist']:.4f} (높을수록 좋음)")
#         print(f"   Separation ratio: {contrastive_metrics['separation_ratio']:.4f}")

#         if contrastive_metrics['separation_ratio'] > 2.0:
#             sep_status = "✅ 우수 (클래스 간 명확히 분리됨)"
#         elif contrastive_metrics['separation_ratio'] > 1.5:
#             sep_status = "✅ 양호 (클래스 간 분리 진행 중)"
#         elif contrastive_metrics['separation_ratio'] > 1.2:
#             sep_status = "⚠️  보통 (약한 분리)"
#         else:
#             sep_status = "❌ 불량 (클래스가 섞여있음)"
#         print(f"   Status: {sep_status}")

#         # Class-wise analysis
#         if len(class_stats) > 0:
#             print(f"\n📦 Class-wise Compactness:")
#             for stats in class_stats:
#                 print(f"   Class {stats['class']}: "
#                       f"compactness={stats['compactness']:.4f}, "
#                       f"n={stats['n_samples']}")

#         # 종합 판단
#         print(f"\n{'='*60}")
#         print(f"💡 종합 판단:")

#         if sil_score > 0.3 or contrastive_metrics['separation_ratio'] > 1.5:
#             print(f"   ✅ 대조학습이 작동하고 있습니다. 계속 학습하세요.")
#         elif sil_score > 0.2 or contrastive_metrics['separation_ratio'] > 1.2:
#             print(f"   ⚠️  약한 클러스터 형성. 5-10 epoch 더 지켜보세요.")
#         else:
#             print(f"   ❌ 클러스터가 형성되지 않음. 전략 변경 고려:")
#             print(f"      1. UCL 비중 줄이기 (ucl_weight 감소)")
#             print(f"      2. Supervised-only로 전환 (UCL 제거)")
#             print(f"      3. Projection head 차원 더 늘리기 (512→768)")
#         print(f"{'='*60}\n")

#     return {
#         'silhouette': sil_score,
#         'intra_dist': contrastive_metrics['intra_dist'],
#         'inter_dist': contrastive_metrics['inter_dist'],
#         'separation_ratio': contrastive_metrics['separation_ratio'],
#         'class_stats': class_stats
#     }


# def prepare_multiview_inputs(batch, device, has_cxr, has_text, num_views, dataset, validation_mode=False, img_augmentor=None, disable_cxr=False, disable_txt=False, max_length=256, valid_seq_mask=None, accelerator=None, disable_augmentation=False):
#     ts = batch['ts_tensor']
#     observed_mask = batch['observed_mask']  # [B, W, T, D] - 1=관측, 0=보간/결측
#     img_index_tensor = batch['img_index_tensor']
#     txt_index_tensor = batch['text_index_tensor']
#     unique_img_paths = batch['unique_img_paths']   # list[str]
#     unique_txt_keys = batch['unique_txt_keys']     # list[(stay_id, hour_slot)]

#     B, W, T, D = ts.shape
#     V = 1 if (validation_mode or disable_augmentation) else num_views

#     has_cxr = has_cxr.unsqueeze(1).expand(-1, V, -1, -1)  # [B, V, W, T]
#     has_text = has_text.unsqueeze(1).expand(-1, V, -1, -1)  # [B, V, W, T]

#     ts_views, cxr_views, text_ids_views = [], [], []
#     aug_masks = []  # Augmentation mask tracking

#     # ==================== TEXT PREPARATION ====================
#     if not disable_txt:
#         text_ids_list, text_masks_list = [], []
#         for stay_id, hour in unique_txt_keys:
#             token = dataset.text_map[stay_id][hour]
#             ids = torch.tensor(token['input_ids'], dtype=torch.long)
#             mask = torch.tensor(token['attention_mask'], dtype=torch.long)

#             # max_length에 맞춰 padding / truncation
#             current_len = len(ids)
#             if current_len > max_length:
#                 ids = ids[:max_length]
#                 mask = mask[:max_length]
#             elif current_len < max_length:
#                 pad_len = max_length - current_len
#                 ids = F.pad(ids, (0, pad_len), value=0)
#                 mask = F.pad(mask, (0, pad_len), value=0)

#             text_ids_list.append(ids)
#             text_masks_list.append(mask)

#         if len(unique_txt_keys) > 0:
#             unique_text_ids = torch.stack(text_ids_list, dim=0).to(device, non_blocking=True)  # [N_unique, max_length]
#             unique_text_masks = torch.stack(text_masks_list, dim=0).to(device, non_blocking=True)
#         else:
#             unique_text_ids = torch.empty(0, max_length, dtype=torch.long, device=device)
#             unique_text_masks = torch.empty(0, max_length, dtype=torch.long, device=device)

#     else:
#         # 텍스트 모달리티 비활성화 시, 빈 텐서로 초기화
#         unique_text_ids = torch.empty(0, max_length, dtype=torch.long, device=device)
#         unique_text_masks = torch.empty(0, max_length, dtype=torch.long, device=device)

#     # ==================== IMG PREPARATION ====================
#     if not disable_cxr:
#         if len(unique_img_paths) > 0 :
#             unique_imgs = torch.stack(
#                 [dataset.load_image_cached(path) for path in unique_img_paths],
#                 dim=0
#             ).to(device, non_blocking=True)
#         else: 
#             unique_imgs = torch.empty(0, 3, 224, 224, device=device)
#     else: 
#         # 이미지 모달리티 비활성화 시, 빈 텐서로 초기화
#         unique_imgs = torch.empty(0, 3, 224, 224, device=device)

#     for v in range(V):
#         # ===============================================
#         # 1) Time-series augmentation
#         # ===============================================
#         with timer("TS augmentation", accelerator):
#             ts_v = ts.clone()
#             aug_mask_v = None
#             if v == 1 and not validation_mode and not disable_augmentation:
#                 ts_v, aug_mask_v = augment_time_series(
#                     ts_v,
#                     valid_seq_mask=valid_seq_mask,  # [B, W, T]: 유효한 timestep만 증강
#                     end_mask_steps=0,
#                     random_mask_ratio=0.1,  # 유효한 timestep의 10% 랜덤 마스킹
#                     feature_mask_prob=0.2,
#                     noise_scale=0.05
#                 )
#             else:
#                 # Validation/view 0: no augmentation
#                 aug_mask_v = torch.ones_like(ts_v)  # 모든 값이 존재

#             ts_views.append(ts_v.to(device, non_blocking=True))
#             aug_masks.append(aug_mask_v)

#         # ===============================================
#         # 2) Text augmentation
#         # ===============================================
#         with timer("Text augmentation", accelerator):
#             if not disable_txt:
#                 ids_v = unique_text_ids.clone()  # [N_unique, max_length]
#                 mask_v = unique_text_masks.clone()

#                 if v == 1 and not validation_mode and not disable_augmentation:
#                     for i in range(len(unique_txt_keys)):
#                         new_ids, new_mask = delete_random_tokens(
#                             ids_v[i], mask_v[i],
#                             delete_prob=0.15,
#                             special_token_ids=[101, 102],  # BioBERT [CLS], [SEP]
#                             max_length=max_length
#                         )
#                         ids_v[i] = new_ids
#                         mask_v[i] = new_mask

#                 # Unique 텍스트 + scatter 인덱스 저장
#                 valid_positions = (txt_index_tensor != -1).nonzero(as_tuple=False)  # [N_valid, 3] (b,w,t)

#                 if len(valid_positions) > 0:
#                     # 유효한 위치의 unique index 추출 (scatter용)
#                     unique_indices = txt_index_tensor[valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]]

#                     text_ids_views.append({
#                         'unique_input_ids': ids_v,           # [N_unique, max_length]
#                         'unique_attention_mask': mask_v,     # [N_unique, max_length]
#                         'unique_indices': unique_indices,    # [N_valid] ← Scatter 인덱스
#                         'positions': valid_positions         # [N_valid, 3] with (b, w, t)
#                     })
#                 else:
#                     # 이 view에 유효한 텍스트가 없는 경우
#                     text_ids_views.append({
#                         'unique_input_ids': torch.empty(0, max_length, dtype=torch.long, device=device),
#                         'unique_attention_mask': torch.empty(0, max_length, dtype=torch.long, device=device),
#                         'unique_indices': torch.empty(0, dtype=torch.long, device=device),
#                         'positions': torch.empty(0, 3, dtype=torch.long, device=device)
#                     })

#             else:
#                 # Text 모달리티 비활성화 시
#                 text_ids_views.append({
#                     'unique_input_ids': torch.empty(0, max_length, dtype=torch.long, device=device),
#                     'unique_attention_mask': torch.empty(0, max_length, dtype=torch.long, device=device),
#                     'unique_indices': torch.empty(0, dtype=torch.long, device=device),
#                     'positions': torch.empty(0, 3, dtype=torch.long, device=device)
#                 })

#         # ===============================================
#         # 3) CXR augmentation
#         # ===============================================
#         with timer("Img augmentation", accelerator):
#             if not disable_cxr:
#                 cxr_v = unique_imgs.clone()
#                 if v == 1 and not validation_mode and img_augmentor is not None and not disable_augmentation:
#                     cxr_v = img_augmentor(cxr_v)  # N_unique만 증강

#                 # Unique 이미지 + scatter 인덱스 저장
#                 valid_positions = (img_index_tensor != -1).nonzero(as_tuple=False)  # [N_valid, 3] (b,w,t)

#                 if len(valid_positions) > 0:
#                     # 유효한 위치의 unique index 추출 (scatter용)
#                     unique_indices = img_index_tensor[valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]]

#                     cxr_views.append({
#                         'unique_images': cxr_v,              # [N_unique, 3, 224, 224] ← Unique만!
#                         'unique_indices': unique_indices,    # [N_valid] ← Scatter 인덱스
#                         'positions': valid_positions         # [N_valid, 3] with (b, w, t)
#                     })
#                 else:
#                     # 이 view에 유효한 이미지가 없는 경우
#                     cxr_views.append({
#                         'unique_images': torch.empty(0, 3, 224, 224, device=device),
#                         'unique_indices': torch.empty(0, dtype=torch.long, device=device),
#                         'positions': torch.empty(0, 3, dtype=torch.long, device=device)
#                     })

#             else:
#                 # CXR 모달리티 비활성화 시
#                 cxr_views.append({
#                     'unique_images': torch.empty(0, 3, 224, 224, device=device),
#                     'unique_indices': torch.empty(0, dtype=torch.long, device=device),
#                     'positions': torch.empty(0, 3, dtype=torch.long, device=device)
#                 })

#     ts_series = torch.stack(ts_views, dim=1)  # [B, V, W, T, D]
#     aug_masks = torch.stack(aug_masks, dim=1)  # [B, V, W, T, D]

#     # ==================== Effective Mask 계산 ====================
#     # effective_mask = observed_mask * aug_mask
#     # 1 = 관측되고 증강 미적용 (신뢰 가능)
#     # 0 = 보간/결측이거나 증강됨 (신뢰 불가)
#     # observed_mask: 1=관측, 0=보간/결측
#     # aug_mask: 1=값이 존재, 0=값이 제거됨 (augmentation으로 마스킹됨)
#     observed_mask_expanded = observed_mask.unsqueeze(1).repeat(1, V, 1, 1, 1)  # [B, V, W, T, D]
#     effective_mask = observed_mask_expanded * aug_masks

#     # =======================================================
#     # total_img_positions = sum(len(view['positions']) for view in cxr_views) if not disable_cxr else 0
#     # total_txt_positions = sum(len(view['positions']) for view in text_ids_views) if not disable_txt else 0

#     # print(f"[prepare_multiview] B={B}, W={W}, T={T}, V={V}")
#         # f"Unique imgs={len(unique_img_paths)}, Positions={total_img_positions} ({total_img_positions/len(unique_img_paths) if len(unique_img_paths)>0 else 0:.1f}x) | "
#         # f"Unique txts={len(unique_txt_keys)}, Positions={total_txt_positions} ({total_txt_positions/len(unique_txt_keys) if len(unique_txt_keys)>0 else 0:.1f}x)")
#     # =======================================================
#     return ts_series, cxr_views, text_ids_views, has_cxr, has_text, effective_mask