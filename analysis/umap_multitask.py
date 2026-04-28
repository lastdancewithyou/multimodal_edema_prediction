import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from umap import UMAP
from sklearn.decomposition import PCA
import torch

from training.engine import prepare_multiview_inputs


def plot_multitask_umap(args, model, dataloader, dataset, epoch, save_dir, max_samples=None, umap_reducers=None):
    is_train_mode = (umap_reducers is None)
    if is_train_mode:
        print("=====Generating Multi-Task UMAP Visualizations =====")
    else:
        print("=====Generating Multi-Task UMAP Visualizations =====")
    model.eval()

    all_edema_embeddings = []
    all_subtype_embeddings = []
    all_edema_labels = []
    all_subtype_labels = []
    all_edema_preds = []
    all_subtype_preds = []

    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting embeddings for UMAP"):
            edema_labels = batch['edema_labels']
            subtype_labels = batch['subtype_labels']

            img_index_tensor = batch['img_index_tensor']
            txt_index_tensor = batch['text_index_tensor']
            has_cxr = (img_index_tensor != -1).long()
            has_text = (txt_index_tensor != -1).long()

            ts_series, cxr_views, text_series, has_cxr, has_text = prepare_multiview_inputs(
                batch, has_cxr, has_text,
                dataset=dataset,
                disable_cxr=args.disable_cxr,
                disable_txt=args.disable_txt,
                max_length=args.token_max_length,
                is_training=False
            )

            time_steps = batch['time_steps']

            prompt_data = {
                'unique_prompt_texts': batch['unique_prompt_texts'],
                'prompt_index_tensor': batch['prompt_index_tensor']
            }

            # Move all inputs to model device (CPU tensors from dataloader → GPU)
            device = next(base_model.parameters()).device
            ts_series = ts_series.to(device)
            has_cxr = has_cxr.to(device)
            has_text = has_text.to(device)
            time_steps = time_steps.to(device)
            for k in ['unique_images', 'unique_indices', 'positions']:
                if isinstance(cxr_views[k], torch.Tensor):
                    cxr_views[k] = cxr_views[k].to(device)
            for k in ['unique_input_ids', 'unique_attention_mask', 'unique_indices', 'positions']:
                if isinstance(text_series[k], torch.Tensor):
                    text_series[k] = text_series[k].to(device)

            model_outputs = base_model(args, ts_series, cxr_views, text_series, prompt_data, has_cxr, has_text, time_steps=time_steps)

            # Extract latent embeddings: [B, L, 256]
            batch_embeddings = model_outputs['batch_embeddings']
            B = batch_embeddings.size(0)

            # Edema task embedding and prediction
            edema_q = base_model.edema_readout.query.expand(B, -1, -1)
            edema_attn_out, _ = base_model.edema_readout.cross_attn(
                query=edema_q,
                key=batch_embeddings,
                value=batch_embeddings
            )
            edema_emb = edema_attn_out.reshape(B, -1)
            # edema_logits = base_model.edema_readout.classifier(edema_attn_out.mean(dim=1))
            B = edema_attn_out.size(0)
            # 4개의 쿼리를 1024차원으로 Flatten (모델의 forward 로직과 동일하게)
            flat_edema_out = edema_attn_out.reshape(B, -1) 
            edema_logits = base_model.edema_readout.classifier(flat_edema_out)
            edema_preds = (torch.sigmoid(edema_logits).squeeze(-1) > 0.8).long()  # [B] - 0 or 1

            # Subtype task embedding and prediction
            subtype_q = base_model.subtype_readout.query.expand(B, -1, -1)
            subtype_attn_out, _ = base_model.subtype_readout.cross_attn(
                query=subtype_q,
                key=batch_embeddings,
                value=batch_embeddings
            )
            subtype_emb = subtype_attn_out.reshape(B, -1)
            # subtype_logits = base_model.subtype_readout.classifier(subtype_attn_out.mean(dim=1))
            subtype_logits = base_model.subtype_readout.classifier(subtype_emb)
            subtype_preds = torch.argmax(subtype_logits, dim=-1)  # [B] - 0, 1, or 2

            all_edema_embeddings.append(edema_emb.cpu())
            all_subtype_embeddings.append(subtype_emb.cpu())
            all_edema_labels.append(edema_labels.cpu())
            all_subtype_labels.append(subtype_labels.cpu())
            all_edema_preds.append(edema_preds.cpu())
            all_subtype_preds.append(subtype_preds.cpu())

    # Concatenate all embeddings, labels, and predictions
    edema_embeddings = torch.cat(all_edema_embeddings, dim=0).numpy()
    subtype_embeddings = torch.cat(all_subtype_embeddings, dim=0).numpy()
    edema_labels = torch.cat(all_edema_labels, dim=0).numpy()
    subtype_labels = torch.cat(all_subtype_labels, dim=0).numpy()
    edema_preds = torch.cat(all_edema_preds, dim=0).numpy()
    subtype_preds = torch.cat(all_subtype_preds, dim=0).numpy()

    total_samples = len(edema_labels)
    print(f"Collected {total_samples} total samples")
    print(f"Edema embeddings shape: {edema_embeddings.shape}")
    print(f"Subtype embeddings shape: {subtype_embeddings.shape}")

    # Random sampling if input more samples than max_samples
    if max_samples is not None and total_samples > max_samples:
        print(f"Sampling {max_samples} samples (prioritizing labeled data) from {total_samples} for UMAP")
        np.random.seed(args.random_seed)
        
        # 1. 라벨 유무에 따라 인덱스 분리 (edema_labels가 -1이 아니면 라벨이 있는 것으로 간주)
        labeled_indices = np.where(edema_labels != -1)[0]
        unlabeled_indices = np.where(edema_labels == -1)[0]
        
        # 2. 라벨 데이터 먼저 채우기
        n_labeled_to_pick = min(len(labeled_indices), max_samples)
        if len(labeled_indices) > max_samples:
            # 라벨 데이터만으로 max_samples를 넘는다면 라벨 데이터 안에서만 랜덤 추출
            selected_labeled = np.random.choice(labeled_indices, size=n_labeled_to_pick, replace=False)
        else:
            # 라벨 데이터가 max_samples보다 적으면 전부 다 가져옴
            selected_labeled = labeled_indices
            
        # 3. 남은 자리를 라벨 없는 데이터에서 랜덤 추출하여 채우기
        n_unlabeled_to_pick = max_samples - n_labeled_to_pick
        if n_unlabeled_to_pick > 0:
            selected_unlabeled = np.random.choice(unlabeled_indices, size=n_unlabeled_to_pick, replace=False)
        else:
            selected_unlabeled = np.array([], dtype=int)
            
        # 4. 최종 선택된 인덱스 병합
        sample_indices = np.concatenate([selected_labeled, selected_unlabeled])
        
        # (선택) 배열 순서를 무작위로 섞어줌
        np.random.shuffle(sample_indices)

        edema_embeddings = edema_embeddings[sample_indices]
        subtype_embeddings = subtype_embeddings[sample_indices]
        edema_labels = edema_labels[sample_indices]
        subtype_labels = subtype_labels[sample_indices]
        edema_preds = edema_preds[sample_indices]
        subtype_preds = subtype_preds[sample_indices]
        
        print(f"Using {len(selected_labeled)} labeled and {len(selected_unlabeled)} unlabeled samples for UMAP")
    else:
        print(f"Using all {total_samples} samples for UMAP (no sampling)")

    os.makedirs(save_dir, exist_ok=True)

    if is_train_mode:
        fitted_reducers = {}

    # ============== Plot 1: Binary Edema (0 vs 1) - Task-Specific Embeddings ==============
    # Use ALL data (both labeled and unlabeled) for UMAP projection
    edema_emb = edema_embeddings  # All samples
    edema_lbl = edema_labels      # May contain -1 for unlabeled
    edema_pred = edema_preds      # Predictions for all samples

    if is_train_mode:
        # Training: fit PCA + UMAP on labeled data only
        labeled_mask = (edema_lbl != -1)
        edema_emb_labeled = edema_emb[labeled_mask]

        pca_dim = min(50, edema_emb_labeled.shape[0], edema_emb_labeled.shape[1])
        pca_edema = PCA(n_components=pca_dim, random_state=args.random_seed)
        pca_edema.fit(edema_emb_labeled)

        umap_edema = UMAP(n_components=2, n_neighbors=50, min_dist=0.0, spread=1.0, metric='cosine', random_state=args.random_seed)
        edema_emb_pca_labeled = pca_edema.transform(edema_emb_labeled)
        umap_edema.fit(edema_emb_pca_labeled)

        fitted_reducers['edema'] = {'pca': pca_edema, 'umap': umap_edema}
        print(f"[Train] Fitted PCA + UMAP on {labeled_mask.sum()} labeled samples for Binary Edema")
    else:
        pca_edema = umap_reducers['edema']['pca']
        umap_edema = umap_reducers['edema']['umap']
        print(f"[Val] Using Train PCA + UMAP for Binary Edema")

    # Transform ALL data (labeled + unlabeled)
    edema_emb_pca = pca_edema.transform(edema_emb)
    edema_2d = umap_edema.transform(edema_emb_pca)

    # Visualization: Separate labeled and unlabeled data
    fig, ax = plt.subplots(figsize=(12, 9))

    # Define colors
    colors = {0: '#9E9E9E', 1: '#1565C0'}
    colors_light = {0: '#E0E0E0', 1: '#90CAF9'}  # Lighter versions for unlabeled
    labels_map = {0: 'No Edema', 1: 'Edema'}

    # Plot unlabeled data first (background layer, using predictions)
    unlabeled_mask = (edema_lbl == -1)
    if unlabeled_mask.sum() > 0:
        for pred_lbl in [0, 1]:
            mask = unlabeled_mask & (edema_pred == pred_lbl)
            if mask.sum() > 0:
                ax.scatter(edema_2d[mask, 0], edema_2d[mask, 1],
                          c=colors_light[pred_lbl],
                          label=f'{labels_map[pred_lbl]} (unlabeled, n={mask.sum()})',
                          s=15, alpha=0.4, edgecolors='gray', linewidths=0.3)

    # Plot labeled data on top (foreground layer, darker and larger)
    labeled_mask = (edema_lbl != -1)
    if labeled_mask.sum() > 0:
        for lbl in [0, 1]:
            mask = labeled_mask & (edema_lbl == lbl)
            if mask.sum() > 0:
                ax.scatter(edema_2d[mask, 0], edema_2d[mask, 1],
                          c=colors[lbl],
                          label=f'{labels_map[lbl]} (labeled, n={mask.sum()})',
                          s=20, alpha=0.8, edgecolors='none')

    ax.legend(fontsize=11, loc='best')
    ax.set_title(f'Binary Edema Detection (Task-Specific Embeddings) - Epoch {epoch}', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    save_path = os.path.join(save_dir, f'umap_binary_edema_epoch{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

    # ============== Plot 2: Subtype Classification - Task-Specific Embeddings ==============
    # Include all samples where edema prediction is 1 (both labeled and unlabeled)
    # For labeled data: use edema_labels == 1
    # For unlabeled data: use edema_preds == 1

    # Create mask for samples to include
    labeled_edema = (edema_labels == 1)
    unlabeled_edema_pred = (edema_labels == -1) & (edema_preds == 1)
    include_mask = labeled_edema | unlabeled_edema_pred

    if include_mask.sum() > 0:
        subtype_emb = subtype_embeddings[include_mask]
        subtype_lbl = subtype_labels[include_mask]
        subtype_pred = subtype_preds[include_mask]

        if is_train_mode:
            # Training: fit PCA + UMAP on labeled subtype data only
            # Labeled subtype data: edema_labels==1 AND subtype_labels in [0,1,2]
            labeled_subtype_mask = (subtype_lbl != -1)
            subtype_emb_labeled = subtype_emb[labeled_subtype_mask]

            if labeled_subtype_mask.sum() > 0:
                pca_dim = min(50, subtype_emb_labeled.shape[0], subtype_emb_labeled.shape[1])
                pca_subtype = PCA(n_components=pca_dim, random_state=args.random_seed)
                pca_subtype.fit(subtype_emb_labeled)

                umap_subtype = UMAP(n_components=2, n_neighbors=50, min_dist=0.0, spread=1.0, metric='cosine', random_state=args.random_seed)
                subtype_emb_pca_labeled = pca_subtype.transform(subtype_emb_labeled)
                umap_subtype.fit(subtype_emb_pca_labeled)

                fitted_reducers['subtype'] = {'pca': pca_subtype, 'umap': umap_subtype}
                print(f"[Train] Fitted PCA + UMAP on {labeled_subtype_mask.sum()} labeled samples for Subtype Classification")
            else:
                print("[Warning] No labeled subtype data available for fitting")
                return fitted_reducers if is_train_mode else None
        else:
            pca_subtype = umap_reducers['subtype']['pca']
            umap_subtype = umap_reducers['subtype']['umap']
            print(f"[Val] Using Train PCA + UMAP for Subtype Classification")

        # Transform ALL data (labeled + unlabeled with edema prediction)
        subtype_emb_pca = pca_subtype.transform(subtype_emb)
        subtype_2d = umap_subtype.transform(subtype_emb_pca)

        # Visualization: Separate labeled and unlabeled data
        fig, ax = plt.subplots(figsize=(12, 9))

        # Define colors (0: Non-cardiogenic, 1: Cardiogenic, 2: Mixed/Unknown)
        colors = {0: '#42A5F5', 1: '#E53935', 2: '#66BB6A'}
        colors_light = {0: '#90CAF9', 1: '#EF9A9A', 2: '#A5D6A7'}  # Lighter versions for unlabeled
        labels_map = {0: 'Non-cardiogenic', 1: 'Cardiogenic', 2: 'Mixed/Unknown'}

        # Plot unlabeled data first (background layer, using predictions)
        unlabeled_mask = (subtype_lbl == -1)
        if unlabeled_mask.sum() > 0:
            for pred_lbl in [0, 1, 2]:
                mask = unlabeled_mask & (subtype_pred == pred_lbl)
                if mask.sum() > 0:
                    ax.scatter(subtype_2d[mask, 0], subtype_2d[mask, 1],
                              c=colors_light[pred_lbl],
                              label=f'{labels_map[pred_lbl]} (unlabeled, n={mask.sum()})',
                              s=15, alpha=0.4, edgecolors='gray', linewidths=0.3)

        # Plot labeled data on top (foreground layer, darker and larger)
        labeled_mask = (subtype_lbl != -1)
        if labeled_mask.sum() > 0:
            for lbl in [0, 1, 2]:
                mask = labeled_mask & (subtype_lbl == lbl)
                if mask.sum() > 0:
                    ax.scatter(subtype_2d[mask, 0], subtype_2d[mask, 1],
                              c=colors[lbl],
                              label=f'{labels_map[lbl]} (labeled, n={mask.sum()})',
                              s=20, alpha=0.8, edgecolors='none')

        ax.legend(fontsize=11, loc='best')
        ax.set_title(f'Subtype Classification - Task-Specific Embeddings (Edema cases) - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = os.path.join(save_dir, f'umap_subtype_epoch{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    if is_train_mode:
        print("✅ Multi-Task UMAP Visualization Complete")
        return fitted_reducers
    else:
        print("✅ Multi-Task UMAP Visualization Complete")
        return None