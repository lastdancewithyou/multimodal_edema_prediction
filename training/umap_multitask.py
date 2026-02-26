import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from umap import UMAP
from sklearn.decomposition import PCA
import torch

from training.engine import prepare_multiview_inputs_v2


def plot_multitask_umap(args, model, dataloader, device, accelerator, dataset, epoch, save_dir, max_samples=None,
                        umap_reducers=None):
    """
    Multi-task Learning UMAP Visualization:
    Creates 3 plots:
    1. Binary Edema embedding space (0 vs 1) - using window embeddings
    2. Subtype embedding space (0 vs 1, edema=1 only) - using window embeddings
    3. Combined 3-class visualization (0, 1, 2) - using window embeddings

    Args:
        max_samples: Maximum number of samples to use for UMAP. If None, use all samples.
        umap_reducers: Dict of pre-fitted reducers:
                       {
                           'edema': {'pca': pca_obj, 'umap': umap_obj},
                           'subtype': {'pca': pca_obj, 'umap': umap_obj},
                           'combined': {'pca': pca_obj, 'umap': umap_obj}
                       }
                       If None, fit new reducers (training mode). If provided, use transform only (validation mode).

    Returns:
        reducers: Dict of fitted PCA + UMAP reducers (only if umap_reducers is None, i.e., training mode)
    """
    is_train_mode = (umap_reducers is None)
    if is_train_mode:
        print("=====Generating Multi-Task UMAP Visualizations =====")
    else:
        print("=====Generating Multi-Task UMAP Visualizations =====")
    model.eval()

    all_window_embeddings = []
    all_edema_labels = []
    all_subtype_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting embeddings for UMAP"):

            img_index_tensor = batch['img_index_tensor']
            txt_index_tensor = batch['text_index_tensor']
            has_cxr = (img_index_tensor != -1).long().to(device, non_blocking=True)
            has_text = (txt_index_tensor != -1).long().to(device, non_blocking=True)

            edema_labels = batch['edema_labels'].to(device)
            subtype_labels = batch['subtype_labels'].to(device)
            window_mask = batch['window_mask'].to(device)
            seq_valid_mask = batch['valid_seq_mask'].to(device)

            demo_features = batch.get('demo_features')
            if demo_features is not None:
                demo_features = demo_features.to(device, non_blocking=True)

            ts_series, cxr_data, text_data, has_cxr, has_text = prepare_multiview_inputs_v2(
                batch, device, has_cxr, has_text, dataset,
                disable_cxr=args.disable_cxr,
                disable_txt=args.disable_txt,
                max_length=args.token_max_length
            )

            time_steps = batch.get('time_steps', None)
            if time_steps is not None:
                time_steps = time_steps.to(device, non_blocking=True)

            # Forward pass to get model outputs
            outputs = model(
                args, ts_series, cxr_data, text_data, has_cxr, has_text,
                window_mask, seq_valid_mask, demo_features, time_steps=time_steps
            )

            # Extract window embeddings
            window_embeddings_bw = outputs['window_embeddings']  # [B, W, 256]

            # Flatten window embeddings: [B, W, 256] -> [B*W, 256]
            B, W = window_mask.shape
            window_embeddings_flat = window_embeddings_bw.reshape(B * W, -1)
            edema_labels_flat = edema_labels.reshape(-1)
            subtype_labels_flat = subtype_labels.reshape(-1)
            window_mask_flat = window_mask.reshape(-1)

            # Filter valid windows AND labeled windows only
            valid_mask = window_mask_flat.bool()
            labeled_mask = (edema_labels_flat != -1)  # Exclude unlabeled windows
            combined_mask = valid_mask & labeled_mask

            valid_window_emb = window_embeddings_flat[combined_mask]
            valid_edema = edema_labels_flat[combined_mask]
            valid_subtype = subtype_labels_flat[combined_mask]

            all_window_embeddings.append(valid_window_emb.cpu())
            all_edema_labels.append(valid_edema.cpu())
            all_subtype_labels.append(valid_subtype.cpu())

    # Concatenate all
    window_embeddings = torch.cat(all_window_embeddings, dim=0).numpy()
    edema_labels = torch.cat(all_edema_labels, dim=0).numpy()
    subtype_labels = torch.cat(all_subtype_labels, dim=0).numpy()

    total_samples = len(window_embeddings)
    print(f"Collected {total_samples} total samples")

    # Random sampling if max_samples is specified and we have more samples than max_samples
    if max_samples is not None and total_samples > max_samples:
        print(f"Randomly sampling {max_samples} samples from {total_samples} for UMAP visualization")
        np.random.seed(args.random_seed)
        sample_indices = np.random.choice(total_samples, size=max_samples, replace=False)
        window_embeddings = window_embeddings[sample_indices]
        edema_labels = edema_labels[sample_indices]
        subtype_labels = subtype_labels[sample_indices]
        print(f"Using {len(window_embeddings)} sampled windows for UMAP")
    else:
        print(f"Using all {total_samples} samples for UMAP (no sampling)")

    os.makedirs(save_dir, exist_ok=True)

    # Initialize reducer storage for training mode
    if is_train_mode:
        fitted_reducers = {}

    # ============== Plot 1: Binary Edema (0 vs 1) - Window Embeddings ==============
    edema_valid_mask = (edema_labels != -1)
    if edema_valid_mask.sum() > 0:
        edema_emb = window_embeddings[edema_valid_mask]
        edema_lbl = edema_labels[edema_valid_mask]

        if is_train_mode:
            # Training: fit new PCA + UMAP
            pca_dim = min(50, edema_emb.shape[0], edema_emb.shape[1])
            pca_edema = PCA(n_components=pca_dim, random_state=args.random_seed)
            edema_emb_pca = pca_edema.fit_transform(edema_emb)

            umap_edema = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=args.random_seed)
            edema_2d = umap_edema.fit_transform(edema_emb_pca)

            fitted_reducers['edema'] = {'pca': pca_edema, 'umap': umap_edema}
            print(f"[Train] Fitted PCA + UMAP for Binary Edema")
        else:
            # Validation: transform only using pre-fitted PCA + UMAP
            pca_edema = umap_reducers['edema']['pca']
            umap_edema = umap_reducers['edema']['umap']

            edema_emb_pca = pca_edema.transform(edema_emb)
            edema_2d = umap_edema.transform(edema_emb_pca)
            print(f"[Val] Transformed using Train PCA + UMAP for Binary Edema")

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = {0: '#9E9E9E', 1: '#1565C0'}
        labels_map = {0: 'No Edema', 1: 'Edema'}

        for lbl in [0, 1]:
            mask = (edema_lbl == lbl)
            if mask.sum() > 0:
                ax.scatter(edema_2d[mask, 0], edema_2d[mask, 1],
                        c=colors[lbl], label=f'{labels_map[lbl]} (n={mask.sum()})', s=3, alpha=0.6, edgecolors='none')

        ax.legend(fontsize=12)
        ax.set_title(f'Binary Edema Detection (Window Embeddings) - Epoch {epoch}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = os.path.join(save_dir, f'umap_binary_edema_epoch{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    # ============== Plot 2: Subtype (0 vs 1, edema=1 only) - Window Embeddings ==============
    subtype_mask = ((edema_labels == 1) & ((subtype_labels == 0) | (subtype_labels == 1)))
    if subtype_mask.sum() > 0:
        subtype_emb = window_embeddings[subtype_mask]
        subtype_lbl = subtype_labels[subtype_mask]

        if is_train_mode:
            # Training: fit new PCA + UMAP
            pca_dim = min(50, subtype_emb.shape[0], subtype_emb.shape[1])
            pca_subtype = PCA(n_components=pca_dim, random_state=args.random_seed)
            subtype_emb_pca = pca_subtype.fit_transform(subtype_emb)

            umap_subtype = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=args.random_seed)
            subtype_2d = umap_subtype.fit_transform(subtype_emb_pca)

            fitted_reducers['subtype'] = {'pca': pca_subtype, 'umap': umap_subtype}
            print(f"[Train] Fitted PCA + UMAP for Subtype Classification")
        else:
            # Validation: transform only using pre-fitted PCA + UMAP
            pca_subtype = umap_reducers['subtype']['pca']
            umap_subtype = umap_reducers['subtype']['umap']

            subtype_emb_pca = pca_subtype.transform(subtype_emb)
            subtype_2d = umap_subtype.transform(subtype_emb_pca)
            print(f"[Val] Transformed using Train PCA + UMAP for Subtype Classification")

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = {0: '#42A5F5', 1: '#E53935'}
        labels_map = {0: 'Non-cardiogenic', 1: 'Cardiogenic'}

        for lbl in [0, 1]:
            mask = (subtype_lbl == lbl)
            if mask.sum() > 0:
                ax.scatter(subtype_2d[mask, 0], subtype_2d[mask, 1],
                          c=colors[lbl], label=f'{labels_map[lbl]} (n={mask.sum()})', s=3, alpha=0.6, edgecolors='none')

        ax.legend(fontsize=12)
        ax.set_title(f'Subtype Classification - Window Embeddings (Edema=1 only) - Epoch {epoch}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = os.path.join(save_dir, f'umap_subtype_epoch{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    # ============== Plot 3: Combined 3-class (0, 1, 2) - Window Embeddings ==============
    # Create 3-class labels: No edema=0, Non-cardiogenic=1, Cardiogenic=2
    combined_labels = np.full_like(edema_labels, -1)
    combined_labels[edema_labels == 0] = 0  # No edema
    combined_labels[(edema_labels == 1) & (subtype_labels == 0)] = 1  # Non-cardiogenic
    combined_labels[(edema_labels == 1) & (subtype_labels == 1)] = 2  # Cardiogenic

    combined_valid_mask = (combined_labels != -1)
    if combined_valid_mask.sum() > 0:
        combined_emb = window_embeddings[combined_valid_mask]
        combined_lbl = combined_labels[combined_valid_mask]

        if is_train_mode:
            # Training: fit new PCA + UMAP
            pca_dim = min(50, combined_emb.shape[0], combined_emb.shape[1])
            pca_combined = PCA(n_components=pca_dim, random_state=args.random_seed)
            combined_emb_pca = pca_combined.fit_transform(combined_emb)

            umap_combined = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=args.random_seed)
            combined_2d = umap_combined.fit_transform(combined_emb_pca)

            fitted_reducers['combined'] = {'pca': pca_combined, 'umap': umap_combined}
            print(f"[Train] Fitted PCA + UMAP for Combined 3-Class")
        else:
            # Validation: transform only using pre-fitted PCA + UMAP
            pca_combined = umap_reducers['combined']['pca']
            umap_combined = umap_reducers['combined']['umap']

            combined_emb_pca = pca_combined.transform(combined_emb)
            combined_2d = umap_combined.transform(combined_emb_pca)
            print(f"[Val] Transformed using Train PCA + UMAP for Combined 3-Class")

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = {0: '#9E9E9E', 1: '#42A5F5', 2: '#E53935'}
        labels_map = {0: 'No Edema', 1: 'Non-cardiogenic', 2: 'Cardiogenic'}

        for lbl in [0, 1, 2]:
            mask = (combined_lbl == lbl)
            if mask.sum() > 0:
                ax.scatter(combined_2d[mask, 0], combined_2d[mask, 1],
                          c=colors[lbl], label=f'{labels_map[lbl]} (n={mask.sum()})', s=3, alpha=0.6, edgecolors='none')

        ax.legend(fontsize=12)
        ax.set_title(f'Combined 3-Class (Window Embeddings) - Epoch {epoch}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = os.path.join(save_dir, f'umap_combined_3class_epoch{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    if is_train_mode:
        print("✅ Multi-Task UMAP Visualization Complete")
        return fitted_reducers
    else:
        print("✅ Multi-Task UMAP Visualization Complete")
        return None
