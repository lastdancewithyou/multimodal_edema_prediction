import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA

from training.run import parse_arguments
from training.engine import prepare_multiview_inputs_v2


def compute_target_alignment(loss_module, device='cuda'):
    target_supcon = loss_module.target_supcon_loss_fn
    target_proto = target_supcon.target_proto  # [num_classes, D]
    class_centroid = target_supcon.class_centroid  # [num_classes, D]

    # Compute cosine similarity per class
    alignment = F.cosine_similarity(class_centroid, target_proto, dim=1)  # [num_classes]

    results = {
        'per_class_alignment': alignment.cpu().numpy(),
        'mean_alignment': alignment.mean().item()
    }

    return results


def print_target_alignment(alignment_results, epoch=None):
    """
    Pretty print target alignment results.
    """
    if alignment_results is None:
        return

    header = f"\n{'='*60}\n"
    if epoch is not None:
        header += f"🎯 Target Alignment (Epoch {epoch})\n"
    else:
        header += f"🎯 Target Alignment\n"
    header += f"{'='*60}"
    print(header)

    per_class = alignment_results['per_class_alignment']
    mean_alignment = alignment_results['mean_alignment']

    print(f"Per-Class Alignment (Cosine Similarity):")
    for c, align in enumerate(per_class):
        # Emoji indicator
        if align > 0.9:
            indicator = "✅"
        elif align > 0.7:
            indicator = "⚠️"
        else:
            indicator = "❌"

        print(f"  {indicator} Class {c}: {align:.4f}")

    print(f"\n📊 Mean Alignment: {mean_alignment:.4f}")

    # Interpretation
    if mean_alignment > 0.9:
        print("   → Excellent! Centroids are well-aligned with targets.")
    elif mean_alignment > 0.7:
        print("   → Good. Centroids are converging to targets.")
    elif mean_alignment > 0.5:
        print("   → Fair. Still learning...")
    else:
        print("   → Poor. Need more training or check hyperparameters.")

    print("="*60 + "\n")


def visualize_target_supcon(
        model,
        dataloader,
        loss_module,
        device,
        save_path='target_vis.png',
        max_samples=5000,
        epoch=None,
        use_target=True,  # False면 KCL-only 구간용 (임베딩만 시각화)
    ):

    mode_str = "Target_SupCon" if use_target else "KCL-only"
    print(f"\n{'='*60}")
    print(f"🎨 Generating {mode_str} UMAP Visualization")
    print(f"{'='*60}")

    target_supcon = loss_module.target_supcon_loss_fn
    kcl = target_supcon.kcl
    num_classes = kcl.n_cls

    embeddings_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(embeddings_list) > 0 and sum(len(x) for x in embeddings_list) >= max_samples:
                break

            for k in ['labels', 'window_mask', 'valid_seq_mask']:
                batch[k] = batch[k].to(device, non_blocking=True)

            demo_features = batch.get('demo_features')
            if demo_features is not None:
                demo_features = demo_features.to(device, non_blocking=True)

            img_index_tensor = batch['img_index_tensor']
            txt_index_tensor = batch['text_index_tensor']
            has_cxr = (img_index_tensor != -1).long().to(device, non_blocking=True)
            has_text = (txt_index_tensor != -1).long().to(device, non_blocking=True)

            labels = batch['labels']
            window_mask = batch['window_mask']
            seq_valid_mask = batch['valid_seq_mask']

            ts = batch['ts_tensor'].to(device, non_blocking=True)

            
            args = parse_arguments()

            ts_series, cxr_views, text_series, has_cxr, has_text = prepare_multiview_inputs_v2(
                batch, device, has_cxr, has_text,
                dataset=dataloader.dataset,
                disable_cxr=args.disable_cxr,
                disable_txt=args.disable_txt,
                max_length=256,
            )

            # Stage 1 model returns: projected_embeddings_multiview [Nwin, 2, proj_dim]
            projected_multiview = model(
                args, ts_series, cxr_views, text_series, has_cxr, has_text,
                window_mask, seq_valid_mask, demo_features
            )  # [Nwin, 2, proj_dim]

            # Use first view for visualization (both views should be similar after projection)
            z_valid = projected_multiview[:, 0, :]  # [Nwin, proj_dim]

            # Reconstruct to [B, W, D] format for compatibility
            Nwin = z_valid.shape[0]
            B, W = labels.shape
            BW = B * W
            D = z_valid.shape[1]

            # Create full tensor and fill valid windows
            z = torch.zeros(BW, D, device=device, dtype=z_valid.dtype)
            window_mask_flat = window_mask.reshape(BW).bool()
            z[window_mask_flat] = z_valid
            z = z.view(B, W, D)  # [B, W, D]

            # Flatten and filter valid windows
            B, W, D = z.shape
            z_flat = z.reshape(B * W, D)
            labels_flat = labels.reshape(B * W)
            mask_flat = window_mask.reshape(B * W).bool()

            z_valid = z_flat[mask_flat]
            labels_valid = labels_flat[mask_flat]

            embeddings_list.append(z_valid.cpu().numpy())
            labels_list.append(labels_valid.cpu().numpy())

    if len(embeddings_list) == 0:
        print("[Error] No embeddings collected. Cannot visualize.")
        return None

    embeddings = np.concatenate(embeddings_list, axis=0)[:max_samples]  # [N, D]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]  # [N]
    print(f"   Collected {len(embeddings)} samples for visualization")

    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # UMAP projection
    print(f"   Running UMAP projection...")

    if use_target:
        target = kcl.optimal_target_unique.t().cpu().numpy()  # [num_classes, D]
        target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
        all_data = np.vstack([embeddings_norm, target_norm])
    else:
        all_data = embeddings_norm

    if all_data.shape[1] > 50:
        pca = PCA(n_components=50)
        all_data_pca = pca.fit_transform(all_data)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        all_projected = reducer.fit_transform(all_data_pca)
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        all_projected = reducer.fit_transform(all_data)

    n_embeddings = len(embeddings)
    emb_proj = all_projected[:n_embeddings]
    if use_target:
        target_proj = all_projected[n_embeddings:]

    print(f"   UMAP projection complete!")

    # Distance to target statistics (TSC 구간만)
    if use_target:
        print(f"\n   📊 Distance to Target Statistics:")
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_embeds = embeddings_norm[mask]
                target_c = target_norm[c:c+1]
                similarities = np.dot(class_embeds, target_c.T).squeeze()
                distances = 1.0 - similarities
                print(f"      Class {c}: mean={distances.mean():.4f}, median={np.median(distances):.4f}, std={distances.std():.4f} (n={mask.sum()})")

    # Intra-class compactness (KCL/TSC 공통)
    print(f"\n   📊 Intra-class Compactness (UMAP space):")
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 1:
            class_proj = emb_proj[mask]
            centroid = class_proj.mean(axis=0)
            spread = np.linalg.norm(class_proj - centroid, axis=1).mean()
            print(f"      Class {c}: mean spread={spread:.4f} (n={mask.sum()})  (작을수록 뭉침)")

    # Plot
    print(f"   Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            ax.scatter(emb_proj[mask, 0], emb_proj[mask, 1],
                    alpha=0.4, s=15, c=colors[c % len(colors)],
                    label=f'Class {c} ({mask.sum()} samples)')

    if use_target:
        for c in range(num_classes):
            ax.scatter(target_proj[c, 0], target_proj[c, 1],
                    marker='*', s=800, c=colors[c % len(colors)],
                    edgecolors='black', linewidths=3, zorder=10)
            ax.text(target_proj[c, 0], target_proj[c, 1] + 0.3,
                    f'Target {c}', fontsize=12, fontweight='bold',
                    ha='center', va='bottom')

    title = f'{mode_str}: Embedding Space Visualization'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    ax.set_title(title, fontsize=16, fontweight='bold')

    from matplotlib.lines import Line2D
    if use_target:
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                markersize=15, label='Target (Fixed)', markeredgecolor='black', markeredgewidth=1.5)
        ]
        ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0],
                loc='upper right', fontsize=10, framealpha=0.9)
    else:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Visualization saved to: {save_path}")
    print("="*60 + "\n")

    return fig
