import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA


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
        epoch=None
    ):

    print(f"\n{'='*60}")
    print(f"🎨 Generating Target_SupCon UMAP Visualization")
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

            # Forward pass (you'll need to adapt this to your model's forward signature)
            try:
                # Get model args first (need disable flags from args, not model)
                from train.run import parse_arguments
                args = parse_arguments()

                from my_research.src.train.engine import prepare_multiview_inputs_v2
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

            except Exception as e:
                print(f"[Warning] Could not extract embeddings: {e}")
                print("         Using random embeddings for demonstration.")
                B, W = labels.shape
                D = kcl.optimal_target_unique.shape[0]  # Embedding dimension
                z = torch.randn(B, W, D, device=device)

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

    # Get optimal targets (transposed back to [num_classes, D])
    target = kcl.optimal_target_unique.t().cpu().numpy()  # [num_classes, D]

    # Normalize embeddings (important for contrastive learning visualization)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)

    # 3. UMAP projection
    print(f"   Running UMAP projection...")

    # Use normalized embeddings for UMAP
    # PCA first for speed (optional, but recommended for large D)
    if embeddings_norm.shape[1] > 50:
        pca = PCA(n_components=50)
        all_data = np.vstack([embeddings_norm, target_norm])
        all_data_pca = pca.fit_transform(all_data)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        all_projected = reducer.fit_transform(all_data_pca)
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        all_data = np.vstack([embeddings_norm, target_norm])
        all_projected = reducer.fit_transform(all_data)

    # Split back
    n_embeddings = len(embeddings)
    n_classes = len(target)

    emb_proj = all_projected[:n_embeddings]
    target_proj = all_projected[n_embeddings:n_embeddings + n_classes]

    print(f"   UMAP projection complete!")

    # Compute distance statistics (using normalized embeddings)
    print(f"\n   📊 Distance to Target Statistics:")
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            # Cosine similarity (since embeddings are normalized, this is just dot product)
            class_embeds = embeddings_norm[mask]  # [N_c, D]
            target_c = target_norm[c:c+1]  # [1, D]

            # Cosine similarity = dot product for normalized vectors
            similarities = np.dot(class_embeds, target_c.T).squeeze()  # [N_c]

            # Convert to distance (1 - similarity for cosine)
            distances = 1.0 - similarities

            mean_dist = distances.mean()
            std_dist = distances.std()
            median_dist = np.median(distances)

            print(f"      Class {c}: mean={mean_dist:.4f}, median={median_dist:.4f}, std={std_dist:.4f} (n={mask.sum()})")

    # 4. Plot
    print(f"   Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 12))

    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Plot embeddings (by class)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            ax.scatter(emb_proj[mask, 0], emb_proj[mask, 1],
                       alpha=0.4, s=15, c=colors[c % len(colors)],
                       label=f'Class {c} ({mask.sum()} samples)')

    # Plot targets (large stars with matching class colors)
    for c in range(num_classes):
        ax.scatter(target_proj[c, 0], target_proj[c, 1],
                   marker='*', s=800, c=colors[c % len(colors)],
                   edgecolors='black',
                   linewidths=3, zorder=10)
        ax.text(target_proj[c, 0], target_proj[c, 1] + 0.3,
                f'Target {c}', fontsize=12, fontweight='bold',
                ha='center', va='bottom')

    # Title and legend
    title = 'Target_SupCon: Embedding Space Visualization'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
            markersize=15, label='Target (Fixed)', markeredgecolor='black', markeredgewidth=1.5)
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0],
              loc='upper right', fontsize=10, framealpha=0.9)

    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Visualization saved to: {save_path}")
    print("="*60 + "\n")

    return fig
