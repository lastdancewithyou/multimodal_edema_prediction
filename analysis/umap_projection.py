import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from umap import UMAP

import torch

from training.engine import prepare_multiview_inputs


@torch.no_grad()
def plot_projection_umap(args, model, dataloader, save_dir, epoch, split_name="train", max_samples=10000, disable_cxr=False, disable_txt=False):
    """
    Contrastive projection head 출력(proj_emb, [B, 128])에 대한 UMAP 시각화.
    edema_soft 라벨로 색칠 (NaN은 회색).
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    device = next(base_model.parameters()).device

    all_proj = []
    all_soft = []
    collected = 0

    for batch in tqdm(dataloader, desc=f"[UMAP/{split_name}] Collecting proj_emb"):
        img_index_tensor = batch['img_index_tensor']
        txt_index_tensor = batch['text_index_tensor']
        has_cxr = (img_index_tensor != -1).long()
        has_text = (txt_index_tensor != -1).long()

        ts_series, cxr_data, text_data, has_cxr, has_text = prepare_multiview_inputs(
            batch, has_cxr, has_text,
            disable_cxr=disable_cxr, disable_txt=disable_txt,
        )

        ts_series = ts_series.to(device)
        has_cxr = has_cxr.to(device)
        has_text = has_text.to(device)
        time_steps = batch['time_steps'].to(device)
        for k in ['unique_embs', 'unique_segment_embs', 'segment_index_tensor', 'unique_indices', 'positions']:
            if k in cxr_data and isinstance(cxr_data[k], torch.Tensor):
                cxr_data[k] = cxr_data[k].to(device)
        for k in ['unique_embs', 'unique_indices', 'positions']:
            if k in text_data and isinstance(text_data[k], torch.Tensor):
                text_data[k] = text_data[k].to(device)

        outputs = base_model(args, ts_series, cxr_data, text_data, has_cxr, has_text, time_steps=time_steps)
        proj_emb = outputs['proj_emb'].float().cpu().numpy()       # [B, 128]
        soft = batch['edema_soft_labels'].float().cpu().numpy()    # [B], NaN 허용

        all_proj.append(proj_emb)
        all_soft.append(soft)
        collected += proj_emb.shape[0]
        if max_samples is not None and collected >= max_samples:
            break

    if not all_proj:
        print(f"[UMAP/{split_name}] No embeddings collected — skipping.")
        return

    proj = np.concatenate(all_proj, axis=0)
    soft = np.concatenate(all_soft, axis=0)
    if max_samples is not None and len(proj) > max_samples:
        idx = np.random.RandomState(0).choice(len(proj), max_samples, replace=False)
        proj, soft = proj[idx], soft[idx]

    # UMAP
    reducer = UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=42,
    )
    emb2d = reducer.fit_transform(proj)

    valid = ~np.isnan(soft)
    fig, ax = plt.subplots(figsize=(7, 6))
    if (~valid).any():
        ax.scatter(emb2d[~valid, 0], emb2d[~valid, 1],
                c='lightgrey', s=4, alpha=0.4, label=f'unlabeled (n={int((~valid).sum())})')
    if valid.any():
        sc = ax.scatter(emb2d[valid, 0], emb2d[valid, 1],
                        c=soft[valid], cmap='viridis', s=6, alpha=0.85,
                        vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('edema_soft', rotation=270, labelpad=12)

    ax.set_title(f"Projection UMAP ({split_name}) — epoch {epoch}\nn={len(proj)}, labeled={int(valid.sum())}")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    if (~valid).any():
        ax.legend(loc='best', fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"proj_umap_{split_name}_epoch{epoch}.png")
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[UMAP/{split_name}] Saved: {out_path}")
