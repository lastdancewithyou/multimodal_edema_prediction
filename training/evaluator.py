import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore', message='Spectral initialisation failed')
from umap import UMAP

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

import torch
import torch.nn.functional as F

from utils import timer
from training.engine import prepare_multiview_inputs_v2


def validate(args, model, dataloader, loss_module, device, accelerator, dataset, epoch=None, disable_cxr=False, disable_txt=False, max_length=256):
    print("=====Running Validation=====")
    model.eval()
    
    ce_sum = torch.zeros(1, device=device, dtype=torch.float32)
    ce_count = torch.zeros(1, device=device, dtype=torch.float32)

    all_probs = []
    all_labels = []

    all_window_embeddings = [] # for umap
    all_window_labels = []     # for umap
    all_window_preds = []      # for umap with predictions
    
    all_ncpe_risk_scores = []
    all_cpe_risk_scores = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="🤖 <Model Inference>"):
            stay_ids = torch.tensor(batch['stay_ids'], dtype=torch.long, device=device)  # [B]

            img_index_tensor = batch['img_index_tensor']
            txt_index_tensor = batch['text_index_tensor']
            has_cxr = (img_index_tensor != -1).long().to(device, non_blocking=True)  # [B, W, T]
            has_text = (txt_index_tensor != -1).long().to(device, non_blocking=True)  # [B, W, T]

            labels = batch['labels'].to(device)
            window_mask = batch['window_mask'].to(device)
            seq_valid_mask = batch['valid_seq_mask'].to(device)

            # Demographic features
            demo_features = batch.get('demo_features')
            if demo_features is not None:
                demo_features = demo_features.to(device, non_blocking=True)

            with accelerator.autocast():
                with timer("Prepare valid multiview inputs", accelerator):
                    ts_series, cxr_data, text_data, has_cxr, has_text = prepare_multiview_inputs_v2(
                        batch, device, has_cxr, has_text, dataset, disable_cxr=disable_cxr, disable_txt=disable_txt,max_length=max_length
                    )

                with timer("Model Inference", accelerator):
                    time_steps = batch.get('time_steps', None)
                    if time_steps is not None:
                        time_steps = time_steps.to(device, non_blocking=True)

                    classification_input, logits = model(
                        args, ts_series, cxr_data, text_data, has_cxr, has_text, window_mask, seq_valid_mask, demo_features, time_steps=time_steps
                    )

                with timer("Loss  computation and post-processing", accelerator):
                    inference_results = loss_module.inference(classification_input, logits, labels, window_mask)

            valid_logits = inference_results['logits'] # [B, W, C]
            valid_labels = inference_results['labels'] # [B, W]
            valid_mask = inference_results['window_mask'] # [B, W]
            window_embeddings = inference_results['window_embeddings'] # [B, W, D]

            mask = valid_mask.bool()

            masked_logits = valid_logits[mask]
            masked_labels = valid_labels[mask]
            masked_embeddings = window_embeddings[mask]

            label_mask = masked_labels != -1
            ce_logits = masked_logits[label_mask]
            ce_labels = masked_labels[label_mask]

            # 라벨이 1개 이상 있을 때만 손실 계산
            if ce_labels.numel() == 0:
                continue

            batch_ce_loss = loss_module.cross_entropy(ce_logits, ce_labels)
            
            ce_ct_local = torch.as_tensor(ce_labels.numel(), device=device, dtype=torch.float32)
            ce_sum += torch.as_tensor(batch_ce_loss.item(), device=device, dtype=torch.float32) * ce_ct_local
            ce_count += ce_ct_local

            probs = F.softmax(masked_logits, dim=-1)
            ncpe_risk_scores = probs[:, 1]
            cpe_risk_scores = probs[:, 2]

            labeled_mask = masked_labels != -1
            if labeled_mask.any():
                labeled_embeddings = masked_embeddings[labeled_mask]
                labeled_labels = masked_labels[labeled_mask]
                labeled_probs = probs[labeled_mask]
                labeled_predictions = torch.argmax(labeled_probs, dim=-1)

                all_probs.append(labeled_probs.detach())
                all_labels.append(labeled_labels.detach())
                
                # (optional) UMAP
                all_window_embeddings.append(labeled_embeddings.detach())
                all_window_labels.append(labeled_labels.detach())
                all_window_preds.append(labeled_predictions.detach())

            all_ncpe_risk_scores.append(ncpe_risk_scores.detach().cpu())
            all_cpe_risk_scores.append(cpe_risk_scores.detach().cpu())

    if accelerator.num_processes > 1:
        total_ce_sum = accelerator.gather_for_metrics(ce_sum).sum()
        total_ce_count = accelerator.gather_for_metrics(ce_count).sum()
    else:
        total_ce_sum = ce_sum
        total_ce_count = ce_count

    with timer("Metric Computation"):
        avg_loss = (total_ce_sum / total_ce_count).item()

        if len(all_probs) > 0:
            local_probs = torch.cat(all_probs, dim=0)   # [N_local, num_classes]
            local_labels = torch.cat(all_labels, dim=0) # [N_local]
            
            local_size = torch.tensor(local_probs.size(0), device=device)
            all_sizes = accelerator.gather_for_metrics(local_size)  # [world_size]
            max_size = all_sizes.max().item()
            
            # 수동 패딩 (최대 크기로 맞춤)
            current_size = local_probs.size(0)
            if current_size < max_size:
                pad_size = max_size - current_size

                # probs 패딩 (0으로 채움)
                pad_probs = torch.zeros(pad_size, local_probs.size(1), device=local_probs.device)
                local_probs = torch.cat([local_probs, pad_probs], dim=0)

                # labels 패딩 (-1로 채움)
                pad_labels = torch.full((pad_size,), -1, device=local_labels.device, dtype=local_labels.dtype)
                local_labels = torch.cat([local_labels, pad_labels], dim=0)
            
            # gather_for_metrics (이제 크기 동일)
            gathered_probs = accelerator.gather_for_metrics(local_probs)
            gathered_labels = accelerator.gather_for_metrics(local_labels)
            
            # 패딩 제거: 실제 데이터만 추출
            total_valid = all_sizes.sum().item()
            all_probs = gathered_probs[:total_valid].cpu().numpy()
            all_labels = gathered_labels[:total_valid].cpu().numpy()

        else:
            all_probs = np.array([]).reshape(0, 3)
            all_labels = np.array([])

        # ---------------- Precision/Recall (macro + weighted) ----------------
        metrics = {}

        if len(all_labels) > 0:
            unique_labels = np.unique(all_labels)
            num_classes = all_probs.shape[1]
            preds = all_probs.argmax(axis=-1)

            for avg_type in ['macro', 'weighted']:
                try: 
                    precision, recall, _, _ = precision_recall_fscore_support(
                        all_labels,
                        preds,
                        labels=list(range(num_classes)),   # 누락 클래스도 포함
                        average=avg_type,                  # macro & weighted 둘 다 계산
                        zero_division=0                    # 0/0 방지
                    )

                except ValueError as e:
                    print(f"[Warning] {avg_type} precision/recall computation failed: {e}")
                    precision, recall = float('nan'), float('nan')

                metrics[f"precision_{avg_type}"] = precision
                metrics[f"recall_{avg_type}"] = recall

            # ---------------- Per-class accuracy ----------------
            print("\n[Summary]")
            print("\n[Accuracy by classes]")
            per_class_metrics = {}

            for label in range(num_classes):
                total = (all_labels == label).sum()
                correct = ((all_labels == label) & (preds == label)).sum()

                if total > 0:
                    acc = 100.0 * correct / total
                    per_class_metrics[f'class_{label}_accuracy'] = acc
                    per_class_metrics[f'class_{label}_count'] = total
                    print(f"Label {label}: {correct}/{total} = {acc:.1f}%")
                else: 
                    per_class_metrics[f'class_{label}_accuracy'] = None
                    per_class_metrics[f'class_{label}_count'] = 0
                    print(f"Label {label}: No samples.")

            # ---------------- AUROC / AUPRC ----------------
            if len(unique_labels) >= 2:
                try: 
                    all_labels_binarized = label_binarize(all_labels, classes=list(range(num_classes)))
                    
                    valid_classes = [
                        i for i in range(num_classes)
                        if 0 < np.sum(all_labels_binarized[:, i]) < len(all_labels)
                    ]

                    if len(valid_classes) >= 2:
                        for avg_type in ["macro", "weighted"]:
                            try:
                                auroc = roc_auc_score(
                                    all_labels_binarized,
                                    all_probs,
                                    average=avg_type,
                                    multi_class='ovr'
                                )

                                auprc = average_precision_score(
                                    all_labels_binarized,
                                    all_probs,
                                    average=avg_type
                                )

                            except Exception as e: 
                                print("[Warning] AUROC/AUPRC computation failed")
                                auroc, auprc = float('nan'), float('nan')

                            metrics[f"auroc_{avg_type}"] = auroc
                            metrics[f"auprc_{avg_type}"] = auprc
        
                    else: 
                        print("[Warning] Only one class present in labels. AUROC/AUPRC not defined.")
                        auroc = float('nan')
                        auprc = float('nan')
    
                except Exception:
                    print("[Warning] AUROC/AUPRC computation failed unexpectedly.")

            if accelerator.is_main_process:
                print(
                    f"Loss: {avg_loss:.4f}\n"
                    f"  ├─ Macro     → Precision: {metrics.get('precision_macro', float('nan')):.4f}, "
                    f"Recall: {metrics.get('recall_macro', float('nan')):.4f}, "
                    f"AUROC: {metrics.get('auroc_macro', float('nan')):.4f}, "
                    f"AUPRC: {metrics.get('auprc_macro', float('nan')):.4f}\n"
                    f"  └─ Weighted  → Precision: {metrics.get('precision_weighted', float('nan')):.4f}, "
                    f"Recall: {metrics.get('recall_weighted', float('nan')):.4f}, "
                    f"AUROC: {metrics.get('auroc_weighted', float('nan')):.4f}, "
                    f"AUPRC: {metrics.get('auprc_weighted', float('nan')):.4f}"
                )

        metrics = {
            **metrics,
            **per_class_metrics
        }

    return avg_loss, ce_count.item(), metrics, all_window_embeddings, all_window_labels, all_window_preds, _


def validate_stage1(args, model, dataloader, loss_module, device, accelerator, dataset, epoch=None, disable_cxr=False, disable_txt=False, max_length=256):
    """
    - Stage 1 전용 Validation 함수
    - CE Loss는 계산하지 않음
    """
    print("=====Running Stage 1 Validation=====")
    model.eval()

    ts_recon_sum = torch.zeros(1, device=device, dtype=torch.float32)
    ts_recon_count = torch.zeros(1, device=device, dtype=torch.float32)
    local_temp_sum = torch.zeros(1, device=device, dtype=torch.float32)
    local_temp_count = torch.zeros(1, device=device, dtype=torch.float32)
    scl_sum = torch.zeros(1, device=device, dtype=torch.float32)
    scl_count = torch.zeros(1, device=device, dtype=torch.float32)
    time_aware_sum = torch.zeros(1, device=device, dtype=torch.float32)
    time_aware_count = torch.zeros(1, device=device, dtype=torch.float32)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="🤖 <Stage1 Validation>"):
            _, _, batch_scl, batch_ts_recon, batch_local_temp, batch_time_aware, _, batch_counts = train_batch(
                args=args,
                model=model,
                batch=batch,
                loss_module=loss_module,
                device=device,
                accelerator=accelerator,
                dataset=dataset,
                max_length=max_length,
                disable_cxr=disable_cxr,
                disable_txt=disable_txt,
                ce_weight=0.0,  # CE 사용 안 함
            )

            window_ct_local = torch.as_tensor(batch_counts['window_count'], device=device, dtype=torch.float32)
            ts_recon_sum += torch.as_tensor(batch_ts_recon, device=device, dtype=torch.float32) * window_ct_local
            ts_recon_count += window_ct_local
            local_temp_sum += torch.as_tensor(batch_local_temp, device=device, dtype=torch.float32) * window_ct_local
            local_temp_count += window_ct_local
            scl_sum += torch.as_tensor(batch_scl, device=device, dtype=torch.float32) * window_ct_local
            scl_count += window_ct_local
            time_aware_sum += torch.as_tensor(batch_time_aware, device=device, dtype=torch.float32) * window_ct_local
            time_aware_count += window_ct_local

    # GPU 간 집계
    if accelerator.num_processes > 1:
        total_ts_recon_sum = accelerator.gather_for_metrics(ts_recon_sum).sum()
        total_ts_recon_count = accelerator.gather_for_metrics(ts_recon_count).sum()
        total_local_temp_sum = accelerator.gather_for_metrics(local_temp_sum).sum()
        total_local_temp_count = accelerator.gather_for_metrics(local_temp_count).sum()
        total_scl_sum = accelerator.gather_for_metrics(scl_sum).sum()
        total_scl_count = accelerator.gather_for_metrics(scl_count).sum()
        total_time_aware_sum = accelerator.gather_for_metrics(time_aware_sum).sum()
        total_time_aware_count = accelerator.gather_for_metrics(time_aware_count).sum()
    else:
        total_ts_recon_sum = ts_recon_sum
        total_ts_recon_count = ts_recon_count
        total_local_temp_sum = local_temp_sum
        total_local_temp_count = local_temp_count
        total_scl_sum = scl_sum
        total_scl_count = scl_count
        total_time_aware_sum = time_aware_sum
        total_time_aware_count = time_aware_count

    ts_recon_avg = (total_ts_recon_sum / (total_ts_recon_count + 1e-8)).item()
    local_temp_avg = (total_local_temp_sum / (total_local_temp_count + 1e-8)).item()
    scl_avg = (total_scl_sum / (total_scl_count + 1e-8)).item()
    time_aware_avg = (total_time_aware_sum / (total_time_aware_count + 1e-8)).item()

    ts_recon_contrib = args.ts_recon_weight * ts_recon_avg
    local_temp_contrib = args.local_temp_weight * local_temp_avg
    scl_contrib = args.scl_weight * scl_avg
    time_aware_weight = args.time_aware_weight
    time_aware_contrib = time_aware_weight * time_aware_avg
    total_stage1_loss = ts_recon_contrib + local_temp_contrib + scl_contrib + time_aware_contrib

    if accelerator.is_main_process:
        print("\n[Stage 1 Validation Summary]")
        print(
            f"Total Loss: {total_stage1_loss:.4f}\n"
            f"  [Raw] TS Recon: {ts_recon_avg:.4f} | Local Temporal: {local_temp_avg:.4f} | SupCon: {scl_avg:.4f} | Time-Aware: {time_aware_avg:.4f}\n"
            f"  [Weighted] TS Recon: {ts_recon_contrib:.4f} (λ={args.ts_recon_weight}) | "
            f"Local Temp: {local_temp_contrib:.4f} (λ={args.local_temp_weight}) | "
            f"SCL: {scl_contrib:.4f} (λ={args.scl_weight}) | "
            f"Time-Aware: {time_aware_contrib:.4f} (λ={time_aware_weight})"
        )

    return total_stage1_loss, ts_recon_avg, local_temp_avg, scl_avg, time_aware_avg


# Test 함수
def test(args, model, dataloader, loss_module, device, accelerator, dataset):
    test_loss, _, test_metrics, test_window_embeddings, test_window_labels, test_window_preds, _ = validate(
        args, model, dataloader, loss_module, device, accelerator, dataset, epoch="final"
    )

    if accelerator.is_main_process:
        wandb_test_metrics = {
            'test/loss': test_loss,
            'test/precision': test_metrics.get('precision_macro'),
            'test/recall': test_metrics.get('recall_macro'),
            'test/auroc': test_metrics.get('auroc_macro'),
            'test/auprc': test_metrics.get('auprc_macro')
        }

        for label in range(args.num_classes):
            acc_key = f'class_{label}_accuracy'
            count_key = f'class_{label}_count'
            if acc_key in test_metrics and test_metrics[acc_key] is not None:
                wandb_test_metrics[f'test/{acc_key}'] = test_metrics[acc_key]
                wandb_test_metrics[f'test/{count_key}'] = test_metrics[count_key]

    return test_loss, test_metrics, test_window_embeddings, test_window_labels, test_window_preds, _


# UMAP 생성 함수
def plot_umap_2d(args, window_embeddings_list, window_labels_list, window_pred_list=None, epoch=None, prefix=None, pca_model=None):
    window_embeddings = torch.cat(window_embeddings_list, dim=0)
    window_labels = torch.cat(window_labels_list, dim=0)
    
    window_pred = None
    if window_pred_list is not None:
        window_pred = torch.cat(window_pred_list, dim=0)
        window_pred = window_pred.cpu().numpy()

    window_embeddings = window_embeddings.cpu().numpy()
    window_labels = window_labels.cpu().numpy()

    # 첫 번째 에포크에서는 PCA 모델 학습
    if pca_model is None:
        pca_model = PCA(n_components=args.pca_components, random_state=args.random_seed)
        pca_model.fit(window_embeddings)

    # 이후 에포크에서는 좌표계 고정함.
    window_embeddings = pca_model.transform(window_embeddings)

    umap_model = UMAP(
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric
    )

    embeddings_2d = umap_model.fit_transform(window_embeddings)

    umap_df = pd.DataFrame(embeddings_2d, columns=['component 0', 'component 1'])
    umap_df['label'] = window_labels
    if window_pred is not None:
        umap_df['pred'] = window_pred

    label_colors = {
        0: 'lightcoral',
        1: 'turquoise', 
        2: 'blueviolet',
        -1: 'black'  # Unlabeled
    }

    label_mapping = {
        0: 'Negative',
        1: 'Noncardiogenic Edema',
        2: 'Cardiogenic Edema',
        -1: 'Unlabeled'
    }

    # visualization
    fig = plt.figure(figsize=(12,12))

    for label, color in label_colors.items():
        # label이 -1인 경우는 별도로 처리
        if label == -1:
            continue
        subset = umap_df[umap_df['label'] == label]
        plt.scatter(
            subset['component 0'],
            subset['component 1'],
            color=color,
            label=label_mapping[label],
            s=8,
            alpha=0.4
        )

    unlabeled_subset = umap_df[umap_df['label'] == -1]
    if len(unlabeled_subset) > 0 and window_pred is not None:
        for pred_label, color in label_colors.items():
            if pred_label == -1:  # Skip unlabeled predictions
                continue
            pred_subset = unlabeled_subset[unlabeled_subset['pred'] == pred_label]
            if len(pred_subset) > 0:  # Only plot if there are samples
                plt.scatter(
                    pred_subset['component 0'],
                    pred_subset['component 1'],
                    facecolors='black',
                    edgecolors=color,
                    linewidths=0.5,
                    s=8,
                    alpha=0.4,
                    label=f"Unlabeled (pred: {label_mapping[pred_label]})"
                )

    plt.legend(title="Classes", fontsize=12, handlelength=3, loc='upper left')
    plt.title(f"{prefix.title()} Window-level Label distribution@ Epoch {epoch}" if epoch is not None else f"{prefix.title()} Window-level Label distribution")
    plt.xticks([])
    plt.yticks([])

    # ===== Save Locally =====
    save_dir = args.umap_save_dir
    os.makedirs(save_dir, exist_ok=True)

    file_name = f"{prefix}_umap_epoch{epoch}.png" if epoch is not None else f"{prefix}_umap.png"
    save_path = os.path.join(save_dir, file_name)

    plt.savefig(save_path)
    plt.close(fig)
    return pca_model