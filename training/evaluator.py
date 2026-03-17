import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore', message='Spectral initialisation failed')
from umap import UMAP

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

import torch
import torch.nn.functional as F

from utils import timer
from training.engine import prepare_multiview_inputs_v2, train_batch
from analysis.calibration import ExpectedCalibrationError, analyze_calibration


def validate_multitask(args, model, dataloader, loss_module, device, accelerator, dataset, epoch=None, disable_cxr=False, disable_txt=False, max_length=256):
    print("=====Running Multi-Task Validation=====")
    model.eval()

    bce_sum = torch.zeros(1, device=device, dtype=torch.float32)
    bce_count = torch.zeros(1, device=device, dtype=torch.float32)
    ce_sum = torch.zeros(1, device=device, dtype=torch.float32)
    ce_count = torch.zeros(1, device=device, dtype=torch.float32)
    ucl_sum = torch.zeros(1, device=device, dtype=torch.float32)
    ucl_count = torch.zeros(1, device=device, dtype=torch.float32)
    scl_sum = torch.zeros(1, device=device, dtype=torch.float32)
    scl_count = torch.zeros(1, device=device, dtype=torch.float32)
    info_ucl_sum = torch.zeros(1, device=device, dtype=torch.float32)
    info_ucl_count = torch.zeros(1, device=device, dtype=torch.float32)

    val_edema_preds_list = []
    val_edema_labels_list = []
    val_subtype_preds_list = []
    val_subtype_labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="🤖 <Multi-Task Validation>"):
            _, batch_bce, batch_ce, batch_ucl, batch_scl, batch_info_ucl, batch_outputs, batch_counts = train_batch(
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
                bce_weight=args.bce_weight,
                ce_weight=args.ce_weight,
                ucl_weight=args.ucl_weight,
                scl_weight=args.scl_weight,
                infonce_weight=args.infonce_weight,
            )

            bce_ct_local = torch.as_tensor(batch_counts['bce_count'], device=device, dtype=torch.float32)
            ce_ct_local = torch.as_tensor(batch_counts['ce_count'], device=device, dtype=torch.float32)
            ucl_ct_local = torch.as_tensor(batch_counts['ucl_count'], device=device, dtype=torch.float32)
            scl_ct_local = torch.as_tensor(batch_counts['scl_count'], device=device, dtype=torch.float32)
            infonce_ct_local = torch.as_tensor(batch_counts['infonce_count'], device=device, dtype=torch.float32)

            bce_sum += torch.as_tensor(batch_bce, device=device, dtype=torch.float32) * bce_ct_local
            bce_count += bce_ct_local
            ce_sum += torch.as_tensor(batch_ce, device=device, dtype=torch.float32) * ce_ct_local
            ce_count += ce_ct_local
            ucl_sum += torch.as_tensor(batch_ucl, device=device, dtype=torch.float32) * ucl_ct_local
            ucl_count += ucl_ct_local
            scl_sum += torch.as_tensor(batch_scl, device=device, dtype=torch.float32) * scl_ct_local
            scl_count += scl_ct_local
            info_ucl_sum += torch.as_tensor(batch_info_ucl, device=device, dtype=torch.float32) * infonce_ct_local
            info_ucl_count += infonce_ct_local
            
            edema_logits = batch_outputs['edema_logits'].squeeze(-1)  # [B, W]
            subtype_logits = batch_outputs['subtype_logits']           # [B, W, 2]
            edema_labels = batch_outputs['edema_labels']               # [B, W]
            subtype_labels = batch_outputs['subtype_labels']           # [B, W]
            window_mask = batch['window_mask']                         # [B, W]

            # Valid windows only (padding excluded)
            valid_mask = window_mask.bool()  # [B, W]

            p_pos = torch.sigmoid(edema_logits)                        # [B, W] P(edema=1)
            p_sub = torch.softmax(subtype_logits, dim=-1)              # [B, W, 2] P(NCPE|pos), P(CPE|pos)

            p_pos_valid = p_pos[valid_mask]                            # [Nwin]
            p_sub_valid = p_sub[valid_mask]                            # [Nwin, 2]
            edema_valid = edema_labels[valid_mask]                     # [Nwin]
            subtype_valid = subtype_labels[valid_mask]                 # [Nwin]

            val_edema_preds_list.append(p_pos_valid.detach().cpu())
            val_subtype_preds_list.append(p_sub_valid.detach().cpu())
            val_edema_labels_list.append(edema_valid.detach().cpu())
            val_subtype_labels_list.append(subtype_valid.detach().cpu())

    # GPU aggregation
    if accelerator.num_processes > 1:
        total_bce_sum = accelerator.gather_for_metrics(bce_sum).sum()
        total_bce_count = accelerator.gather_for_metrics(bce_count).sum()
        total_ce_sum = accelerator.gather_for_metrics(ce_sum).sum()
        total_ce_count = accelerator.gather_for_metrics(ce_count).sum()
        total_ucl_sum = accelerator.gather_for_metrics(ucl_sum).sum()
        total_ucl_count = accelerator.gather_for_metrics(ucl_count).sum()
        total_scl_sum = accelerator.gather_for_metrics(scl_sum).sum()
        total_scl_count = accelerator.gather_for_metrics(scl_count).sum()
        total_info_ucl_sum = accelerator.gather_for_metrics(info_ucl_sum).sum()
        total_info_ucl_count = accelerator.gather_for_metrics(info_ucl_count).sum()
    else:
        total_bce_sum = bce_sum
        total_bce_count = bce_count
        total_ce_sum = ce_sum
        total_ce_count = ce_count
        total_ucl_sum = ucl_sum
        total_ucl_count = ucl_count
        total_scl_sum = scl_sum
        total_scl_count = scl_count
        total_info_ucl_sum = info_ucl_sum
        total_info_ucl_count = info_ucl_count

    bce_avg = (total_bce_sum / (total_bce_count + 1e-8)).item()
    ce_avg = (total_ce_sum / (total_ce_count + 1e-8)).item()
    ucl_avg = (total_ucl_sum / (total_ucl_count + 1e-8)).item()
    scl_avg = (total_scl_sum / (total_scl_count + 1e-8)).item()
    info_ucl_avg = (total_info_ucl_sum / (total_info_ucl_count + 1e-8)).item()

    bce_contrib = args.bce_weight * bce_avg 
    ce_contrib = args.ce_weight * ce_avg
    ucl_contrib = args.ucl_weight * ucl_avg
    scl_contrib = args.scl_weight * scl_avg
    info_ucl_contrib = args.infonce_weight * info_ucl_avg
    total_loss = bce_contrib + ce_contrib + ucl_contrib + scl_contrib + info_ucl_contrib

    # Validation metrics - Multi-task learning (same structure as training)
    val_metrics = {}
    if accelerator.is_main_process and len(val_edema_preds_list) > 0:
        p_pos_all = torch.cat(val_edema_preds_list, dim=0).numpy()      # [N] P(edema=1)
        p_sub_all = torch.cat(val_subtype_preds_list, dim=0).numpy()    # [N, 2] P(NCPE|pos), P(CPE|pos)
        edema_all = torch.cat(val_edema_labels_list, dim=0).numpy()     # [N] in {0, 1, -1}
        subtype_all = torch.cat(val_subtype_labels_list, dim=0).numpy() # [N] in {0, 1, -1}

        # ==================== Level 1: Binary Edema Detection (0 vs 1) ====================
        mask_l1 = (edema_all == 0) | (edema_all == 1)
        y_l1 = edema_all[mask_l1].astype(int)       # {0, 1}
        p_l1 = p_pos_all[mask_l1]                   # P(pos)

        if mask_l1.sum() >= 2 and len(np.unique(y_l1)) >= 2:
            val_metrics['level1_auroc'] = roc_auc_score(y_l1, p_l1)
            val_metrics['level1_auprc'] = average_precision_score(y_l1, p_l1)
            val_metrics["level1_brier"] = brier_score_loss(y_l1, p_l1)
        else:
            val_metrics['level1_auroc'] = float('nan')
            val_metrics['level1_auprc'] = float('nan')
            val_metrics["level1_brier"] = float('nan')

        # ==================== Level 2: Subtype Classification (NCPE vs CPE | edema=1) ====================
        # Conditional: edema=1 AND subtype in {0, 1}
        mask_l2 = (edema_all == 1) & ((subtype_all == 0) | (subtype_all == 1))
        y_l2 = subtype_all[mask_l2].astype(int)           # Already 0=NCPE, 1=CPE
        p_l2 = p_sub_all[mask_l2, 1]                      # P(CPE|pos)

        if mask_l2.sum() >= 2 and len(np.unique(y_l2)) >= 2:
            val_metrics['level2_auroc'] = roc_auc_score(y_l2, p_l2)
            val_metrics['level2_auprc'] = average_precision_score(y_l2, p_l2)
        else:
            val_metrics['level2_auroc'] = float('nan')
            val_metrics['level2_auprc'] = float('nan')

        # ==================== Level 3: Final 3-Class (Neg, NCPE, CPE) ====================
        # 3-class GT is determined for samples:
        #   - edema==0 -> Neg(0)
        #   - edema==1 & subtype==0 -> NCPE(1)
        #   - edema==1 & subtype==1 -> CPE(2)
        mask_l3 = (edema_all == 0) | ((edema_all == 1) & ((subtype_all == 0) | (subtype_all == 1)))

        if mask_l3.sum() >= 3:
            edema_m = edema_all[mask_l3]
            subtype_m = subtype_all[mask_l3]
            p_pos_m = p_pos_all[mask_l3]
            p_sub_m = p_sub_all[mask_l3]

            y3 = np.zeros(mask_l3.sum(), dtype=int)
            y3[(edema_m == 1) & (subtype_m == 0)] = 1
            y3[(edema_m == 1) & (subtype_m == 1)] = 2

            p_neg = 1.0 - p_pos_m
            p_ncpe = p_pos_m * p_sub_m[:, 0]
            p_cpe = p_pos_m * p_sub_m[:, 1]
            probs_3 = np.stack([p_neg, p_ncpe, p_cpe], axis=1)

            y3_bin = label_binarize(y3, classes=[0, 1, 2])

            valid_classes = [k for k in range(3) if 0 < y3_bin[:, k].sum() < len(y3)]

            if len(valid_classes) >= 2:
                val_metrics['level3_auroc'] = roc_auc_score(
                    y3_bin, probs_3, average='macro', multi_class='ovr'
                )
                val_metrics['level3_auprc'] = average_precision_score(
                    y3_bin, probs_3, average='macro'
                )
            else:
                val_metrics['level3_auroc'] = float('nan')
                val_metrics['level3_auprc'] = float('nan')
        else:
            val_metrics['level3_auroc'] = float('nan')
            val_metrics['level3_auprc'] = float('nan')

    # ==================== Calibration Analysis ====================
    if accelerator.is_main_process and val_metrics:

        # Prepare calibration data
        y_true_dict = {}
        y_prob_dict = {}

        # Level 1: Binary Edema Detection
        if mask_l1.sum() >= 2 and len(np.unique(y_l1)) >= 2:
            y_true_dict['Edema Detection'] = y_l1
            y_prob_dict['Edema Detection'] = p_l1

        # Level 2: Subtype Classification
        if mask_l2.sum() >= 2 and len(np.unique(y_l2)) >= 2:
            y_true_dict['Subtype Classification'] = y_l2
            y_prob_dict['Subtype Classification'] = p_l2

        # Level 3: 3-class (per-class calibration)
        if mask_l3.sum() >= 3 and len(valid_classes) >= 2:
            # For each class, compute binary calibration (one-vs-rest)
            for class_idx, class_name in enumerate(['Negative', 'NCPE', 'CPE']):
                if class_idx in valid_classes:
                    y_binary = (y3 == class_idx).astype(int)
                    p_binary = probs_3[:, class_idx]
                    y_true_dict[f'3-class: {class_name}'] = y_binary
                    y_prob_dict[f'3-class: {class_name}'] = p_binary

        # Compute calibration metrics
        if len(y_true_dict) > 0:
            # Quick ECE calculation (no plots during training)
            ece_calc = ExpectedCalibrationError(n_bins=15)

            for task_name in y_true_dict.keys():
                ece, _ = ece_calc.compute(y_true_dict[task_name], y_prob_dict[task_name])
                val_metrics[f'{task_name}_ece'] = ece

            # For final epoch or specific epochs, generate plots
            if epoch == "final" or (epoch is not None and epoch % 10 == 0):
                save_dir = f'./output/calibration/{args.run_name}'
                prefix = f'epoch_{epoch}' if epoch != "final" else 'final'

                try:
                    calibration_results = analyze_calibration(
                        y_true_dict, y_prob_dict,
                        save_dir=save_dir,
                        prefix=prefix
                    )
                    # Store calibration results in metrics
                    for task_name, result in calibration_results.items():
                        val_metrics[f'{task_name}_calibration_quality'] = result['quality']
                except Exception as e:
                    print(f"⚠️  Calibration analysis failed: {e}")

    if accelerator.is_main_process:
        print("\n[Multi-Task Validation Summary]")
        print(f"Total Val Loss: {total_loss:.4f}")

        if val_metrics:
            print(f"\n[Hierarchical Performance Metrics]")
            print(f"[Edema Detection]   AUROC={val_metrics['level1_auroc']:.4f}  "
                f"AUPRC={val_metrics['level1_auprc']:.4f}  "
                f"Brier={val_metrics['level1_brier']:.4f}  "
                f"ECE={val_metrics.get('Edema Detection_ece', float('nan')):.4f}")

            print(f"[Subtype Classification] AUROC={val_metrics['level2_auroc']:.4f}  "
                f"AUPRC={val_metrics['level2_auprc']:.4f}  "
                f"ECE={val_metrics.get('Subtype Classification_ece', float('nan')):.4f}")

            print(f"[3-class Classification] AUROC={val_metrics['level3_auroc']:.4f}  "
                f"AUPRC={val_metrics['level3_auprc']:.4f}")

            # Print 3-class per-class ECE
            for class_name in ['Negative', 'NCPE', 'CPE']:
                ece_key = f'3-class: {class_name}_ece'
                if ece_key in val_metrics:
                    print(f"  └─ {class_name} ECE={val_metrics[ece_key]:.4f}")
            print()

    return total_loss, bce_avg, ce_avg, ucl_avg, scl_avg, info_ucl_avg, val_metrics


# Test 함수
def test(args, model, dataloader, loss_module, device, accelerator, dataset):
    test_loss, test_bce_avg, test_ce_avg, test_ucl_avg, test_scl, test_info_ucl, test_metrics = validate_multitask(
        args, model, dataloader, loss_module, device, accelerator, dataset, epoch="final"
    )

    wandb_test_metrics = {
        # Level 1: Binary Edema Detection
        'test/level1_auroc': test_metrics['level1_auroc'],
        'test/level1_auprc': test_metrics['level1_auprc'],
        'test/level1_brier': test_metrics['level1_brier'],
        # Level 2: Subtype Classification
        'test/level2_auroc': test_metrics['level2_auroc'],
        'test/level2_auprc': test_metrics['level2_auprc'],
        # Level 3: 3-class Combined
        'test/level3_auroc': test_metrics['level3_auroc'],
        'test/level3_auprc': test_metrics['level3_auprc'],
    }

    print("\n" + "="*80)
    print("📊 [Final Test Results]")
    print("="*80)

    print(f"\n   [Hierarchical Performance Metrics]")
    print(f"[Edema Detection]   AUROC={test_metrics['level1_auroc']:.4f}  "
        f"AUPRC={test_metrics['level1_auprc']:.4f}  "
        f"Brier={test_metrics['level1_brier']:.4f}")

    print(f"[Subtype Classification] AUROC={test_metrics['level2_auroc']:.4f}  "
        f"AUPRC={test_metrics['level2_auprc']:.4f}")

    print(f"[3-class Classification] AUROC={test_metrics['level3_auroc']:.4f}  "
        f"AUPRC={test_metrics['level3_auprc']:.4f}")
    print("="*80 + "\n")

    return test_loss, test_bce_avg, test_ce_avg, test_ucl_avg, test_scl, test_info_ucl, test_metrics, wandb_test_metrics