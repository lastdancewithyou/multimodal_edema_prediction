import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore', message='Spectral initialisation failed')
from umap import UMAP

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils.utils import timer
from training.engine import train_batch
from analysis.calibration import ExpectedCalibrationError, analyze_calibration


def validate_multitask(args, model, dataloader, loss_module, device, accelerator, dataset, epoch=None, disable_cxr=False, disable_txt=True, max_length=256):
    print("=====Running Multi-Task Validation=====")
    model.eval()

    bce_sum = torch.zeros(1, device=device, dtype=torch.float32)
    bce_count = torch.zeros(1, device=device, dtype=torch.float32)
    ce_sum = torch.zeros(1, device=device, dtype=torch.float32)
    ce_count = torch.zeros(1, device=device, dtype=torch.float32)

    val_edema_preds_list = []
    val_edema_labels_list = []
    val_subtype_preds_list = []
    val_subtype_labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="🤖 <Multi-Task Validation>"):
            _, batch_bce, batch_ce, batch_outputs, batch_counts = train_batch(
                args=args,
                model=model,
                batch=batch,
                loss_module=loss_module,
                device=device,
                accelerator=accelerator,
                dataset=dataset,
                max_length=max_length,
                disable_cxr=disable_cxr,
                # disable_txt=disable_txt,
                disable_txt=False,
                bce_weight=args.bce_weight,
                ce_weight=args.ce_weight,
                is_training=False,
            )

            bce_ct_local = torch.as_tensor(batch_counts['bce_count'], device=device, dtype=torch.float32)
            ce_ct_local = torch.as_tensor(batch_counts['ce_count'], device=device, dtype=torch.float32)

            bce_sum += torch.as_tensor(batch_bce, device=device, dtype=torch.float32) * bce_ct_local
            bce_count += bce_ct_local
            ce_sum += torch.as_tensor(batch_ce, device=device, dtype=torch.float32) * ce_ct_local
            ce_count += ce_ct_local

            edema_logits = batch_outputs['edema_logits'].squeeze(-1)  # [B]
            subtype_logits = batch_outputs['subtype_logits']           # [B, 3]
            edema_labels = batch_outputs['edema_labels']               # [B]
            subtype_labels = batch_outputs['subtype_labels']           # [B]

            p_pos = torch.sigmoid(edema_logits)                        # [B] P(edema=1)
            p_sub = torch.softmax(subtype_logits, dim=-1)              # [B, 3] P(Intermediate|pos), P(NCPE|pos), P(CPE|pos)

            val_edema_preds_list.append(p_pos.detach().cpu())
            val_subtype_preds_list.append(p_sub.detach().cpu())

            val_edema_labels_list.append(edema_labels.detach().cpu())
            val_subtype_labels_list.append(subtype_labels.detach().cpu())

    # GPU aggregation
    if accelerator.num_processes > 1:
        total_bce_sum = accelerator.gather_for_metrics(bce_sum).sum()
        total_bce_count = accelerator.gather_for_metrics(bce_count).sum()
        total_ce_sum = accelerator.gather_for_metrics(ce_sum).sum()
        total_ce_count = accelerator.gather_for_metrics(ce_count).sum()
    else:
        total_bce_sum = bce_sum
        total_bce_count = bce_count
        total_ce_sum = ce_sum
        total_ce_count = ce_count

    bce_avg = (total_bce_sum / (total_bce_count + 1e-8)).item()
    ce_avg = (total_ce_sum / (total_ce_count + 1e-8)).item()

    bce_contrib = args.bce_weight * bce_avg
    ce_contrib = args.ce_weight * ce_avg
    total_loss = bce_contrib + ce_contrib

    # Gather predictions from all GPUs
    if accelerator.num_processes > 1:
        local_preds = {
            'p_pos': [p.cpu() for p in val_edema_preds_list],
            'p_sub': [p.cpu() for p in val_subtype_preds_list],
            'edema': [e.cpu() for e in val_edema_labels_list],
            'subtype': [s.cpu() for s in val_subtype_labels_list]
        }

        # Gather to rank 0 only
        if accelerator.is_main_process:
            gathered_preds = [None] * accelerator.num_processes
            dist.gather_object(local_preds, gathered_preds, dst=0)

            all_p_pos = []
            all_p_sub = []
            all_edema = []
            all_subtype = []

            for gpu_preds in gathered_preds:
                all_p_pos.extend(gpu_preds['p_pos'])
                all_p_sub.extend(gpu_preds['p_sub'])
                all_edema.extend(gpu_preds['edema'])
                all_subtype.extend(gpu_preds['subtype'])

            p_pos_all = torch.cat(all_p_pos, dim=0).numpy() if all_p_pos else np.array([])
            p_sub_all = torch.cat(all_p_sub, dim=0).numpy() if all_p_sub else np.array([])
            edema_all = torch.cat(all_edema, dim=0).numpy() if all_edema else np.array([])
            subtype_all = torch.cat(all_subtype, dim=0).numpy() if all_subtype else np.array([])
        else:
            dist.gather_object(local_preds, dst=0)
            p_pos_all = None
            p_sub_all = None
            edema_all = None
            subtype_all = None

        accelerator.wait_for_everyone()
    else:
        # Single GPU
        if len(val_edema_preds_list) > 0:
            p_pos_all = torch.cat(val_edema_preds_list, dim=0).numpy()
            p_sub_all = torch.cat(val_subtype_preds_list, dim=0).numpy()
            edema_all = torch.cat(val_edema_labels_list, dim=0).numpy()
            subtype_all = torch.cat(val_subtype_labels_list, dim=0).numpy()
        else:
            p_pos_all = None

    # Validation metrics - Multi-task learning
    val_metrics = {}
    if accelerator.is_main_process and p_pos_all is not None and len(p_pos_all) > 0:

        # ==================== Level 1: Binary Edema Detection (0 vs 1) ====================
        mask_l1 = (edema_all == 0) | (edema_all == 1)
        y_l1 = edema_all[mask_l1].astype(int)       # {0, 1}
        p_l1 = p_pos_all[mask_l1]                   # P(pos)

        if mask_l1.sum() >= 2 and len(np.unique(y_l1)) >= 2:
            val_metrics['level1_auroc'] = roc_auc_score(y_l1, p_l1)
            val_metrics['level1_auprc'] = average_precision_score(y_l1, p_l1)
        else:
            val_metrics['level1_auroc'] = float('nan')
            val_metrics['level1_auprc'] = float('nan')

        # ==================== Level 2: Subtype Classification (Intermediate vs NCPE vs CPE | edema=1) ====================
        # 3-way classification: Intermediate (0) vs NCPE (1) vs CPE (2)
        mask_l2 = (edema_all == 1) & ((subtype_all == 0) | (subtype_all == 1) | (subtype_all == 2))

        if mask_l2.sum() >= 2 and len(np.unique(subtype_all[mask_l2])) >= 2:
            y_l2 = subtype_all[mask_l2].astype(int)  # {0, 1, 2}
            y_l2_bin = label_binarize(y_l2, classes=[0, 1, 2])
            p_l2_probs = p_sub_all[mask_l2, :]  # [N, 3]

            val_metrics['level2_auroc'] = roc_auc_score(y_l2_bin, p_l2_probs, average='macro', multi_class='ovr')
            val_metrics['level2_auprc'] = average_precision_score(y_l2_bin, p_l2_probs, average='macro')
        else:
            val_metrics['level2_auroc'] = float('nan')
            val_metrics['level2_auprc'] = float('nan')

        # # ==================== Level 3: Final 4-Class (Neg, Intermediate, NCPE, CPE) ====================
        # # 4-class GT is determined for samples:
        # #   - edema==0 -> Negative(0)
        # #   - edema==1 & subtype==0 -> Intermediate(1)
        # #   - edema==1 & subtype==1 -> NCPE(2)
        # #   - edema==1 & subtype==2 -> CPE(3)
        # mask_l3 = (edema_all == 0) | ((edema_all == 1) & ((subtype_all == 0) | (subtype_all == 1) | (subtype_all == 2)))

        # if mask_l3.sum() >= 3:
        #     edema_m = edema_all[mask_l3]
        #     subtype_m = subtype_all[mask_l3]
        #     p_pos_m = p_pos_all[mask_l3]
        #     p_sub_m = p_sub_all[mask_l3]

        #     # Ground truth mapping: 4 classes
        #     y4 = np.zeros(mask_l3.sum(), dtype=int)  # Default: Negative (0)
        #     y4[(edema_m == 1) & (subtype_m == 0)] = 1  # Intermediate
        #     y4[(edema_m == 1) & (subtype_m == 1)] = 2  # NCPE
        #     y4[(edema_m == 1) & (subtype_m == 2)] = 3  # CPE

        #     # Predicted probabilities: 4 classes
        #     p_neg = 1.0 - p_pos_m
        #     p_intermediate = p_pos_m * p_sub_m[:, 0]
        #     p_ncpe = p_pos_m * p_sub_m[:, 1]
        #     p_cpe = p_pos_m * p_sub_m[:, 2]
        #     probs_4 = np.stack([p_neg, p_intermediate, p_ncpe, p_cpe], axis=1)

        #     y4_bin = label_binarize(y4, classes=[0, 1, 2, 3])

        #     valid_classes = [k for k in range(4) if 0 < y4_bin[:, k].sum() < len(y4)]

        #     if len(valid_classes) >= 2:
        #         val_metrics['level3_auroc'] = roc_auc_score(y4_bin, probs_4, average='macro', multi_class='ovr')
        #         val_metrics['level3_auprc'] = average_precision_score(y4_bin, probs_4, average='macro')
        #     else:
        #         val_metrics['level3_auroc'] = float('nan')
        #         val_metrics['level3_auprc'] = float('nan')
        # else:
        #     val_metrics['level3_auroc'] = float('nan')
        #     val_metrics['level3_auprc'] = float('nan')

    # # ==================== Calibration Analysis ====================
    # if accelerator.is_main_process and val_metrics:
    #     # Prepare calibration data
    #     y_true_dict = {}
    #     y_prob_dict = {}

    #     # Level 1: Binary Edema Detection
    #     if mask_l1.sum() >= 2 and len(np.unique(y_l1)) >= 2:
    #         y_true_dict['Edema Detection'] = y_l1
    #         y_prob_dict['Edema Detection'] = p_l1

    #     # Level 2: Subtype Classification
    #     if mask_l2.sum() >= 2 and len(np.unique(y_l2)) >= 2:
    #         y_true_dict['Subtype Classification'] = y_l2
    #         y_prob_dict['Subtype Classification'] = p_l2

    #     # Level 3: 3-class (per-class calibration)
    #     if mask_l3.sum() >= 3 and len(valid_classes) >= 2:
    #         # For each class, compute binary calibration (one-vs-rest)
    #         for class_idx, class_name in enumerate(['Negative', 'NCPE', 'CPE']):
    #             if class_idx in valid_classes:
    #                 y_binary = (y3 == class_idx).astype(int)
    #                 p_binary = probs_3[:, class_idx]
    #                 y_true_dict[f'3-class: {class_name}'] = y_binary
    #                 y_prob_dict[f'3-class: {class_name}'] = p_binary

    #     # Compute calibration metrics
    #     if len(y_true_dict) > 0:
    #         ece_calc = ExpectedCalibrationError(n_bins=15)

    #         for task_name in y_true_dict.keys():
    #             ece, _ = ece_calc.compute(y_true_dict[task_name], y_prob_dict[task_name])
    #             val_metrics[f'{task_name}_ece'] = ece

    #         if epoch == "final" or (epoch is not None and epoch % 10 == 0):
    #             save_dir = f'./output/calibration/{args.run_name}'
    #             prefix = f'epoch_{epoch}' if epoch != "final" else 'final'

    #             try:
    #                 calibration_results = analyze_calibration(
    #                     y_true_dict, y_prob_dict,
    #                     save_dir=save_dir,
    #                     prefix=prefix
    #                 )
    #                 # Store calibration results in metrics
    #                 for task_name, result in calibration_results.items():
    #                     val_metrics[f'{task_name}_calibration_quality'] = result['quality']
    #             except Exception as e:
    #                 print(f"⚠️  Calibration analysis failed: {e}")

    if accelerator.is_main_process:
        print("\n[Multi-Task Validation Summary]")
        print(f"Total Val Loss: {total_loss:.4f}")

        if val_metrics:
            print(f"\n[Hierarchical Performance Metrics]")
            print(f"[Level 1: Edema Detection]    AUROC={val_metrics['level1_auroc']:.4f}  "
                f"AUPRC={val_metrics['level1_auprc']:.4f}  "
                # f"Brier={val_metrics['level1_brier']:.4f}"
                )

            print(f"[Level 2: Subtype (3-way)]    AUROC={val_metrics['level2_auroc']:.4f}  "
                f"AUPRC={val_metrics['level2_auprc']:.4f}")

            # print(f"[Level 3: Combined (4-class)] AUROC={val_metrics['level3_auroc']:.4f}  "
            #     f"AUPRC={val_metrics['level3_auprc']:.4f}")
            print()

    # return total_loss, bce_avg, ce_avg, mse_avg, val_metrics
    return total_loss, bce_avg, ce_avg, val_metrics


# Test 함수
def test(args, model, dataloader, loss_module, device, accelerator, dataset):
    test_loss, test_bce_avg, test_ce_avg, test_metrics = validate_multitask(
        args, model, dataloader, loss_module, device, accelerator, dataset,
        epoch="final",
        disable_cxr=args.disable_cxr,
        disable_txt=args.disable_txt,
        max_length=args.token_max_length,
    )

    wandb_test_metrics = {}
    if accelerator.is_main_process and test_metrics:
        wandb_test_metrics = {
            # Level 1: Binary Edema Detection
            'test/level1_auroc': test_metrics['level1_auroc'],
            'test/level1_auprc': test_metrics['level1_auprc'],
            # Level 2: Subtype Classification
            'test/level2_auroc': test_metrics['level2_auroc'],
            'test/level2_auprc': test_metrics['level2_auprc'],
        }

        print("\n" + "="*80)
        print("📊 [Final Test Results]")
        print("="*80)

        print(f"\n   [Hierarchical Performance Metrics]")
        print(f"[Level 1: Edema Detection]    AUROC={test_metrics['level1_auroc']:.4f}  "
            f"AUPRC={test_metrics['level1_auprc']:.4f}")

        print(f"[Level 2: Subtype (3-way)]    AUROC={test_metrics['level2_auroc']:.4f}  "
            f"AUPRC={test_metrics['level2_auprc']:.4f}")

        print("="*80 + "\n")

    return test_loss, test_bce_avg, test_ce_avg, test_metrics, wandb_test_metrics
