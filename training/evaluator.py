import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore', message='Spectral initialisation failed')

from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.distributed as dist

from utils.utils import timer
from training.engine import train_batch


def _threshold_free_metrics(p_all, lab_all, cxr_anchor_all):
    out = {'auroc': float('nan'), 'auprc': float('nan')}
    if p_all is None or lab_all is None or len(p_all) == 0:
        return out

    valid = (cxr_anchor_all == 1) & (~np.isnan(lab_all))
    if valid.sum() < 2:
        return out

    y = lab_all[valid].astype(int)
    p = p_all[valid]
    if len(np.unique(y)) < 2:
        return out

    out['auroc'] = roc_auc_score(y, p)
    out['auprc'] = average_precision_score(y, p)
    return out


def validate_multitask(args, model, dataloader, loss_module, device, accelerator, epoch=None, disable_cxr=False, disable_txt=True):
    print("=====Running Multi-Task Validation=====")
    model.eval()

    bce_sum = torch.zeros(1, device=device, dtype=torch.float32)
    bce_count = torch.zeros(1, device=device, dtype=torch.float32)
    cardio_sum = torch.zeros(1, device=device, dtype=torch.float32)
    cardio_count = torch.zeros(1, device=device, dtype=torch.float32)
    pneumo_sum = torch.zeros(1, device=device, dtype=torch.float32)
    pneumo_count = torch.zeros(1, device=device, dtype=torch.float32)

    val_p_pos_list = []
    val_edema_hard_list = []
    val_cxr_anchor_list = []
    val_cardio_p_list = []
    val_cardio_lab_list = []
    val_pneumo_p_list = []
    val_pneumo_lab_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="🤖 <Multi-Task Validation>"):
            _, batch_bce, _, batch_cardio, batch_pneumo, _, batch_outputs, batch_counts = train_batch(
                args=args,
                model=model,
                batch=batch,
                loss_module=loss_module,
                device=device,
                accelerator=accelerator,
                disable_cxr=disable_cxr,
                disable_txt=disable_txt,
                bce_weight=args.bce_weight,
                cardio_weight=args.cardio_weight,
                pneumo_weight=args.pneumo_weight,
                info_weight=0.0,  # validation/test에서는 InfoNCE alignment 비활성
                is_training=False,
            )

            bce_ct_local = torch.as_tensor(batch_counts['bce_count'], device=device, dtype=torch.float32)
            bce_sum += torch.as_tensor(batch_bce, device=device, dtype=torch.float32) * bce_ct_local
            bce_count += bce_ct_local

            cardio_ct_local = torch.as_tensor(batch_counts['cardio_count'], device=device, dtype=torch.float32)
            cardio_sum += torch.as_tensor(batch_cardio, device=device, dtype=torch.float32) * cardio_ct_local
            cardio_count += cardio_ct_local

            pneumo_ct_local = torch.as_tensor(batch_counts['pneumo_count'], device=device, dtype=torch.float32)
            pneumo_sum += torch.as_tensor(batch_pneumo, device=device, dtype=torch.float32) * pneumo_ct_local
            pneumo_count += pneumo_ct_local

            edema_logits = batch_outputs['edema_logits'].squeeze(-1)
            edema_hard = batch_outputs['edema_hard_labels']
            cxr_anchor = batch_outputs['cxr_anchor_mask']

            p_pos = torch.sigmoid(edema_logits)

            val_p_pos_list.append(p_pos.detach().cpu())
            val_edema_hard_list.append(edema_hard.detach().cpu())
            val_cxr_anchor_list.append(cxr_anchor.detach().cpu())

            cardio_p = torch.sigmoid(batch_outputs['cardiomegaly_logits'].squeeze(-1))
            pneumo_p = torch.sigmoid(batch_outputs['pneumonia_logits'].squeeze(-1))
            val_cardio_p_list.append(cardio_p.detach().cpu())
            val_cardio_lab_list.append(batch_outputs['cardiomegaly_labels'].detach().cpu())
            val_pneumo_p_list.append(pneumo_p.detach().cpu())
            val_pneumo_lab_list.append(batch_outputs['pneumonia_labels'].detach().cpu())

    # Loss aggregation across GPUs
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(bce_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bce_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(cardio_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(cardio_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(pneumo_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(pneumo_count, op=dist.ReduceOp.SUM)

    bce_avg = (bce_sum / (bce_count + 1e-8)).item()
    cardio_avg = (cardio_sum / (cardio_count + 1e-8)).item()
    pneumo_avg = (pneumo_sum / (pneumo_count + 1e-8)).item()
    total_loss = (
        args.bce_weight * bce_avg
        + args.cardio_weight * cardio_avg
        + args.pneumo_weight * pneumo_avg
    )

    # Gather predictions from all GPUs
    if accelerator.num_processes > 1:
        local_preds = {
            'p_pos': [p.cpu() for p in val_p_pos_list],
            'edema_hard': [e.cpu() for e in val_edema_hard_list],
            'cxr_anchor': [c.cpu() for c in val_cxr_anchor_list],
            'cardio_p': [p.cpu() for p in val_cardio_p_list],
            'cardio_lab': [l.cpu() for l in val_cardio_lab_list],
            'pneumo_p': [p.cpu() for p in val_pneumo_p_list],
            'pneumo_lab': [l.cpu() for l in val_pneumo_lab_list],
        }

        if accelerator.is_main_process:
            gathered_preds = [None] * accelerator.num_processes
            dist.gather_object(local_preds, gathered_preds, dst=0)

            all_p_pos, all_edema_hard, all_cxr_anchor = [], [], []
            all_cardio_p, all_cardio_lab = [], []
            all_pneumo_p, all_pneumo_lab = [], []
            for gpu_preds in gathered_preds:
                all_p_pos.extend(gpu_preds['p_pos'])
                all_edema_hard.extend(gpu_preds['edema_hard'])
                all_cxr_anchor.extend(gpu_preds['cxr_anchor'])
                all_cardio_p.extend(gpu_preds['cardio_p'])
                all_cardio_lab.extend(gpu_preds['cardio_lab'])
                all_pneumo_p.extend(gpu_preds['pneumo_p'])
                all_pneumo_lab.extend(gpu_preds['pneumo_lab'])

            p_pos_all = torch.cat(all_p_pos, dim=0).numpy() if all_p_pos else np.array([])
            edema_hard_all = torch.cat(all_edema_hard, dim=0).numpy() if all_edema_hard else np.array([])
            cxr_anchor_all = torch.cat(all_cxr_anchor, dim=0).numpy() if all_cxr_anchor else np.array([])
            cardio_p_all = torch.cat(all_cardio_p, dim=0).numpy() if all_cardio_p else np.array([])
            cardio_lab_all = torch.cat(all_cardio_lab, dim=0).numpy() if all_cardio_lab else np.array([])
            pneumo_p_all = torch.cat(all_pneumo_p, dim=0).numpy() if all_pneumo_p else np.array([])
            pneumo_lab_all = torch.cat(all_pneumo_lab, dim=0).numpy() if all_pneumo_lab else np.array([])
        else:
            dist.gather_object(local_preds, dst=0)
            p_pos_all = None
            edema_hard_all = None
            cxr_anchor_all = None
            cardio_p_all = cardio_lab_all = None
            pneumo_p_all = pneumo_lab_all = None

        accelerator.wait_for_everyone()
    else:
        if len(val_p_pos_list) > 0:
            p_pos_all = torch.cat(val_p_pos_list, dim=0).numpy()
            edema_hard_all = torch.cat(val_edema_hard_list, dim=0).numpy()
            cxr_anchor_all = torch.cat(val_cxr_anchor_list, dim=0).numpy()
            cardio_p_all = torch.cat(val_cardio_p_list, dim=0).numpy()
            cardio_lab_all = torch.cat(val_cardio_lab_list, dim=0).numpy()
            pneumo_p_all = torch.cat(val_pneumo_p_list, dim=0).numpy()
            pneumo_lab_all = torch.cat(val_pneumo_lab_list, dim=0).numpy()
        else:
            p_pos_all = None
            cardio_p_all = cardio_lab_all = None
            pneumo_p_all = pneumo_lab_all = None

    # Validation metrics — cxr_flag==1 & edema_hard ∈ {0, 1}
    val_metrics = {}
    if accelerator.is_main_process and p_pos_all is not None and len(p_pos_all) > 0:
        mask = (cxr_anchor_all == 1) & ((edema_hard_all == 0) | (edema_hard_all == 1))
        if mask.sum() >= 2 and len(np.unique(edema_hard_all[mask])) >= 2:
            y = edema_hard_all[mask].astype(int)
            p = p_pos_all[mask]
            val_metrics['auroc'] = roc_auc_score(y, p)
            val_metrics['auprc'] = average_precision_score(y, p)
        else:
            val_metrics['auroc'] = float('nan')
            val_metrics['auprc'] = float('nan')

        # Sub-task metrics: threshold-free (AUROC/AUPRC) only — F1처럼 0.5 cut-off 의존 지표는 제외
        for prefix, p_all, lab_all in [
            ('cardio', cardio_p_all, cardio_lab_all),
            ('pneumo', pneumo_p_all, pneumo_lab_all),
        ]:
            metrics = _threshold_free_metrics(p_all, lab_all, cxr_anchor_all)
            val_metrics[f'{prefix}_auroc'] = metrics['auroc']
            val_metrics[f'{prefix}_auprc'] = metrics['auprc']

        # Loss components도 dict에 함께 노출 (trainer/wandb에서 사용)
        val_metrics['bce_loss'] = bce_avg
        val_metrics['cardio_loss'] = cardio_avg
        val_metrics['pneumo_loss'] = pneumo_avg

    if accelerator.is_main_process:
        print("\n[Multi-Task Validation Summary]")
        print(f"Total Val Loss: {total_loss:.4f}")
        print(f"   [Loss Components]")
        print(f"      BCE (Edema soft): {bce_avg:.4f} (λ={args.bce_weight})")
        print(f"      Cardio BCE:       {cardio_avg:.4f} (λ={args.cardio_weight})")
        print(f"      Pneumo BCE:       {pneumo_avg:.4f} (λ={args.pneumo_weight})")

        if val_metrics:
            print("\n   [Edema (cxr_flag==1)]")
            print(f"   AUROC={val_metrics['auroc']:.4f}  "
                  f"AUPRC={val_metrics['auprc']:.4f}")
            print("\n   [Cardiomegaly (cxr_flag==1 & label observed)]")
            print(f"   Loss={val_metrics['cardio_loss']:.4f}  "
                  f"AUROC={val_metrics['cardio_auroc']:.4f}  "
                  f"AUPRC={val_metrics['cardio_auprc']:.4f}")
            print("\n   [Pneumonia (cxr_flag==1 & label observed)]")
            print(f"   Loss={val_metrics['pneumo_loss']:.4f}  "
                  f"AUROC={val_metrics['pneumo_auroc']:.4f}  "
                  f"AUPRC={val_metrics['pneumo_auprc']:.4f}")
            print()

    return total_loss, bce_avg, val_metrics


def test(args, model, dataloader, loss_module, device, accelerator):
    test_loss, test_bce_avg, test_metrics = validate_multitask(
        args, model, dataloader, loss_module, device, accelerator,
        epoch="final",
        disable_cxr=args.disable_cxr,
        disable_txt=args.disable_txt,
    )

    wandb_test_metrics = {}
    if accelerator.is_main_process and test_metrics:
        wandb_test_metrics = {
            'test/auroc': test_metrics['auroc'],
            'test/auprc': test_metrics['auprc'],
            'test/cardio_loss': test_metrics.get('cardio_loss', float('nan')),
            'test/cardio_auroc': test_metrics.get('cardio_auroc', float('nan')),
            'test/cardio_auprc': test_metrics.get('cardio_auprc', float('nan')),
            'test/pneumo_loss': test_metrics.get('pneumo_loss', float('nan')),
            'test/pneumo_auroc': test_metrics.get('pneumo_auroc', float('nan')),
            'test/pneumo_auprc': test_metrics.get('pneumo_auprc', float('nan')),
        }

        print("\n" + "="*80)
        print("📊 [Final Test Results]")
        print("="*80)
        print("\n   [Edema (cxr_flag==1)]")
        print(f"   AUROC={test_metrics['auroc']:.4f}  "
              f"AUPRC={test_metrics['auprc']:.4f}")
        print("\n   [Cardiomegaly (cxr_flag==1 & label observed)]")
        print(f"   Loss={test_metrics.get('cardio_loss', float('nan')):.4f}  "
              f"AUROC={test_metrics.get('cardio_auroc', float('nan')):.4f}  "
              f"AUPRC={test_metrics.get('cardio_auprc', float('nan')):.4f}")
        print("\n   [Pneumonia (cxr_flag==1 & label observed)]")
        print(f"   Loss={test_metrics.get('pneumo_loss', float('nan')):.4f}  "
              f"AUROC={test_metrics.get('pneumo_auroc', float('nan')):.4f}  "
              f"AUPRC={test_metrics.get('pneumo_auprc', float('nan')):.4f}")
        print("="*80 + "\n")

    return test_loss, test_bce_avg, test_metrics, wandb_test_metrics
