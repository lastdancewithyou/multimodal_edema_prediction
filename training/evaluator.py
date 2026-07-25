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


def validate_multitask(args, model, dataloader, loss_module, device, accelerator, epoch=None, disable_cxr=False, disable_txt=False, eval_path='deploy'):
    """
    'deploy' (default): TS+CXR만 (text 영구 제외)
    'priv'            : TS+CXR+Text 모두 사용
    """
    print(f"=====Running Multi-Task Validation [{eval_path}]=====")
    model.eval()

    bce_sum = torch.zeros(1, device=device, dtype=torch.float32)
    bce_count = torch.zeros(1, device=device, dtype=torch.float32)

    val_p_pos_list = []
    val_edema_hard_list = []
    val_cxr_anchor_list = []

    # Subtype eval buffers (softmax probs over [mixed, NCPE, CPE], hard label, valid mask)
    val_subtype_probs_list = []
    val_subtype_label_list = []
    val_subtype_mask_list  = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="🤖 <Multi-Task Validation>"):
            disable_txt_local = disable_txt if eval_path == 'deploy' else False
            _, _, batch_bce, batch_outputs, batch_counts = train_batch(
                args=args,
                model=model,
                batch=batch,
                loss_module=loss_module,
                device=device,
                accelerator=accelerator,
                disable_cxr=disable_cxr,
                disable_txt=disable_txt_local,
                is_training=False,
                eval_path=eval_path,
            )

            bce_ct_local = torch.as_tensor(batch_counts['bce_count'], device=device, dtype=torch.float32)
            bce_sum += torch.as_tensor(batch_bce, device=device, dtype=torch.float32) * bce_ct_local
            bce_count += bce_ct_local

            edema_logits = batch_outputs['edema_logits'].squeeze(-1)
            edema_hard = batch_outputs['edema_hard_labels']
            cxr_anchor = batch_outputs['cxr_anchor_mask']

            p_pos = torch.sigmoid(edema_logits)

            val_p_pos_list.append(p_pos.detach().cpu())
            val_edema_hard_list.append(edema_hard.detach().cpu())
            val_cxr_anchor_list.append(cxr_anchor.detach().cpu())

            # Subtype: softmax over 3 classes; metric uses hard label where subtype_mask==1
            subtype_logits = batch_outputs.get('subtype_logits', None)
            if subtype_logits is not None:
                subtype_probs = torch.softmax(subtype_logits.float(), dim=-1)
                val_subtype_probs_list.append(subtype_probs.detach().cpu())
                val_subtype_label_list.append(batch_outputs['subtype_label'].detach().cpu())
                val_subtype_mask_list.append(batch_outputs['subtype_mask'].detach().cpu())

    # Loss aggregation across GPUs
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(bce_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bce_count, op=dist.ReduceOp.SUM)

    bce_avg = (bce_sum / (bce_count + 1e-8)).item()
    total_loss = bce_avg     # LUPI eval은 deploy BCE 자체

    # Gather predictions from all GPUs
    if accelerator.num_processes > 1:
        local_preds = {
            'p_pos': [p.cpu() for p in val_p_pos_list],
            'edema_hard': [e.cpu() for e in val_edema_hard_list],
            'cxr_anchor': [c.cpu() for c in val_cxr_anchor_list],
            'subtype_probs': [p.cpu() for p in val_subtype_probs_list],
            'subtype_label': [h.cpu() for h in val_subtype_label_list],
            'subtype_mask':  [m.cpu() for m in val_subtype_mask_list],
        }

        if accelerator.is_main_process:
            gathered_preds = [None] * accelerator.num_processes
            dist.gather_object(local_preds, gathered_preds, dst=0)

            all_p_pos, all_edema_hard, all_cxr_anchor = [], [], []
            all_sub_probs, all_sub_label, all_sub_mask = [], [], []
            for gpu_preds in gathered_preds:
                all_p_pos.extend(gpu_preds['p_pos'])
                all_edema_hard.extend(gpu_preds['edema_hard'])
                all_cxr_anchor.extend(gpu_preds['cxr_anchor'])
                all_sub_probs.extend(gpu_preds['subtype_probs'])
                all_sub_label.extend(gpu_preds['subtype_label'])
                all_sub_mask.extend(gpu_preds['subtype_mask'])

            p_pos_all = torch.cat(all_p_pos, dim=0).numpy() if all_p_pos else np.array([])
            edema_hard_all = torch.cat(all_edema_hard, dim=0).numpy() if all_edema_hard else np.array([])
            cxr_anchor_all = torch.cat(all_cxr_anchor, dim=0).numpy() if all_cxr_anchor else np.array([])
            subtype_probs_all = torch.cat(all_sub_probs, dim=0).numpy() if all_sub_probs else np.empty((0, 3))
            subtype_label_all = torch.cat(all_sub_label, dim=0).numpy() if all_sub_label else np.array([])
            subtype_mask_all  = torch.cat(all_sub_mask,  dim=0).numpy() if all_sub_mask  else np.array([])
        else:
            dist.gather_object(local_preds, dst=0)
            p_pos_all = None
            edema_hard_all = None
            cxr_anchor_all = None
            subtype_probs_all = None
            subtype_label_all = None
            subtype_mask_all = None

        accelerator.wait_for_everyone()
    else:
        if len(val_p_pos_list) > 0:
            p_pos_all = torch.cat(val_p_pos_list, dim=0).numpy()
            edema_hard_all = torch.cat(val_edema_hard_list, dim=0).numpy()
            cxr_anchor_all = torch.cat(val_cxr_anchor_list, dim=0).numpy()
        else:
            p_pos_all = None

        if len(val_subtype_probs_list) > 0:
            subtype_probs_all = torch.cat(val_subtype_probs_list, dim=0).numpy()
            subtype_label_all = torch.cat(val_subtype_label_list, dim=0).numpy()
            subtype_mask_all  = torch.cat(val_subtype_mask_list,  dim=0).numpy()
        else:
            subtype_probs_all = np.empty((0, 3))
            subtype_label_all = np.array([])
            subtype_mask_all  = np.array([])

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

        val_metrics['bce_loss'] = bce_avg

        # ── Subtype OvR AUROC/AUPRC (mixed=0, NCPE=1, CPE=2) on subtype_mask==1 ──
        subtype_names = ['mixed', 'ncpe', 'cpe']
        sub_metrics = {f'auroc_{n}': float('nan') for n in subtype_names}
        sub_metrics.update({f'auprc_{n}': float('nan') for n in subtype_names})
        sub_metrics.update({'auroc_macro': float('nan'), 'auprc_macro': float('nan'), 'n_valid': 0})

        if subtype_probs_all is not None and len(subtype_probs_all) > 0:
            sm = (subtype_mask_all == 1) & (subtype_label_all >= 0)
            n_valid = int(sm.sum())
            sub_metrics['n_valid'] = n_valid
            if n_valid >= 2:
                y_sub = subtype_label_all[sm].astype(int)
                p_sub = subtype_probs_all[sm]                         # [N, 3]
                aurocs, auprcs = [], []
                for c, name in enumerate(subtype_names):
                    y_bin = (y_sub == c).astype(int)
                    # AUROC requires both positives and negatives present for class c.
                    if len(np.unique(y_bin)) >= 2:
                        a = roc_auc_score(y_bin, p_sub[:, c])
                        ap = average_precision_score(y_bin, p_sub[:, c])
                        sub_metrics[f'auroc_{name}'] = a
                        sub_metrics[f'auprc_{name}'] = ap
                        aurocs.append(a)
                        auprcs.append(ap)
                if aurocs:
                    sub_metrics['auroc_macro'] = float(np.mean(aurocs))
                if auprcs:
                    sub_metrics['auprc_macro'] = float(np.mean(auprcs))

        val_metrics['subtype'] = sub_metrics

    if accelerator.is_main_process:
        print("\n[Multi-Task Validation Summary]")
        print(f"Edema soft: {bce_avg:.4f}")

        if val_metrics:
            print("\n   [Edema (cxr_flag==1)]")
            print(f"   AUROC={val_metrics['auroc']:.4f}  "
                  f"AUPRC={val_metrics['auprc']:.4f}")

            sm = val_metrics.get('subtype', {})
            print(f"\n   [Subtype (subtype_mask==1, n={sm.get('n_valid', 0):,})]")
            print(f"   AUROC  macro={sm.get('auroc_macro', float('nan')):.4f}  "
                  f"mixed={sm.get('auroc_mixed', float('nan')):.4f}  "
                  f"ncpe={sm.get('auroc_ncpe',  float('nan')):.4f}  "
                  f"cpe={sm.get('auroc_cpe',   float('nan')):.4f}")
            print(f"   AUPRC  macro={sm.get('auprc_macro', float('nan')):.4f}  "
                  f"mixed={sm.get('auprc_mixed', float('nan')):.4f}  "
                  f"ncpe={sm.get('auprc_ncpe',  float('nan')):.4f}  "
                  f"cpe={sm.get('auprc_cpe',   float('nan')):.4f}")
            print()

    return total_loss, bce_avg, val_metrics


def test(args, model, dataloader, loss_module, device, accelerator):
    # Deploy path (TS+CXR only) — 배포 모델 평가 (text 영구 제외).
    test_loss, test_bce_avg, test_metrics = validate_multitask(
        args, model, dataloader, loss_module, device, accelerator,
        epoch="final",
        disable_cxr=args.disable_cxr,
        disable_txt=True,
        eval_path='deploy',
    )

    # Priv path (TS+CXR+Text, full modal) — privileged 정보 포함 시 상한 측정.
    test_loss_priv, test_bce_avg_priv, test_metrics_priv = validate_multitask(
        args, model, dataloader, loss_module, device, accelerator,
        epoch="final",
        disable_cxr=args.disable_cxr,
        disable_txt=False,
        eval_path='priv',
    )

    wandb_test_metrics = {}
    if accelerator.is_main_process and test_metrics:
        # ── Deploy (기존 키, 하위 호환) ──
        wandb_test_metrics = {
            'test/auroc': test_metrics['auroc'],
            'test/auprc': test_metrics['auprc'],
        }
        sm = test_metrics.get('subtype', {})
        if sm:
            wandb_test_metrics.update({
                'test/subtype_auroc_macro': sm.get('auroc_macro', float('nan')),
                'test/subtype_auprc_macro': sm.get('auprc_macro', float('nan')),
                'test/subtype_auroc_mixed': sm.get('auroc_mixed', float('nan')),
                'test/subtype_auroc_ncpe':  sm.get('auroc_ncpe',  float('nan')),
                'test/subtype_auroc_cpe':   sm.get('auroc_cpe',   float('nan')),
                'test/subtype_auprc_mixed': sm.get('auprc_mixed', float('nan')),
                'test/subtype_auprc_ncpe':  sm.get('auprc_ncpe',  float('nan')),
                'test/subtype_auprc_cpe':   sm.get('auprc_cpe',   float('nan')),
                'test/subtype_n_valid':     sm.get('n_valid', 0),
            })

        # ── Priv (full modal) ──
        if test_metrics_priv:
            wandb_test_metrics.update({
                'test_priv/auroc': test_metrics_priv['auroc'],
                'test_priv/auprc': test_metrics_priv['auprc'],
                'test_priv/bce':   test_bce_avg_priv,
                'test/auroc_gap':  test_metrics_priv['auroc'] - test_metrics['auroc'],
            })
            sm_p = test_metrics_priv.get('subtype', {})
            if sm_p:
                wandb_test_metrics.update({
                    'test_priv/subtype_auroc_macro': sm_p.get('auroc_macro', float('nan')),
                    'test_priv/subtype_auprc_macro': sm_p.get('auprc_macro', float('nan')),
                    'test_priv/subtype_auroc_mixed': sm_p.get('auroc_mixed', float('nan')),
                    'test_priv/subtype_auroc_ncpe':  sm_p.get('auroc_ncpe',  float('nan')),
                    'test_priv/subtype_auroc_cpe':   sm_p.get('auroc_cpe',   float('nan')),
                    'test_priv/subtype_auprc_mixed': sm_p.get('auprc_mixed', float('nan')),
                    'test_priv/subtype_auprc_ncpe':  sm_p.get('auprc_ncpe',  float('nan')),
                    'test_priv/subtype_auprc_cpe':   sm_p.get('auprc_cpe',   float('nan')),
                    'test_priv/subtype_n_valid':     sm_p.get('n_valid', 0),
                })

        # ── 콘솔 요약 (deploy vs priv 한눈에 비교) ──
        print("\n" + "="*70)
        print("[Final Test Summary — Deploy vs Priv (full modal)]")
        print("="*70)
        print(f"  Edema   AUROC: deploy={test_metrics['auroc']:.4f}  "
              f"priv={test_metrics_priv['auroc']:.4f}  "
              f"gap={test_metrics_priv['auroc'] - test_metrics['auroc']:+.4f}")
        print(f"  Edema   AUPRC: deploy={test_metrics['auprc']:.4f}  "
              f"priv={test_metrics_priv['auprc']:.4f}")

        sm_p = test_metrics_priv.get('subtype', {})
        nan = float('nan')
        print(f"  Subtype (n={sm.get('n_valid', 0):,} deploy / {sm_p.get('n_valid', 0):,} priv)")
        print(f"    AUROC  macro: deploy={sm.get('auroc_macro', nan):.4f}  priv={sm_p.get('auroc_macro', nan):.4f}")
        for cls in ('mixed', 'ncpe', 'cpe'):
            print(f"           {cls:<5}: deploy={sm.get(f'auroc_{cls}', nan):.4f}  priv={sm_p.get(f'auroc_{cls}', nan):.4f}")
        print(f"    AUPRC  macro: deploy={sm.get('auprc_macro', nan):.4f}  priv={sm_p.get('auprc_macro', nan):.4f}")
        for cls in ('mixed', 'ncpe', 'cpe'):
            print(f"           {cls:<5}: deploy={sm.get(f'auprc_{cls}', nan):.4f}  priv={sm_p.get(f'auprc_{cls}', nan):.4f}")
        print("="*70 + "\n")

    return test_loss, test_bce_avg, test_metrics, wandb_test_metrics


def extract_multitask_outputs(
    args,
    model,
    dataloader,
    loss_module,
    device,
    accelerator,
    save_path,
    disable_cxr=False,
    disable_txt=True,
    eval_path="deploy",
):

    model.eval()
    local_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, total=len(dataloader), desc=f"Extract Outputs [{eval_path}]")
        ):
            disable_txt_local = disable_txt if eval_path == "deploy" else False

            _, _, batch_bce, batch_outputs, batch_counts = train_batch(
                args=args,
                model=model,
                batch=batch,
                loss_module=loss_module,
                device=device,
                accelerator=accelerator,
                disable_cxr=disable_cxr,
                disable_txt=disable_txt_local,
                is_training=False,
                eval_path=eval_path,
            )

            # =====================================================
            # 1. Edema output
            # =====================================================
            edema_logits = batch_outputs["edema_logits"].squeeze(-1)
            edema_probs = torch.sigmoid(edema_logits)
            edema_preds = (edema_probs >= 0.5).long()

            edema_hard = batch_outputs["edema_hard_labels"]
            cxr_anchor = batch_outputs["cxr_anchor_mask"]

            edema_logits_np = edema_logits.detach().cpu().float().numpy().reshape(-1)
            edema_probs_np = edema_probs.detach().cpu().float().numpy().reshape(-1)
            edema_preds_np = edema_preds.detach().cpu().numpy().reshape(-1)
            edema_hard_np = edema_hard.detach().cpu().numpy().reshape(-1)
            cxr_anchor_np = cxr_anchor.detach().cpu().numpy().reshape(-1)

            batch_size = len(edema_probs_np)

            # =====================================================
            # 2. Subtype output
            # =====================================================
            subtype_logits = batch_outputs.get("subtype_logits", None)

            if subtype_logits is not None:
                subtype_probs = torch.softmax(subtype_logits.float(), dim=-1)
                subtype_preds = torch.argmax(subtype_probs, dim=-1)

                subtype_probs_np = subtype_probs.detach().cpu().float().numpy()
                subtype_preds_np = subtype_preds.detach().cpu().numpy().reshape(-1)
                subtype_label_np = batch_outputs["subtype_label"].detach().cpu().numpy().reshape(-1)
                subtype_mask_np = batch_outputs["subtype_mask"].detach().cpu().numpy().reshape(-1)
            else:
                subtype_probs_np = None
                subtype_preds_np = np.full(batch_size, -1)
                subtype_label_np = np.full(batch_size, -1)
                subtype_mask_np = np.zeros(batch_size)

            # =====================================================
            # 3. Metadata 추출
            # Dataset 수정 없이 collate_fn이 반환하는 값 사용
            #
            # batch["stay_ids"]    : 각 sample의 stay_id list
            # batch["time_steps"]  : [B, window_size], window 내 slot_idx들
            # batch["ts_valid_mask"]: [B, window_size], real slot=1, padded slot=0
            #
            # 여기서 저장하는 slot_idx는 window의 마지막 valid slot.
            # Dataset에서 label을 마지막 valid slot 기준으로 만들었기 때문에
            # output이 대응되는 label 시점으로 해석하면 됨.
            # =====================================================
            stay_ids = batch.get("stay_ids", None)
            time_steps = batch.get("time_steps", None)
            ts_valid_mask = batch.get("ts_valid_mask", None)

            if torch.is_tensor(time_steps):
                time_steps_np = time_steps.detach().cpu().numpy()
            else:
                time_steps_np = None

            if torch.is_tensor(ts_valid_mask):
                ts_valid_mask_np = ts_valid_mask.detach().cpu().numpy()
            else:
                ts_valid_mask_np = None

            # =====================================================
            # 4. Row-wise save
            # =====================================================
            for i in range(batch_size):
                # -----------------------------
                # stay_id
                # -----------------------------
                if stay_ids is not None:
                    stay_id_val = stay_ids[i]
                else:
                    stay_id_val = None

                # -----------------------------
                # slot_idx = 마지막 valid slot
                # -----------------------------
                if time_steps_np is not None:
                    if ts_valid_mask_np is not None:
                        valid_positions = np.where(ts_valid_mask_np[i] == 1)[0]

                        if len(valid_positions) > 0:
                            last_valid_pos = valid_positions[-1]
                            slot_idx_val = int(time_steps_np[i, last_valid_pos])
                        else:
                            slot_idx_val = None
                    else:
                        valid_slots = time_steps_np[i][time_steps_np[i] >= 0]
                        slot_idx_val = int(valid_slots[-1]) if len(valid_slots) > 0 else None
                else:
                    slot_idx_val = None

                row = {
                    "stay_id": stay_id_val,
                    "slot_idx": slot_idx_val,

                    "edema_logit": float(edema_logits_np[i]),
                    "edema_prob": float(edema_probs_np[i]),
                    "edema_pred_05": int(edema_preds_np[i]),
                    "edema_hard_label": int(edema_hard_np[i]),
                    "cxr_anchor_mask": int(cxr_anchor_np[i]),

                    "eval_path": eval_path,
                }

                if subtype_probs_np is not None:
                    row.update({
                        "subtype_prob_mixed": float(subtype_probs_np[i, 0]),
                        "subtype_prob_ncpe": float(subtype_probs_np[i, 1]),
                        "subtype_prob_cpe": float(subtype_probs_np[i, 2]),
                        "subtype_pred": int(subtype_preds_np[i]),
                        "subtype_label": int(subtype_label_np[i]),
                        "subtype_mask": int(subtype_mask_np[i]),
                    })
                else:
                    row.update({
                        "subtype_prob_mixed": np.nan,
                        "subtype_prob_ncpe": np.nan,
                        "subtype_prob_cpe": np.nan,
                        "subtype_pred": -1,
                        "subtype_label": -1,
                        "subtype_mask": 0,
                    })

                local_rows.append(row)

    # =========================================================
    # 5. Multi-GPU gather
    # =========================================================
    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        if accelerator.is_main_process:
            gathered_rows = [None] * accelerator.num_processes
            dist.gather_object(local_rows, gathered_rows, dst=0)

            all_rows = []
            for rows in gathered_rows:
                if rows is not None:
                    all_rows.extend(rows)
        else:
            dist.gather_object(local_rows, dst=0)
            all_rows = None

        accelerator.wait_for_everyone()
    else:
        all_rows = local_rows

    # =========================================================
    # 6. Save
    # =========================================================
    if accelerator.is_main_process:
        output_df = pd.DataFrame(all_rows)

        # 보기 좋게 컬럼 순서 고정
        preferred_cols = [
            "stay_id",
            "slot_idx",
            "eval_path",

            "edema_logit",
            "edema_prob",
            "edema_pred_05",
            "edema_hard_label",
            "cxr_anchor_mask",

            "subtype_prob_mixed",
            "subtype_prob_ncpe",
            "subtype_prob_cpe",
            "subtype_pred",
            "subtype_label",
            "subtype_mask",
        ]

        existing_preferred_cols = [c for c in preferred_cols if c in output_df.columns]
        remaining_cols = [c for c in output_df.columns if c not in existing_preferred_cols]
        output_df = output_df[existing_preferred_cols + remaining_cols]

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        output_df.to_csv(save_path, index=False)

        print(f"\n✅ Saved model outputs to: {save_path}")
        print(f"Rows: {len(output_df):,}")
        print(f"Columns: {list(output_df.columns)}")

        return output_df

    return None