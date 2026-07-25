import os
import random
import numpy as np
import wandb
import warnings
import gc
import logging
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)

from training.data_processing import get_dataloaders
from training.engine import train_batch
from training.evaluator import test, validate_multitask
from models.main_architecture import MultiModalEncoder, MultiModalMultiTaskModel
from loss.losses import DualStreamDistillationLoss
from utils.utils import timer, Earlystopping

##################################################################################################
# Model Training Control Center
##################################################################################################
def train_single_stage_multimodal_model(train_df, val_df, test_df, args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
        device_placement=True
    )
    accelerator.replace_sampler = False
    device = accelerator.device
    wandb_on = args.wandb_on

    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"End-to-End MultiModal Multi-Task Training")
        print(f"{'='*80}\n")

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"[GPU Configuration]")
        print(f"   Number of GPUs: {accelerator.num_processes}")
        print(f"   Device: {device}")
        print(f"   Mixed Precision: bf16")
        print(f"{'='*60}\n")

        if wandb_on:
            wandb.init(
                project=args.project_name,
                name=args.wandb_run_name,
                config=vars(args),
                tags=["end_to_end", "multi_task"]
            )

    # DataLoader
    with timer("Dataset Loading"):
        train_loader, val_loader, test_loader, train_sampler = get_dataloaders(train_df, val_df, test_df, args, accelerator)

    # Shared encoder + shared readout 구조 (롤백)
    encoder = MultiModalEncoder(args, disable_cxr=args.disable_cxr, disable_txt=args.disable_txt)
    accelerator.print("[TS Encoder] PatchTSTTSEncoder — pretrained backbone loaded inside encoder")

    model = MultiModalMultiTaskModel(args, encoder=encoder)
    accelerator.print(f"\n[Model] MultiModalMultiTaskModel initialized (shared encoder + shared readout)")
    accelerator.print(f"\n[모달리티 상태] CXR 사용: {not model.encoder.disable_cxr}, "
                      f"TEXT 사용: {not model.encoder.disable_txt}")

    # Loss Module
    loss_module = DualStreamDistillationLoss(args)

    # Optimizer — 5 그룹: backbone(frozen) / ts_pre_proj / text_pre_proj / encoder_rest / heads
    # Shared encoder 구조: 단일 encoder가 BCE_priv + BCE_deploy 둘 다로 학습 (2x gradient)
    backbone_params = list(model.encoder.ts_encoder.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}

    ts_proj_params = list(model.encoder.ts_pre_proj.parameters())
    ts_proj_ids = {id(p) for p in ts_proj_params}

    # text_pre_proj 분리: text가 fusion으로 들어오는 속도를 느리게 → over-reliance 완화
    text_proj_params = list(model.encoder.text_pre_proj.parameters())
    text_proj_ids = {id(p) for p in text_proj_params}

    # shared_readout: priv/deploy 공유 (LN + classifier 모두 공유)
    # + subtype_head: deploy path에 붙은 3-class 보조 head (head_lr 그룹에 포함)
    head_params = (
        list(model.shared_readout.parameters())
        + list(model.subtype_head.parameters())
    )

    encoder_rest_params = [
        p for p in model.encoder.parameters()
        if id(p) not in backbone_ids
        and id(p) not in ts_proj_ids
        and id(p) not in text_proj_ids
        and p.requires_grad   # frozen BERT 등 학습 안 하는 params는 optimizer에서 제외 (weight decay 방지)
    ]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params,     'lr': args.ts_encoder_lr},
        {'params': ts_proj_params,      'lr': args.ts_proj_lr},
        {'params': text_proj_params,    'lr': args.text_proj_lr},
        {'params': encoder_rest_params, 'lr': args.encoder_lr},
        {'params': head_params,         'lr': args.head_lr},
    ], weight_decay=args.weight_decay)
    accelerator.print(
        f"[Optimizer] backbone_lr={args.ts_encoder_lr:.0e} | "
        f"ts_proj_lr={args.ts_proj_lr:.0e} | "
        f"text_proj_lr={args.text_proj_lr:.0e} | "
        f"encoder_lr={args.encoder_lr:.0e} | "
        f"head_lr={args.head_lr:.0e}"
    )

    # ─── [진단] Optimizer 누락 파라미터 점검 ───
    if accelerator.is_main_process:
        optim_ids = set()
        for g in optimizer.param_groups:
            for p in g['params']:
                optim_ids.add(id(p))

        n_total_train = 0
        n_in_optim    = 0
        missing_named = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            n_total_train += p.numel()
            if id(p) in optim_ids:
                n_in_optim += p.numel()
            else:
                missing_named.append((name, p.numel()))

        print(f"\n[Optimizer Coverage 진단]")
        print(f"   Trainable params (model): {n_total_train:,}")
        print(f"   Optimizer가 보유   :     {n_in_optim:,}")
        if missing_named:
            n_missing = n_total_train - n_in_optim
            print(f"   ⚠ 누락: {n_missing:,} params, {len(missing_named)} tensors")
            print(f"   [누락 목록 (상위 20개)]")
            for name, num in sorted(missing_named, key=lambda x: -x[1])[:20]:
                print(f"      - {name}: {num:,}")
        else:
            print(f"   ✓ 모든 trainable params가 optimizer에 포함됨")
        print()

    if accelerator.is_main_process:
        for name, module in model.encoder.named_children():
            n = sum(p.numel() for p in module.parameters())
            nt = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  encoder.{name}: total={n:,}, trainable={nt:,}")
        
    model, optimizer, loss_module, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, loss_module, train_loader, val_loader, test_loader
    )

    num_epochs = args.epochs
    patience = args.patience

    # Scheduler: optional linear warmup → cosine annealing
    warmup_epochs = max(0, int(args.warmup_epochs))
    cosine_epochs = max(1, num_epochs - warmup_epochs)
    cosine = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-6
    )
    if warmup_epochs > 0:
        warmup = lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        accelerator.print(f"[Scheduler] LinearWarmup({warmup_epochs}ep, start_factor={args.warmup_start_factor}) "
                          f"→ CosineAnnealingLR({cosine_epochs}ep, eta_min=1e-6)")
    else:
        scheduler = cosine
        accelerator.print(f"[Scheduler] CosineAnnealingLR({cosine_epochs}ep, eta_min=1e-6) — no warmup")

    single_best_model_path = os.path.join(args.best_model_dir, "multitask_best_model.pth")
    early_stopper = Earlystopping(
        patience=patience,
        start_epoch=0,
        save_path=single_best_model_path,
        experiment_id=args.experiment_id,
        mode='max'
    )

    if accelerator.is_main_process:
        print(f"\n[Early Stopping] Monitoring: AUROC (mode=max, patience={patience})")

    best_val_auroc, best_val_auprc = 0.0, 0.0
    stop_flag = torch.zeros(1, dtype=torch.uint8, device=device)
    local_rank = accelerator.local_process_index

    # ==================== Training Loop ====================
    for epoch in tqdm(range(num_epochs), total=num_epochs,
            desc=f"[Rank {local_rank}] End-to-End MultiTask Training",
            position=local_rank, leave=True, dynamic_ncols=True
        ):

        # LUPI loss accumulators
        bce_priv_sum   = torch.zeros(1, device=device, dtype=torch.float32)
        bce_deploy_sum = torch.zeros(1, device=device, dtype=torch.float32)
        fd_sum         = torch.zeros(1, device=device, dtype=torch.float32)
        rd_sum         = torch.zeros(1, device=device, dtype=torch.float32)
        rd_cos_sum     = torch.zeros(1, device=device, dtype=torch.float32)
        rd_mse_sum     = torch.zeros(1, device=device, dtype=torch.float32)
        kd_sum         = torch.zeros(1, device=device, dtype=torch.float32)
        cov_sum        = torch.zeros(1, device=device, dtype=torch.float32)
        subtype_sum    = torch.zeros(1, device=device, dtype=torch.float32)
        subtype_count  = torch.zeros(1, device=device, dtype=torch.float32)
        bce_count      = torch.zeros(1, device=device, dtype=torch.float32)
        batch_total    = torch.zeros(1, device=device, dtype=torch.float32)

        # Modality Dropout 통계 (priv path에서 text가 차단된 batch 수)
        priv_drop_count = torch.zeros(1, device=device, dtype=torch.float32)

        # privileged vs deploy 차이 진단
        diag_fused_diff_sum = torch.zeros(1, device=device, dtype=torch.float32)
        diag_logit_diff_sum = torch.zeros(1, device=device, dtype=torch.float32)

        train_p_pos_list = []
        train_edema_hard_list = []
        train_cxr_anchor_list = []

        optimizer.zero_grad(set_to_none=True)
        train_sampler.set_epoch(epoch)

        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch+1}] Learning Rate: {current_lr:.2e}")

        # ── 진단: 매 에포크 첫 배치에서 collapse 통계 1회 캡처 ──
        diag_captured_this_epoch = False
        epoch_diag_deploy = {}
        epoch_diag_priv   = {}

        # Training
        for batch_idx, batch in enumerate(tqdm(train_loader, total=len(train_loader),
                                                desc=f"[Rank {local_rank}] Epoch {epoch+1}/{num_epochs}",
                                                position=local_rank, leave=True, dynamic_ncols=True)):

            # 첫 배치 전에 diag flag 켜기 (main process만, model unwrap 필요)
            is_diag_batch = accelerator.is_main_process and not diag_captured_this_epoch
            if is_diag_batch:
                unwrap_model = accelerator.unwrap_model(model)
                unwrap_model.diag_enabled = True

            # ── Modality Dropout on teacher: 확률 p로 priv path도 text 차단 ──
            disable_txt_priv = (
                (not is_diag_batch)
                and args.priv_text_dropout_prob > 0.0
                and random.random() < args.priv_text_dropout_prob
            )
            if disable_txt_priv:
                priv_drop_count += 1.0

            with accelerator.accumulate(model):
                total_batch_loss, batch_bce_priv, batch_bce_deploy, batch_outputs, batch_counts = train_batch(
                    args=args,
                    model=model,
                    batch=batch,
                    loss_module=loss_module,
                    device=accelerator.device,
                    accelerator=accelerator,
                    disable_cxr=args.disable_cxr,
                    disable_txt=args.disable_txt,
                    is_training=True,
                    disable_txt_priv=disable_txt_priv,
                )

                accelerator.backward(total_batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # 첫 배치 forward 끝나면 diag 결과 수확
            if accelerator.is_main_process and not diag_captured_this_epoch:
                unwrap_model.diag_enabled = False
                epoch_diag_deploy = dict(getattr(unwrap_model, 'diag_deploy', {}) or {})
                epoch_diag_priv   = dict(getattr(unwrap_model, 'diag_priv',   {}) or {})
                diag_captured_this_epoch = True

            bce_ct_local = torch.as_tensor(batch_counts['bce_count'], device=device, dtype=torch.float32)
            bce_priv_sum   += torch.as_tensor(batch_bce_priv,   device=device, dtype=torch.float32) * bce_ct_local
            bce_deploy_sum += torch.as_tensor(batch_bce_deploy, device=device, dtype=torch.float32) * bce_ct_local
            fd_sum         += torch.as_tensor(batch_counts['fd_loss'], device=device, dtype=torch.float32)
            rd_sum         += torch.as_tensor(batch_counts['rd_loss'], device=device, dtype=torch.float32)
            rd_cos_sum     += torch.as_tensor(batch_counts['rd_cos'],  device=device, dtype=torch.float32)
            rd_mse_sum     += torch.as_tensor(batch_counts['rd_mse'],  device=device, dtype=torch.float32)
            kd_sum         += torch.as_tensor(batch_counts['kd_loss'], device=device, dtype=torch.float32)
            cov_sum        += torch.as_tensor(batch_counts['cov_loss'], device=device, dtype=torch.float32)
            # Subtype loss is already averaged over the masked-in samples per batch;
            # weight per batch by its valid-sample count for an unbiased epoch-level mean.
            sub_ct_local = torch.as_tensor(batch_counts['subtype_count'], device=device, dtype=torch.float32)
            subtype_sum    += torch.as_tensor(batch_counts['subtype_loss'], device=device, dtype=torch.float32) * sub_ct_local
            subtype_count  += sub_ct_local
            bce_count      += bce_ct_local
            batch_total    += 1.0

            diag_fused_diff_sum += torch.as_tensor(batch_counts['diag_fused_diff'], device=device, dtype=torch.float32)
            diag_logit_diff_sum += torch.as_tensor(batch_counts['diag_logit_diff'], device=device, dtype=torch.float32)

            with torch.no_grad():
                edema_logits = batch_outputs['edema_logits'].squeeze(-1)
                edema_hard = batch_outputs['edema_hard_labels']
                cxr_anchor = batch_outputs['cxr_anchor_mask']

                p_pos = torch.sigmoid(edema_logits)

                train_p_pos_list.append(p_pos.detach().cpu())
                train_edema_hard_list.append(edema_hard.detach().cpu())
                train_cxr_anchor_list.append(cxr_anchor.detach().cpu())

        # Metric aggregation
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bce_priv_sum,   op=dist.ReduceOp.SUM)
            dist.all_reduce(bce_deploy_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(fd_sum,         op=dist.ReduceOp.SUM)
            dist.all_reduce(rd_sum,         op=dist.ReduceOp.SUM)
            dist.all_reduce(rd_cos_sum,     op=dist.ReduceOp.SUM)
            dist.all_reduce(rd_mse_sum,     op=dist.ReduceOp.SUM)
            dist.all_reduce(kd_sum,         op=dist.ReduceOp.SUM)
            dist.all_reduce(cov_sum,        op=dist.ReduceOp.SUM)
            dist.all_reduce(subtype_sum,    op=dist.ReduceOp.SUM)
            dist.all_reduce(subtype_count,  op=dist.ReduceOp.SUM)
            dist.all_reduce(bce_count,      op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_total,    op=dist.ReduceOp.SUM)
            dist.all_reduce(priv_drop_count,    op=dist.ReduceOp.SUM)
            dist.all_reduce(diag_fused_diff_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(diag_logit_diff_sum, op=dist.ReduceOp.SUM)

        bce_priv_avg   = (bce_priv_sum   / (bce_count   + 1e-8)).item()
        bce_deploy_avg = (bce_deploy_sum / (bce_count   + 1e-8)).item()
        fd_avg         = (fd_sum         / (batch_total + 1e-8)).item()
        rd_avg         = (rd_sum         / (batch_total + 1e-8)).item()
        rd_cos_avg     = (rd_cos_sum     / (batch_total + 1e-8)).item()
        rd_mse_avg     = (rd_mse_sum     / (batch_total + 1e-8)).item()
        kd_avg         = (kd_sum         / (batch_total + 1e-8)).item()
        cov_avg        = (cov_sum        / (batch_total + 1e-8)).item()
        subtype_avg    = (subtype_sum    / (subtype_count + 1e-8)).item()
        priv_drop_rate = (priv_drop_count / (batch_total + 1e-8)).item()
        diag_fused_diff_avg = (diag_fused_diff_sum / (batch_total + 1e-8)).item()
        diag_logit_diff_avg = (diag_logit_diff_sum / (batch_total + 1e-8)).item()
        avg_total_loss = (
            bce_priv_avg + bce_deploy_avg
            + args.lupi_fd_weight  * fd_avg
            + args.lupi_rd_weight  * rd_avg
            + args.lupi_kd_weight  * kd_avg
            + args.lupi_cov_weight * cov_avg
            + args.subtype_weight  * subtype_avg
        )

        # Gather train predictions from all GPUs
        if accelerator.num_processes > 1:
            local_preds = {
                'p_pos': [p.cpu() for p in train_p_pos_list],
                'edema_hard': [e.cpu() for e in train_edema_hard_list],
                'cxr_anchor': [c.cpu() for c in train_cxr_anchor_list],
            }

            if accelerator.is_main_process:
                gathered_preds = [None] * accelerator.num_processes
                dist.gather_object(local_preds, gathered_preds, dst=0)

                all_p_pos, all_edema_hard, all_cxr_anchor = [], [], []
                for gpu_preds in gathered_preds:
                    all_p_pos.extend(gpu_preds['p_pos'])
                    all_edema_hard.extend(gpu_preds['edema_hard'])
                    all_cxr_anchor.extend(gpu_preds['cxr_anchor'])

                p_pos_all = torch.cat(all_p_pos, dim=0).numpy() if all_p_pos else np.array([])
                edema_hard_all = torch.cat(all_edema_hard, dim=0).numpy() if all_edema_hard else np.array([])
                cxr_anchor_all = torch.cat(all_cxr_anchor, dim=0).numpy() if all_cxr_anchor else np.array([])
            else:
                dist.gather_object(local_preds, dst=0)
                p_pos_all = None

            accelerator.wait_for_everyone()

        else:
            if len(train_p_pos_list) > 0:
                p_pos_all = torch.cat(train_p_pos_list, dim=0).numpy()
                edema_hard_all = torch.cat(train_edema_hard_list, dim=0).numpy()
                cxr_anchor_all = torch.cat(train_cxr_anchor_list, dim=0).numpy()
            else:
                p_pos_all = None

        # Train metrics — cxr_flag==1 & edema_hard ∈ {0, 1}
        train_metrics = {}
        if accelerator.is_main_process and p_pos_all is not None and len(p_pos_all) > 0:

            mask_a = (cxr_anchor_all == 1) & ((edema_hard_all == 0) | (edema_hard_all == 1))
            if mask_a.sum() >= 2 and len(np.unique(edema_hard_all[mask_a])) >= 2:
                y = edema_hard_all[mask_a].astype(int)
                p = p_pos_all[mask_a]
                train_metrics['auroc'] = roc_auc_score(y, p)
                train_metrics['auprc'] = average_precision_score(y, p)
            else:
                train_metrics['auroc'] = float('nan')
                train_metrics['auprc'] = float('nan')

        # Performance Output
        if accelerator.is_main_process:
            print(f"\n✅ Epoch {epoch+1} - Train Total Loss: {avg_total_loss:.4f}")
            print(f"   [KD Loss Components]")
            print(f"      BCE_priv   (TS+CXR+Text):  {bce_priv_avg:.4f}")
            print(f"      BCE_deploy (TS+CXR    ):  {bce_deploy_avg:.4f}")
            print(f"      Feature KD (fused level)    : {fd_avg:.4f} (λ={args.lupi_fd_weight})")
            print(f"      Readout KD (post-readout)   : {rd_avg:.4f} (λ={args.lupi_rd_weight})")
            print(f"         └─ rd_cos: {rd_cos_avg:.4f}  |  rd_smooth_l1: {rd_mse_avg:.4f}")
            print(f"      Logit   KD (final logit)    : {kd_avg:.4f} (λ={args.lupi_kd_weight})")
            print(f"      Covariance Reg              : {cov_avg:.4f} (λ={args.lupi_cov_weight})")
            print(f"      Subtype Soft-CE (masked)    : {subtype_avg:.4f} (λ={args.subtype_weight}, valid={int(subtype_count.item()):,})")
            print(f"      [Modality Dropout] priv text dropout rate: {priv_drop_rate:.3f} (p={args.priv_text_dropout_prob})")
            print(f"      [Δ] priv - deploy BCE:    {bce_deploy_avg - bce_priv_avg:+.4f}")
            print(f"   [Path Difference Diagnostic]")
            print(f"      |fused_priv - fused_deploy|:  {diag_fused_diff_avg:.6f}")
            print(f"      |logit_priv - logit_deploy|:  {diag_logit_diff_avg:.6f}")
            print(f"      |covariance_loss|:  {cov_avg:.6f}")

            if train_metrics:
                print("\n   [Train AUROC — deploy path] (cxr_flag==1)")
                print(f"   AUROC={train_metrics['auroc']:.4f}  "
                      f"AUPRC={train_metrics['auprc']:.4f}")

            # ── Collapse 진단 출력 (epoch 첫 배치 샘플) ──
            if epoch_diag_deploy or epoch_diag_priv:
                def _fmt(d, k):
                    v = d.get(k, float('nan'))
                    return f"{v:.4f}" if v == v else "  nan"  # nan check via NaN != NaN
                print("\n   [Patch/Attn Collapse 진단 — inter-patch cosine (1=collapse, 0=다양)]")
                print(f"      TS post-PatchTST       : {_fmt(epoch_diag_deploy, 'ts_postpatch_inter_patch_cos')}")
                print(f"      TS post-projection     : {_fmt(epoch_diag_deploy, 'ts_proj_inter_patch_cos')}")
                print(f"      TS +time2vec           : {_fmt(epoch_diag_deploy, 'ts_with_time_inter_patch_cos')}")
                print(f"      TS after self-attn     : {_fmt(epoch_diag_deploy, 'ts_after_sa_inter_patch_cos')}")
                print(f"      [deploy] TS final query: {_fmt(epoch_diag_deploy, 'ts_final_inter_patch_cos')}  "
                      f"img-attn={_fmt(epoch_diag_deploy, 'img_attn_inter_patch_cos')}")
                print(f"      [priv]   TS final query: {_fmt(epoch_diag_priv,   'ts_final_inter_patch_cos')}  "
                      f"img-attn={_fmt(epoch_diag_priv,   'img_attn_inter_patch_cos')}  "
                      f"text-attn={_fmt(epoch_diag_priv,  'text_attn_inter_patch_cos')}")

        gc.collect()
        torch.cuda.empty_cache()

        # ==================== Validation (deploy path — text 영구 제외) ====================
        val_loss, val_bce_avg, val_metrics = validate_multitask(
            args=args,
            model=model,
            dataloader=val_loader,
            loss_module=loss_module,
            device=accelerator.device,
            accelerator=accelerator,
            epoch=epoch+1,
            disable_cxr=args.disable_cxr,
            disable_txt=True,
            eval_path='deploy',
        )

        # ==================== Validation (priv path — text 포함) ====================
        val_loss_priv, val_bce_avg_priv, val_metrics_priv = validate_multitask(
            args=args,
            model=model,
            dataloader=val_loader,
            loss_module=loss_module,
            device=accelerator.device,
            accelerator=accelerator,
            epoch=epoch+1,
            disable_cxr=args.disable_cxr,
            disable_txt=False,
            eval_path='priv',
        )

        # LR scheduler step (warmup -> cosine)
        scheduler.step()

        if accelerator.is_main_process:
            phase = "Warmup" if epoch < warmup_epochs else "Cosine"
            lrs = " | ".join(f"g{i}={pg['lr']:.2e}" for i, pg in enumerate(optimizer.param_groups))
            print(f"   [LR/{phase}] {lrs}")

        # ==================== Early Stopping ====================
        if accelerator.is_main_process and val_metrics:
            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']

            if val_metrics['auprc'] > best_val_auprc:
                best_val_auprc = val_metrics['auprc']

        # ==================== WandB Logging ====================
        if accelerator.is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "train/total_loss":      avg_total_loss,
                "train/bce_priv":        bce_priv_avg,
                "train/bce_deploy":      bce_deploy_avg,
                "train/fd_loss":         fd_avg,
                "train/rd_loss":         rd_avg,
                "train/rd_cos":          rd_cos_avg,
                "train/rd_smooth_l1":    rd_mse_avg,
                "train/kd_loss":         kd_avg,
                "train/cov_loss":        cov_avg,
                "train/subtype_loss":    subtype_avg,
                "train/subtype_valid":   int(subtype_count.item()),
                "train/priv_drop_rate":  priv_drop_rate,
                "train/bce_gap":         bce_deploy_avg - bce_priv_avg,

                "val/total_loss": val_loss,
                "val/bce_loss":   val_bce_avg,

                "val_deploy/auroc": val_metrics['auroc'],
                "val_deploy/auprc": val_metrics['auprc'],

                "val_priv/auroc":   val_metrics_priv['auroc'],
                "val_priv/auprc":   val_metrics_priv['auprc'],
                
                "val_priv/bce":     val_bce_avg_priv,

                "val/auroc_gap":    val_metrics_priv['auroc'] - val_metrics['auroc'],
            }

            if train_metrics:
                log_dict.update({
                    "train/auroc": train_metrics['auroc'],
                    "train/auprc": train_metrics['auprc'],
                })

            # ── Subtype OvR metrics (deploy path eval) ──
            sub_deploy = val_metrics.get('subtype', {}) if val_metrics else {}
            sub_priv   = val_metrics_priv.get('subtype', {}) if val_metrics_priv else {}
            log_dict.update({
                "val_deploy/subtype_auroc_macro": sub_deploy.get('auroc_macro', float('nan')),
                "val_deploy/subtype_auprc_macro": sub_deploy.get('auprc_macro', float('nan')),
                "val_deploy/subtype_auroc_mixed": sub_deploy.get('auroc_mixed', float('nan')),
                "val_deploy/subtype_auroc_ncpe":  sub_deploy.get('auroc_ncpe',  float('nan')),
                "val_deploy/subtype_auroc_cpe":   sub_deploy.get('auroc_cpe',   float('nan')),
                "val_deploy/subtype_auprc_mixed": sub_deploy.get('auprc_mixed', float('nan')),
                "val_deploy/subtype_auprc_ncpe":  sub_deploy.get('auprc_ncpe',  float('nan')),
                "val_deploy/subtype_auprc_cpe":   sub_deploy.get('auprc_cpe',   float('nan')),
                "val_deploy/subtype_n_valid":     sub_deploy.get('n_valid', 0),

                "val_priv/subtype_auroc_macro":   sub_priv.get('auroc_macro', float('nan')),
                "val_priv/subtype_auprc_macro":   sub_priv.get('auprc_macro', float('nan')),
            })

            # ── Collapse 진단 wandb 로깅 ──
            for k, v in epoch_diag_deploy.items():
                log_dict[f"diag_deploy/{k}"] = v
            for k, v in epoch_diag_priv.items():
                log_dict[f"diag_priv/{k}"] = v

            if wandb_on:
                wandb.log(log_dict)

            if early_stopper(args, val_metrics['auroc'], model, epoch, accelerator):
                stop_flag.fill_(1)
                print(f"Early stopping triggered at epoch {epoch+1}")

        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        gc.collect()

        if dist.is_initialized() and dist.get_world_size() > 1:
            gathered_stop_flag = accelerator.gather_for_metrics(stop_flag)
            stop_flag.fill_(gathered_stop_flag.max().item())

        if stop_flag.item() == 1:
            break

    accelerator.wait_for_everyone()

    # ==================== Load Best Model ====================
    actual_best_model_path = early_stopper.get_best_model_path()
    if actual_best_model_path and os.path.exists(actual_best_model_path):
        accelerator.print(f"✅ Loading best model from: {actual_best_model_path}")
        checkpoint = torch.load(actual_best_model_path, map_location=accelerator.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(state_dict)

    else:
        accelerator.print(f"⚠️ Best model not found, using current model state")

    accelerator.wait_for_everyone()

    # ==================== Test ====================
    test_loss, _, _, wandb_test_metrics = test(
        args=args,
        model=model,
        dataloader=test_loader,
        loss_module=loss_module,
        device=accelerator.device,
        accelerator=accelerator,
    )

    # ==================== Final Metrics ====================
    if wandb_on:
        if accelerator.is_main_process and wandb_test_metrics:
            wandb.run.summary.update({
                'final_test/total_loss': test_loss,
                'final_test/auroc': wandb_test_metrics['test/auroc'],
                'final_test/auprc': wandb_test_metrics['test/auprc'],
            })

    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("✅ MULTI-TASK TRAINING COMPLETED!")
        print(f"   Best Val AUROC (Edema Detection): {best_val_auroc:.4f}")
        print(f"   Best Val AUPRC (Edema Detection): {best_val_auprc:.4f}\n")

        if wandb_test_metrics:
            print("   [Test Results - Clinical Diagnostic (cxr_flag==1)]")
            print(f"   AUROC={wandb_test_metrics['test/auroc']:.4f}  "
                  f"AUPRC={wandb_test_metrics['test/auprc']:.4f}")
        print("="*80 + "\n")

    results = {}
    if accelerator.is_main_process and wandb_test_metrics:
        results = {
            'val_auroc': best_val_auroc,
            'val_auprc': best_val_auprc,
            'test_auroc': wandb_test_metrics['test/auroc'],
            'test_auprc': wandb_test_metrics['test/auprc'],
        }

    if dist.is_initialized():
        dist.destroy_process_group()

    return results