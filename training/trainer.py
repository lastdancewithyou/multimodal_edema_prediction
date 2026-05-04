import os
import numpy as np
import wandb
import warnings
import gc
import logging
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention*")
logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)

from training.data_processing import get_dataloaders
from training.engine import train_batch
from training.evaluator import test, validate_multitask
from models.main_architecture import MultiModalEncoder, MultiModalMultiTaskModel
from loss.losses import MultiModalLoss
from utils.utils import timer, plot_latent_time_attention, Earlystopping, count_params
from analysis.umap_multitask import plot_multitask_umap


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

    # Create model
    encoder = MultiModalEncoder(args, disable_cxr=args.disable_cxr, disable_txt=args.disable_txt, disable_prompt=args.disable_prompt)
    ssl_save_path = "./best_ssl_model_mask_prob=0.15_noIOcols.pt"
    checkpoint = torch.load(ssl_save_path, map_location=device)
    ssl_state_dict = checkpoint

    pretrained_ts_weights = {}

    for key, value in ssl_state_dict.items():
        if key.startswith('encoder.'):
            new_key = key[8:]
            if 'pos_encoder.pe' in key:
                pretrained_ts_weights[new_key] = value[:, :args.window_size, :]
            else:
                pretrained_ts_weights[new_key] = value

    encoder.ts_encoder.load_state_dict(pretrained_ts_weights, strict=True)
    accelerator.print(f"SSL TS 인코더 가중치 이식 완료")

    model = MultiModalMultiTaskModel(args, encoder)
    accelerator.print(f"\n[Model] MultiModalMultiTaskModel initialized")

    accelerator.print(f"\n[모달리티 상태] CXR 사용: {not model.encoder.disable_cxr}, TEXT 사용: {not model.encoder.disable_txt}, PROMPT 사용: {not model.encoder.disable_prompt}")

    # Loss Module
    loss_module = MultiModalLoss(args)

    # Optimizer: separate LR for encoder vs task heads
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': list(model.edema_readout.parameters()) + list(model.subtype_readout.parameters()), 'lr': args.head_lr},
    ], weight_decay=args.weight_decay)
    accelerator.print(f"[Optimizer] encoder_lr={args.encoder_lr:.0e}, head_lr={args.head_lr:.0e}")

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'─'*80}")
        print(f"OVERALL MODEL STATISTICS")
        print(f"{'─'*80}")
        print(f"   Total Parameters:       {total_params:>15,}")
        print(f"   Trainable Parameters:   {trainable_params:>15,}")
        print(f"   Frozen Parameters:      {total_params - trainable_params:>15,}")
        print(f"   Trainable Ratio:        {100 * trainable_params / total_params:>14.1f}%")
        print(f"{'='*80}\n")

    # Prepare model, optimizer, loss_module, dataloaders with Accelerator
    model, optimizer, loss_module, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, loss_module, train_loader, val_loader, test_loader
    )

    num_epochs = args.epochs
    patience = args.patience

    # Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

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

        bce_sum = torch.zeros(1, device=device, dtype=torch.float32)
        bce_count = torch.zeros(1, device=device, dtype=torch.float32)
        ce_sum = torch.zeros(1, device=device, dtype=torch.float32)
        ce_count = torch.zeros(1, device=device, dtype=torch.float32)
        align_sum = 0.0
        align_count = 0

        train_edema_preds_list = []
        train_edema_labels_list = []
        train_subtype_preds_list = []
        train_subtype_labels_list = []

        optimizer.zero_grad(set_to_none=True)
        train_sampler.set_epoch(epoch)

        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch+1}] Learning Rate: {current_lr:.2e}")

        # Training
        for batch_idx, batch in enumerate(tqdm(train_loader, total=len(train_loader),
                                                desc=f"[Rank {local_rank}] Epoch {epoch+1}/{num_epochs}",
                                                position=local_rank, leave=True, dynamic_ncols=True)):

            with accelerator.accumulate(model):
                total_batch_loss, batch_bce, batch_ce, batch_outputs, batch_counts = train_batch(
                    args=args,
                    model=model,
                    batch=batch,
                    loss_module=loss_module,
                    device=accelerator.device,
                    accelerator=accelerator,
                    dataset=train_loader.dataset,
                    max_length=args.token_max_length,
                    disable_cxr=args.disable_cxr,
                    disable_txt=args.disable_txt,
                    bce_weight=args.bce_weight,
                    ce_weight=args.ce_weight,
                    current_epoch=epoch,
                )

                accelerator.backward(total_batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bce_ct_local = torch.as_tensor(batch_counts['bce_count'], device=device, dtype=torch.float32)
            ce_ct_local = torch.as_tensor(batch_counts['ce_count'], device=device, dtype=torch.float32)

            bce_sum += torch.as_tensor(batch_bce, device=device, dtype=torch.float32) * bce_ct_local
            ce_sum += torch.as_tensor(batch_ce, device=device, dtype=torch.float32) * ce_ct_local

            bce_count += bce_ct_local
            ce_count += ce_ct_local

            batch_align = batch_outputs['align_loss']
            if batch_align > 0:
                align_sum += batch_align
                align_count += 1

            with torch.no_grad():
                edema_logits = batch_outputs['edema_logits'].squeeze(-1)
                subtype_logits = batch_outputs['subtype_logits']

                edema_labels = batch_outputs['edema_labels']
                subtype_labels = batch_outputs['subtype_labels']

                p_pos = torch.sigmoid(edema_logits)
                p_sub = torch.softmax(subtype_logits, dim=-1)

                train_edema_preds_list.append(p_pos.detach().cpu())
                train_subtype_preds_list.append(p_sub.detach().cpu())
                train_edema_labels_list.append(edema_labels.detach().cpu())
                train_subtype_labels_list.append(subtype_labels.detach().cpu())

        # Metric aggregation
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bce_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(bce_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(ce_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ce_count, op=dist.ReduceOp.SUM)

        bce_avg = (bce_sum / (bce_count + 1e-8)).item()
        ce_avg = (ce_sum / (ce_count + 1e-8)).item()
        align_avg = align_sum / align_count if align_count > 0 else 0.0

        bce_contrib = args.bce_weight * bce_avg
        ce_contrib = args.ce_weight * ce_avg
        align_contrib = args.align_weight * align_avg
        avg_total_loss = bce_contrib + ce_contrib + align_contrib

        # Gather train predictions from all GPUs
        if accelerator.num_processes > 1:
            local_preds = {
                'p_pos': [p.cpu() for p in train_edema_preds_list],
                'p_sub': [p.cpu() for p in train_subtype_preds_list],
                'edema': [e.cpu() for e in train_edema_labels_list],
                'subtype': [s.cpu() for s in train_subtype_labels_list]
            }

            if accelerator.is_main_process:
                gathered_preds = [None] * accelerator.num_processes
                dist.gather_object(local_preds, gathered_preds, dst=0)

                all_p_pos, all_p_sub, all_edema, all_subtype = [], [], [], []
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

            accelerator.wait_for_everyone()

        else:
            if len(train_edema_preds_list) > 0:
                p_pos_all = torch.cat(train_edema_preds_list, dim=0).numpy()
                p_sub_all = torch.cat(train_subtype_preds_list, dim=0).numpy()
                edema_all = torch.cat(train_edema_labels_list, dim=0).numpy()
                subtype_all = torch.cat(train_subtype_labels_list, dim=0).numpy()
            else:
                p_pos_all = None

        # Train metrics
        train_metrics = {}
        if accelerator.is_main_process and p_pos_all is not None and len(p_pos_all) > 0:

            mask_l1 = (edema_all == 0) | (edema_all == 1)
            y_l1 = edema_all[mask_l1].astype(int)
            p_l1 = p_pos_all[mask_l1]

            if mask_l1.sum() >= 2 and len(np.unique(y_l1)) >= 2:
                train_metrics['level1_auroc'] = roc_auc_score(y_l1, p_l1)
                train_metrics['level1_auprc'] = average_precision_score(y_l1, p_l1)
            else:
                train_metrics['level1_auroc'] = float('nan')
                train_metrics['level1_auprc'] = float('nan')

            mask_l2 = (edema_all == 1) & ((subtype_all == 0) | (subtype_all == 1) | (subtype_all == 2))

            if mask_l2.sum() >= 2 and len(np.unique(subtype_all[mask_l2])) >= 2:
                y_l2 = subtype_all[mask_l2].astype(int)
                y_l2_bin = label_binarize(y_l2, classes=[0, 1, 2])
                p_l2_probs = p_sub_all[mask_l2, :]

                train_metrics['level2_auroc'] = roc_auc_score(y_l2_bin, p_l2_probs, average='macro', multi_class='ovr')
                train_metrics['level2_auprc'] = average_precision_score(y_l2_bin, p_l2_probs, average='macro')
            else:
                train_metrics['level2_auroc'] = float('nan')
                train_metrics['level2_auprc'] = float('nan')

        # Performance Output
        if accelerator.is_main_process:
            print(f"\n✅ Epoch {epoch+1} - Train Total Loss: {avg_total_loss:.4f}")
            print(f"   [Loss Components]")
            print(f"      BCE (Edema): {bce_avg:.4f} → Weighted: {bce_contrib:.4f} (λ={args.bce_weight})")
            print(f"      CE (Subtype): {ce_avg:.4f} → Weighted: {ce_contrib:.4f} (λ={args.ce_weight})")
            print(f"      Align (Virtual Prompt): {align_avg:.4f} → Weighted: {align_contrib:.4f} (λ={args.align_weight}, n={align_count})")

            if train_metrics:
                print(f"\n   [Hierarchical Performance Metrics]")
                print(f"[Level 1: Edema Detection]    AUROC={train_metrics['level1_auroc']:.4f}  "
                    f"AUPRC={train_metrics['level1_auprc']:.4f}")
                print(f"[Level 2: Subtype (3-way)]    AUROC={train_metrics['level2_auroc']:.4f}  "
                    f"AUPRC={train_metrics['level2_auprc']:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

        # ==================== Validation ====================
        val_loss, val_bce_avg, val_ce_avg, val_metrics = validate_multitask(
            args=args,
            model=model,
            dataloader=val_loader,
            loss_module=loss_module,
            device=accelerator.device,
            accelerator=accelerator,
            dataset=val_loader.dataset,
            epoch=epoch+1,
            disable_cxr=args.disable_cxr,
            disable_txt=args.disable_txt,
            max_length=args.token_max_length,
        )

        # CosineAnnealingLR scheduler step
        scheduler.step()
        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   CosineAnnealingLR - Current LR: {current_lr:.2e}")

        # ==================== Early Stopping ====================
        if accelerator.is_main_process and val_metrics:
            if val_metrics['level1_auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['level1_auroc']

            if val_metrics['level1_auprc'] > best_val_auprc:
                best_val_auprc = val_metrics['level1_auprc']

        # ==================== Multi-Task UMAP Visualization ====================
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and ((epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs):
            print("Generating Training UMAP...")
            train_umap_dir = os.path.join(args.umap_save_dir, 'train')
            train_reducers = plot_multitask_umap(
                args=args,
                model=model,
                dataloader=train_loader,
                dataset=train_loader.dataset,
                epoch=epoch+1,
                save_dir=train_umap_dir,
                max_samples=10000,
                umap_reducers=None
            )
            print("Training UMAP completed!")

            print("Generating Validation UMAP...")
            val_umap_dir = os.path.join(args.umap_save_dir, 'val')
            plot_multitask_umap(
                args=args,
                model=model,
                dataloader=val_loader,
                dataset=val_loader.dataset,
                epoch=epoch+1,
                save_dir=val_umap_dir,
                max_samples=None,
                umap_reducers=train_reducers
            )
            print("Validation UMAP completed!")
        accelerator.wait_for_everyone()

        # ==================== WandB Logging ====================
        if accelerator.is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "train/total_loss": avg_total_loss,
                "train/bce_loss": bce_avg,
                "train/ce_loss": ce_avg,
                "train/align_loss": align_avg,

                "val/total_loss": val_loss,
                "val/bce_loss": val_bce_avg,
                "val/ce_loss": val_ce_avg,

                "val/level1_auroc": val_metrics['level1_auroc'],
                "val/level1_auprc": val_metrics['level1_auprc'],
                "val/level2_auroc": val_metrics['level2_auroc'],
                "val/level2_auprc": val_metrics['level2_auprc'],
            }

            if train_metrics:
                log_dict.update({
                    "train/level1_auroc": train_metrics['level1_auroc'],
                    "train/level1_auprc": train_metrics['level1_auprc'],
                    "train/level2_auroc": train_metrics['level2_auroc'],
                    "train/level2_auprc": train_metrics['level2_auprc'],
                })

            if wandb_on:
                wandb.log(log_dict)

            if early_stopper(args, val_metrics['level1_auroc'], model, epoch, accelerator):
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

    # ==================== Test ====================
    test_loss, _, _, _, wandb_test_metrics = test(
        args=args,
        model=model,
        dataloader=test_loader,
        loss_module=loss_module,
        device=accelerator.device,
        accelerator=accelerator,
        dataset=test_loader.dataset
    )

    # ==================== Final Metrics and UMAP ====================
    if wandb_on:
        if accelerator.is_main_process and wandb_test_metrics:
            wandb.run.summary.update({
                'final_test/total_loss': test_loss,
                'final_test/level1_auroc': wandb_test_metrics['test/level1_auroc'],
                'final_test/level1_auprc': wandb_test_metrics['test/level1_auprc'],
                'final_test/level2_auroc': wandb_test_metrics['test/level2_auroc'],
                'final_test/level2_auprc': wandb_test_metrics['test/level2_auprc'],
            })

    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("Generating Test UMAP...")
        test_umap_dir = os.path.join(args.umap_save_dir, 'test')
        plot_multitask_umap(
            args=args,
            model=model,
            dataloader=test_loader,
            dataset=test_loader.dataset,
            epoch=epoch+1,
            save_dir=test_umap_dir,
            max_samples=None,
            umap_reducers=train_reducers
        )
        print("Test UMAP completed!")

    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("✅ MULTI-TASK TRAINING COMPLETED!")
        print(f"   Best Val Level 1 AUROC (Edema Detection): {best_val_auroc:.4f}")
        print(f"   Best Val Level 1 AUPRC (Edema Detection): {best_val_auprc:.4f}\n")

        if wandb_test_metrics:
            print("   [Test Results - Hierarchical Metrics]")
            print(f"   Level 1 (Edema Detection):     AUROC={wandb_test_metrics['test/level1_auroc']:.4f}  AUPRC={wandb_test_metrics['test/level1_auprc']:.4f}")
            print(f"   Level 2 (Subtype 3-way):       AUROC={wandb_test_metrics['test/level2_auroc']:.4f}  AUPRC={wandb_test_metrics['test/level2_auprc']:.4f}")
        print("="*80 + "\n")

    results = {}
    if accelerator.is_main_process and wandb_test_metrics:
        results = {
            'val_level1_auroc': best_val_auroc,
            'val_level1_auprc': best_val_auprc,
            'test_level1_auroc': wandb_test_metrics['test/level1_auroc'],
            'test_level1_auprc': wandb_test_metrics['test/level1_auprc'],
            'test_level2_auroc': wandb_test_metrics['test/level2_auroc'],
            'test_level2_auprc': wandb_test_metrics['test/level2_auprc'],
        }

    if dist.is_initialized():
        dist.destroy_process_group()

    return results