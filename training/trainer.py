import os
import numpy as np
import wandb
import warnings
import gc
import logging
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
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
from analysis.umap_projection import plot_projection_umap


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
    encoder = MultiModalEncoder(args, disable_cxr=args.disable_cxr, disable_txt=args.disable_txt)
    
    # ssl_save_path = "./best_ssl_model_mask_prob=0.15_noIOcols.pt"
    # checkpoint = torch.load(ssl_save_path, map_location=device)
    # ssl_state_dict = checkpoint

    # pretrained_ts_weights = {}

    # for key, value in ssl_state_dict.items():
    #     if key.startswith('encoder.'):
    #         new_key = key[8:]
    #         if 'pos_encoder.pe' in key:
    #             pretrained_ts_weights[new_key] = value[:, :args.window_size, :]
    #         else:
    #             pretrained_ts_weights[new_key] = value

    # encoder.ts_encoder.load_state_dict(pretrained_ts_weights, strict=True)
    # accelerator.print(f"SSL TS 인코더 가중치 이식 완료")

    model = MultiModalMultiTaskModel(args, encoder)
    accelerator.print(f"\n[Model] MultiModalMultiTaskModel initialized")
    accelerator.print(f"\n[모달리티 상태] CXR 사용: {not model.encoder.disable_cxr}, TEXT 사용: {not model.encoder.disable_txt}")

    # Loss Module
    loss_module = MultiModalLoss(args)

    # Optimizer: separate LR for encoder vs task heads
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': list(model.edema_readout.parameters()), 'lr': args.head_lr},
        # {'params': list(model.subtype_readout.parameters()), 'lr': args.head_lr},
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
        cardio_sum = torch.zeros(1, device=device, dtype=torch.float32)
        cardio_count = torch.zeros(1, device=device, dtype=torch.float32)
        pneumo_sum = torch.zeros(1, device=device, dtype=torch.float32)
        pneumo_count = torch.zeros(1, device=device, dtype=torch.float32)
        info_sum = torch.zeros(1, device=device, dtype=torch.float32)
        info_count = torch.zeros(1, device=device, dtype=torch.float32)

        train_p_pos_list = []
        train_edema_hard_list = []
        train_cxr_anchor_list = []
        # train_subtype_preds_list, train_subtype_labels_list = [], []

        optimizer.zero_grad(set_to_none=True)
        train_sampler.set_epoch(epoch)

        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch+1}] Learning Rate: {current_lr:.2e}")

        # Training
        for _, batch in enumerate(tqdm(train_loader, total=len(train_loader),
                                                desc=f"[Rank {local_rank}] Epoch {epoch+1}/{num_epochs}",
                                                position=local_rank, leave=True, dynamic_ncols=True)):

            with accelerator.accumulate(model):
                total_batch_loss, batch_bce, _, batch_cardio, batch_pneumo, batch_info, batch_outputs, batch_counts = train_batch( # batch_ce는 현재 사용하지 않으니 주석 처리함.
                    args=args,
                    model=model,
                    batch=batch,
                    loss_module=loss_module,
                    device=accelerator.device,
                    accelerator=accelerator,
                    disable_cxr=args.disable_cxr,
                    disable_txt=args.disable_txt,
                    bce_weight=args.bce_weight,
                    cardio_weight=args.cardio_weight,
                    pneumo_weight=args.pneumo_weight,
                    info_weight=args.info_weight,
                    current_epoch=epoch,
                )

                accelerator.backward(total_batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bce_ct_local = torch.as_tensor(batch_counts['bce_count'], device=device, dtype=torch.float32)
            bce_sum += torch.as_tensor(batch_bce, device=device, dtype=torch.float32) * bce_ct_local
            bce_count += bce_ct_local

            cardio_ct_local = torch.as_tensor(batch_counts['cardio_count'], device=device, dtype=torch.float32)
            cardio_sum += torch.as_tensor(batch_cardio, device=device, dtype=torch.float32) * cardio_ct_local
            cardio_count += cardio_ct_local

            pneumo_ct_local = torch.as_tensor(batch_counts['pneumo_count'], device=device, dtype=torch.float32)
            pneumo_sum += torch.as_tensor(batch_pneumo, device=device, dtype=torch.float32) * pneumo_ct_local
            pneumo_count += pneumo_ct_local

            info_ct_local = torch.as_tensor(batch_counts['info_count'], device=device, dtype=torch.float32)
            info_sum += torch.as_tensor(batch_info, device=device, dtype=torch.float32) * info_ct_local
            info_count += info_ct_local

            with torch.no_grad():
                edema_logits = batch_outputs['edema_logits'].squeeze(-1)
                edema_hard = batch_outputs['edema_hard_labels']
                cxr_anchor = batch_outputs['cxr_anchor_mask']

                p_pos = torch.sigmoid(edema_logits)

                train_p_pos_list.append(p_pos.detach().cpu())
                train_edema_hard_list.append(edema_hard.detach().cpu())
                train_cxr_anchor_list.append(cxr_anchor.detach().cpu())
                # subtype 라인 모두 주석

        # Metric aggregation
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bce_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(bce_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(cardio_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cardio_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(pneumo_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(pneumo_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(info_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(info_count, op=dist.ReduceOp.SUM)

        bce_avg = (bce_sum / (bce_count + 1e-8)).item()
        cardio_avg = (cardio_sum / (cardio_count + 1e-8)).item()
        pneumo_avg = (pneumo_sum / (pneumo_count + 1e-8)).item()
        info_avg = (info_sum / (info_count + 1e-8)).item()
        avg_total_loss = (
            args.bce_weight * bce_avg
            + args.cardio_weight * cardio_avg
            + args.pneumo_weight * pneumo_avg
            + args.info_weight * info_avg
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
            print(f"   [Loss Components]")
            print(f"      BCE (Edema soft): {bce_avg:.4f} (λ={args.bce_weight})")
            print(f"      Cardio BCE:       {cardio_avg:.4f} (λ={args.cardio_weight})")
            print(f"      Pneumo BCE:       {pneumo_avg:.4f} (λ={args.pneumo_weight})")
            print(f"      InfoNCE (text):   {info_avg:.4f} (λ={args.info_weight})")

            if train_metrics:
                print("\n   [Clinical Diagnostic (cxr_flag==1)]")
                print(f"   AUROC={train_metrics['auroc']:.4f}  "
                      f"AUPRC={train_metrics['auprc']:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

        # ==================== Validation ====================
        val_loss, val_bce_avg, val_metrics = validate_multitask(
            args=args,
            model=model,
            dataloader=val_loader,
            loss_module=loss_module,
            device=accelerator.device,
            accelerator=accelerator,
            epoch=epoch+1,
            disable_cxr=args.disable_cxr,
            disable_txt=True
            # disable_txt=args.disable_txt,
        )

        # CosineAnnealingLR scheduler step
        scheduler.step()

        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   CosineAnnealingLR - Current LR: {current_lr:.2e}")

        # ==================== Early Stopping ====================
        if accelerator.is_main_process and val_metrics:
            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']

            if val_metrics['auprc'] > best_val_auprc:
                best_val_auprc = val_metrics['auprc']

        # # ==================== Multi-Task UMAP Visualization (주석) ====================
        # accelerator.wait_for_everyone()
        # if accelerator.is_main_process and ((epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs):
        #     print("Generating Training UMAP...")
        #     train_umap_dir = os.path.join(args.umap_save_dir, 'train')
        #     train_reducers = plot_multitask_umap(
        #         args=args, model=model, dataloader=train_loader, dataset=train_loader.dataset,
        #         epoch=epoch+1, save_dir=train_umap_dir, max_samples=10000, umap_reducers=None,
        #     )
        #     print("Training UMAP completed!")
        #
        #     print("Generating Validation UMAP...")
        #     val_umap_dir = os.path.join(args.umap_save_dir, 'val')
        #     plot_multitask_umap(
        #         args=args, model=model, dataloader=val_loader, dataset=val_loader.dataset,
        #         epoch=epoch+1, save_dir=val_umap_dir, max_samples=None, umap_reducers=train_reducers,
        #     )
        #     print("Validation UMAP completed!")
        # accelerator.wait_for_everyone()

        # ==================== WandB Logging ====================
        if accelerator.is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "train/total_loss": avg_total_loss,
                "train/bce_loss": bce_avg,
                "train/cardio_loss": cardio_avg,
                "train/pneumo_loss": pneumo_avg,
                "train/info_loss": info_avg,

                "val/total_loss": val_loss,
                "val/bce_loss": val_bce_avg,
                "val/cardio_loss": val_metrics.get('cardio_loss', float('nan')),
                "val/pneumo_loss": val_metrics.get('pneumo_loss', float('nan')),

                "val/auroc": val_metrics['auroc'],
                "val/auprc": val_metrics['auprc'],

                "val/cardio_auroc": val_metrics.get('cardio_auroc', float('nan')),
                "val/cardio_auprc": val_metrics.get('cardio_auprc', float('nan')),
                "val/pneumo_auroc": val_metrics.get('pneumo_auroc', float('nan')),
                "val/pneumo_auprc": val_metrics.get('pneumo_auprc', float('nan')),
            }

            if train_metrics:
                log_dict.update({
                    "train/auroc": train_metrics['auroc'],
                    "train/auprc": train_metrics['auprc'],
                })

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

    # ==================== Projection UMAP (best 모델 기준 1회만 수행함) ====================
    if accelerator.is_main_process:
        proj_umap_dir = os.path.join(args.umap_save_dir, 'projection')
        for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            try:
                plot_projection_umap(
                    args=args, model=model, dataloader=loader,
                    save_dir=proj_umap_dir, epoch='best', split_name=split_name,
                    max_samples=10000,
                    disable_cxr=args.disable_cxr, disable_txt=args.disable_txt,
                )
            except Exception as e:
                print(f"[UMAP/{split_name}] Skipped due to error: {e}")

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