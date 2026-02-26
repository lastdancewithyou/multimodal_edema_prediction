import os
import numpy as np
import wandb
import warnings
import gc
import logging
from tqdm.auto import tqdm

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

from training.run import parse_arguments
from training.data_processing import get_dataloaders
from training.engine import train_batch
from training.evaluator import test, plot_umap_2d, validate_multitask
from models.main_architecture import MultiModalEncoder, MultiModalContrastiveModel, MultiModalClassificationModel
from loss.losses import MultiModalLoss
from loss.target_metrics import visualize_target_supcon
from utils import stage2_Earlystopping, timer, plot_latent_time_attention, Earlystopping, stage1_earlystopping
from training.umap_multitask import plot_multitask_umap

##################################################################################################
# Model Training Control Center
##################################################################################################
def train_multimodal_model(ts_df, img_df, text_df, demo_df, args):
    if args.stage1_only:
        print("🔀 Running Stage 1 Only (Contrastive Pretraining)")
        return train_representation(ts_df, img_df, text_df, demo_df, args)

    elif args.stage2_only:
        print("🔀 Running Stage 2 Only (Classification Fine-tuning)")
        return train_classifier_with_ce(ts_df, img_df, text_df, demo_df, args, pretrained_encoder=None)

    elif args.use_two_stage:
        print("🔀 Using Two-Stage Training Strategy")
        return train_two_stage_multimodal_model(ts_df, img_df, text_df, demo_df, args)

    else:
        print("🔀 Using Single-Stage Training Strategy")
        return train_single_stage_multimodal_model(ts_df, img_df, text_df, demo_df, args)


def train_two_stage_multimodal_model(ts_df, img_df, text_df, demo_df, args):
    mode = "Linear Probing" if args.freeze_encoder_stage2 else "Fine-tuning"

    print("\n" + "="*80)
    print("🚀 TWO-STAGE TRAINING PROCESS")
    print("="*80)
    print(f"📋 Stage 1: Representation Pretraining ({args.stage1_epochs} epochs)")
    print(f"📋 Stage 2: Classification {mode} ({args.stage2_epochs} epochs)")
    print("="*80 + "\n")

    # Stage 1: Representation Pretraining
    pretrained_encoder = train_representation(ts_df, img_df, text_df, demo_df, args)

    # Stage 2: Classification Fine-tuning (pass pretrained encoder directly)
    results = train_classifier_with_ce(ts_df, img_df, text_df, demo_df, args, pretrained_encoder=pretrained_encoder)

    print("\n" + "="*80)
    print("✅ TWO-STAGE TRAINING COMPLETED!")
    print("="*80)
    print(f"🏆 Final Results:")
    print(f"   Mode: {mode}")
    print(f"   Val AUROC: {results['val_auroc']:.4f}")
    print(f"   Val AUPRC: {results['val_auprc']:.4f}")
    print(f"   Test AUROC: {results['test_auroc']:.4f}")
    print(f"   Test AUPRC: {results['test_auprc']:.4f}")
    print("="*80 + "\n")

    return results


##################################################################################################
# # Single-Stage Training Function
##################################################################################################
def train_single_stage_multimodal_model(ts_df, img_df, text_df, demo_df, args):
    print("\n" + "="*80)
    print("SINGLE-STAGE MULTI-TASK TRAINING")
    print("="*80)
    print(f"Configuration:")
    print(f"   - Epochs: {args.single_stage_epochs}")
    print(f"   - Learning Rate: {args.single_learning_rate}")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   Main Task (Edema Detection):")
    print(f"      BCE Loss: {'Enabled' if args.use_bce else 'Disabled'} (λ={args.bce_weight})")
    print(f"   Sub Task (Subtype Classification & Representation Learning):")
    print(f"      CE Loss: {'Enabled' if args.use_ce else 'Disabled'} (λ={args.ce_weight})")
    print(f"      UCL Loss: {'Enabled' if args.use_ucl else 'Disabled'} (λ={args.ucl_weight})")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # if args.use_target_supcon:
    #     print(f"   Target_SupCon (Optional): Enabled (λ={args.target_supcon_weight})")
    print("="*80 + "\n")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
        device_placement=True
    )
    accelerator.replace_sampler = False
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"🚀 [GPU Configuration]")
        print(f"   Number of GPUs: {accelerator.num_processes}")
        print(f"   Device: {device}")
        print(f"   Mixed Precision: bf16")
        print(f"{'='*60}\n")

        wandb.init(
            project=args.project_name,
            name=args.wandb_run_name,
            config=vars(args),
            tags=["single_stage", "multi_task"]
        )

    # DataLoader
    with timer("Dataset Loading"):
        train_loader, val_loader, test_loader, train_sampler = get_dataloaders(
            ts_df, img_df, text_df, demo_df, args, accelerator
        )

    encoder = MultiModalEncoder(args, disable_cxr=args.disable_cxr, disable_txt=args.disable_txt).to(device)
    model = MultiModalContrastiveModel(encoder).to(device)
    accelerator.print(f"[모달리티 상태] CXR 사용: {not model.encoder.disable_cxr}, TEXT 사용: {not model.encoder.disable_txt}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 [Model Parameters]")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Frozen: {total_params - trainable_params:,}")
    print(f"   Trainable Ratio: {100 * trainable_params / total_params:.1f}%")

    # Loss Module
    loss_module = MultiModalLoss(args, class_weights=None)

    # ==================== Optimizer (Differential Learning Rate) ====================
    encoder_params = [p for n, p in model.named_parameters()
                    if 'edema_classifier' not in n and 'subtype_classifier' not in n and p.requires_grad]
    edema_classifier_params = list(model.edema_classifier.parameters())
    subtype_classifier_params = list(model.subtype_classifier.parameters())

    # Differential learning rates: encoder > edema_classifier > subtype_classifier
    param_groups = [
        {
            'params': encoder_params,
            'lr': args.single_learning_rate,  # Full LR
            'weight_decay': 5e-5
        },
        {
            'params': edema_classifier_params,
            'lr': args.single_learning_rate * 0.5,  # 50% of full LR 
            'weight_decay': 5e-5
        },
        {
            'params': subtype_classifier_params,
            'lr': args.single_learning_rate * 0.1,  # 10% of full LR
            'weight_decay': 5e-5
        }
    ]

    optimizer = torch.optim.AdamW(param_groups)

    print(f"\n🎯 [Optimizer Configuration - Differential Learning Rates]")
    print(f"   Encoder LR: {args.single_learning_rate:.2e} (weight_decay: 5e-5)")
    print(f"   Edema Classifier LR: {args.single_learning_rate * 0.5:.2e} (weight_decay: 5e-5)")
    print(f"   Subtype Classifier LR: {args.single_learning_rate * 0.1:.2e} (weight_decay: 5e-5)")

    model, optimizer, loss_module = accelerator.prepare(model, optimizer, loss_module)

    # ==================== Scheduler Configuration ====================
    warmup_epochs = 5
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )
    main_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.single_stage_epochs - warmup_epochs,
        eta_min=args.single_learning_rate * 0.01
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )

    print(f"\n📅 [Scheduler Configuration]")
    print(f"   Warmup Epochs: {warmup_epochs}")
    print(f"   Warmup Start Factor: 0.1")
    print(f"   CosineAnnealing T_max: {args.single_stage_epochs - warmup_epochs}")
    print(f"   Minimum LR (eta_min): {args.single_learning_rate * 0.01:.2e}")

    # Early Stopping
    single_best_model_path = os.path.join(args.best_model_dir, "single_stage_best_model.pth")
    early_stopper = Earlystopping(
        patience=args.single_patience,
        start_epoch=0,
        save_path=single_best_model_path,
        experiment_id=args.experiment_id
    )

    best_val_auroc, best_val_auprc = 0.0, 0.0
    stop_flag = torch.zeros(1, dtype=torch.uint8, device=device)
    local_rank = accelerator.local_process_index

    # ==================== Training Loop ====================
    for epoch in tqdm(range(args.single_stage_epochs), total=args.single_stage_epochs,
            desc=f"[Rank {local_rank}] 🔄 Single-Stage Training",
            position=local_rank, leave=True, dynamic_ncols=True
        ):

        bce_sum = torch.zeros(1, device=device, dtype=torch.float32)
        bce_count = torch.zeros(1, device=device, dtype=torch.float32)
        ce_sum = torch.zeros(1, device=device, dtype=torch.float32)
        ce_count = torch.zeros(1, device=device, dtype=torch.float32)
        ucl_sum = torch.zeros(1, device=device, dtype=torch.float32)
        ucl_count = torch.zeros(1, device=device, dtype=torch.float32)

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
                                                desc=f"[Rank {local_rank}] Epoch {epoch+1}/{args.single_stage_epochs}",
                                                position=local_rank, leave=True, dynamic_ncols=True)):
            with accelerator.accumulate(model):
                total_batch_loss, batch_bce, batch_ce, batch_ucl, batch_scl, batch_info_ucl, batch_outputs, batch_counts = train_batch(
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
                    ucl_weight=args.ucl_weight,
                )

                accelerator.backward(total_batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Get loss-specific sample counts
            bce_ct_local = torch.as_tensor(batch_counts['bce_count'], device=device, dtype=torch.float32)
            ce_ct_local = torch.as_tensor(batch_counts['ce_count'], device=device, dtype=torch.float32)
            ucl_ct_local = torch.as_tensor(batch_counts['ucl_count'], device=device, dtype=torch.float32)
            scl_ct_local = torch.as_tensor(batch_counts['scl_count'], device=device, dtype=torch.float32)
            infonce_ct_local = torch.as_tensor(batch_counts['infonce_count'], device=device, dtype=torch.float32)

            # Accumulate losses weighted by their actual sample counts
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

            with torch.no_grad():
                edema_logits = batch_outputs['edema_logits'].squeeze(-1)  # [B, W]
                subtype_logits = batch_outputs['subtype_logits']           # [B, W, 2]
                edema_labels = batch_outputs['edema_labels']               # [B, W]
                subtype_labels = batch_outputs['subtype_labels']           # [B, W]
                window_mask = batch['window_mask']                         # [B, W]

                # Valid windows only (padding excluded)
                valid_mask = window_mask.bool()  # [B, W]

                # Probabilities
                p_pos = torch.sigmoid(edema_logits)                       # [B, W] P(edema=1)
                p_sub = torch.softmax(subtype_logits, dim=-1)              # [B, W, 2] P(NCPE|pos), P(CPE|pos)

                # Flatten + mask (collect all valid windows in consistent order)
                p_pos_valid = p_pos[valid_mask]                            # [Nwin]
                p_sub_valid = p_sub[valid_mask]                            # [Nwin, 2]
                edema_valid = edema_labels[valid_mask]                     # [Nwin]
                subtype_valid = subtype_labels[valid_mask]                 # [Nwin]

                train_edema_preds_list.append(p_pos_valid.detach().cpu())
                train_subtype_preds_list.append(p_sub_valid.detach().cpu())
                train_edema_labels_list.append(edema_valid.detach().cpu())
                train_subtype_labels_list.append(subtype_valid.detach().cpu())
        
        # Aggregate across GPUs
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bce_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(bce_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(ce_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ce_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(ucl_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ucl_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(scl_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(scl_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(info_ucl_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(info_ucl_count, op=dist.ReduceOp.SUM)

        # Calculate average losses
        bce_avg = (bce_sum / (bce_count + 1e-8)).item()
        ce_avg = (ce_sum / (ce_count + 1e-8)).item()
        ucl_avg = (ucl_sum / (ucl_count + 1e-8)).item()
        scl_avg = (scl_sum / (scl_count + 1e-8)).item()
        info_ucl_avg = (info_ucl_sum / (info_ucl_count + 1e-8)).item()

        # Weighted contributions
        bce_contrib = args.bce_weight * bce_avg
        ce_contrib = args.ce_weight * ce_avg
        ucl_contrib = args.ucl_weight * ucl_avg
        scl_contrib = args.scl_weight * scl_avg
        info_ucl_contrib = args.infonce_weight * info_ucl_avg
        avg_total_loss = bce_contrib + ce_contrib + ucl_contrib + scl_contrib + info_ucl_contrib

        # Scheduler step
        scheduler.step()

        # Train metrics - Multi-task learning
        train_metrics = {}
        if accelerator.is_main_process and len(train_edema_preds_list) > 0:
            p_pos_all = torch.cat(train_edema_preds_list, dim=0).numpy()      # [N] P(edema=1)
            p_sub_all = torch.cat(train_subtype_preds_list, dim=0).numpy()    # [N, 2] P(NCPE|pos), P(CPE|pos)
            edema_all = torch.cat(train_edema_labels_list, dim=0).numpy()     # [N] in {0, 1, -1}
            subtype_all = torch.cat(train_subtype_labels_list, dim=0).numpy() # [N] in {1, 2, -1}

            # ==================== Level 1: Binary Edema Detection (0 vs 1) ====================
            mask_l1 = (edema_all == 0) | (edema_all == 1)
            y_l1 = edema_all[mask_l1].astype(int)       # {0, 1}
            p_l1 = p_pos_all[mask_l1]                   # P(pos)

            if mask_l1.sum() >= 2 and len(np.unique(y_l1)) >= 2:
                train_metrics['level1_auroc'] = roc_auc_score(y_l1, p_l1)
                train_metrics['level1_auprc'] = average_precision_score(y_l1, p_l1)
                train_metrics["level1_brier"] = brier_score_loss(y_l1, p_l1)
            else: 
                train_metrics['level1_auroc'] = float('nan')
                train_metrics['level1_auprc'] = float('nan')
                train_metrics["level1_brier"] = float('nan')

            # ==================== Level 2: Subtype Classification (NCPE vs CPE | edema=1) ====================
            # Conditional: edema=1 AND subtype in {0, 1}
            mask_l2 = (edema_all == 1) & ((subtype_all == 0) | (subtype_all == 1))
            y_l2 = subtype_all[mask_l2].astype(int)           # Already 0=NCPE, 1=CPE
            p_l2 = p_sub_all[mask_l2, 1]                      # P(CPE|pos)

            if mask_l2.sum() >= 2 and len(np.unique(y_l2)) >= 2:
                train_metrics['level2_auroc'] = roc_auc_score(y_l2, p_l2)
                train_metrics['level2_auprc'] = average_precision_score(y_l2, p_l2)
            else:
                train_metrics['level2_auroc'] = float('nan')
                train_metrics['level2_auprc'] = float('nan')

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

                # # Verify probability sum equals 1
                # prob_sums = probs_3.sum(axis=1)
                # if not np.allclose(prob_sums, 1.0, atol=1e-6):
                #     print(f"[WARNING] Level 3 probability sum check failed!")
                #     print(f"  Min sum: {prob_sums.min():.6f}, Max sum: {prob_sums.max():.6f}")
                #     print(f"  Mean deviation from 1.0: {np.abs(prob_sums - 1.0).mean():.6e}")
                # else:
                #     print(f"✅ Level 3 probability sum verification passed (mean={prob_sums.mean():.2f})")

                y3_bin = label_binarize(y3, classes=[0, 1, 2])

                valid_classes = [k for k in range(3) if 0 < y3_bin[:, k].sum() < len(y3)]

                if len(valid_classes) >= 2:
                    train_metrics['level3_auroc'] = roc_auc_score(
                        y3_bin, probs_3, average='macro', multi_class='ovr'
                    )
                    train_metrics['level3_auprc'] = average_precision_score(
                        y3_bin, probs_3, average='macro'
                    )
                else:
                    train_metrics['level3_auroc'] = float('nan')
                    train_metrics['level3_auprc'] = float('nan')

                # Predictions
                pred_3 = np.argmax(probs_3, axis=1)
                train_metrics['level3_accuracy'] = (pred_3 == y3).mean() * 100

            else:
                train_metrics['level3_auroc'] = float('nan')
                train_metrics['level3_auprc'] = float('nan')
                train_metrics['level3_accuracy'] = float('nan')

        if accelerator.is_main_process:
            print(f"\n✅ Epoch {epoch+1} - Train Total Loss: {avg_total_loss:.4f}")
            # ##################################### Multi-Task Learning #####################################
            print(f"   [Loss Components]")
            print(f"      BCE (Edema): {bce_avg:.4f} → Weighted: {bce_contrib:.4f} (λ={args.bce_weight})")
            print(f"      CE (Subtype): {ce_avg:.4f} → Weighted: {ce_contrib:.4f} (λ={args.ce_weight})")
            print(f"      Temporal UCL: {ucl_avg:.4f} → Weighted: {ucl_contrib:.4f} (λ={args.ucl_weight})")
            print(f"      SCL (Edema): {scl_avg:.4f} → Weighted: {scl_contrib:.4f} (λ={args.scl_weight})")
            print(f"      InfoNCE: {info_ucl_avg:.4f} → Weighted: {info_ucl_contrib:.4f} (λ={args.info_ucl_weight})")


            print(f"\n   [Hierarchical Performance Metrics]")
            print(f"[Edema Detection]   AUROC={train_metrics['level1_auroc']:.4f}  "
                f"AUPRC={train_metrics['level1_auprc']:.4f}  "
                f"Brier={train_metrics['level1_brier']:.4f}")

            print(f"[Subtype Classification] AUROC={train_metrics['level2_auroc']:.4f}  "
                f"AUPRC={train_metrics['level2_auprc']:.4f}")

            print(f"[3-class Classification] AUROC={train_metrics['level3_auroc']:.4f}  "
                f"AUPRC={train_metrics['level3_auprc']:.4f}\n")

        gc.collect()
        torch.cuda.empty_cache()

        # ==================== Validation ====================
        val_loss, val_bce_avg, val_ce_avg, val_ucl_avg, val_scl_avg, val_info_ucl_avg, val_metrics = validate_multitask(
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
        )

        # Early stopping based on Level 3 AUROC 
        if val_metrics['level3_auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['level3_auroc']

        if val_metrics['level3_auprc'] > best_val_auprc:
            best_val_auprc = val_metrics['level3_auprc']

        # ==================== Multi-Task UMAP Visualization ====================
        if accelerator.is_main_process and ((epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == args.single_stage_epochs):
            print("🖼️ Generating Training UMAP...")
            train_umap_dir = os.path.join(args.umap_save_dir, 'train')
            train_reducers = plot_multitask_umap(
                args=args,
                model=model,
                dataloader=train_loader,
                device=accelerator.device,
                accelerator=accelerator,
                dataset=train_loader.dataset,
                epoch=epoch+1,
                save_dir=train_umap_dir,
                max_samples=40000,
                umap_reducers=None
            )
            print("✅ Training UMAP completed!")

            # Validation UMAP (transform using train PCA + UMAP coordinate system)
            print("🖼️ Generating Validation UMAP...")
            val_umap_dir = os.path.join(args.umap_save_dir, 'val')
            plot_multitask_umap(
                args=args,
                model=model,
                dataloader=val_loader,
                device=accelerator.device,
                accelerator=accelerator,
                dataset=val_loader.dataset,
                epoch=epoch+1,
                save_dir=val_umap_dir,
                max_samples=None,
                umap_reducers=train_reducers  # Val mode: use train PCA + UMAP
            )
            print("✅ Validation UMAP completed!")

        if accelerator.is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                #################### Loss components ####################
                "train/total_loss": avg_total_loss,
                "train/bce_loss": bce_avg,
                "train/ce_loss": ce_avg,
                "train/ucl_loss": ucl_avg,
                "train/scl_loss": scl_avg,
                "train/info_ucl_loss": info_ucl_avg,
    
                "val/total_loss": val_loss,
                "val/bce_loss": val_bce_avg,
                "val/ce_loss": val_ce_avg,
                "val/ucl_loss": val_ucl_avg,
                "val/scl_loss": val_scl_avg,
                "val/info_ucl_loss": val_info_ucl_avg,

                #################### Hierarchical Metrics ####################
                "val/level1_auroc": val_metrics['level1_auroc'],
                "val/level1_auprc": val_metrics['level1_auprc'],
                "val/level1_brier": val_metrics['level1_brier'],
                "val/level2_auroc": val_metrics['level2_auroc'],
                "val/level2_auprc": val_metrics['level2_auprc'],
                "val/level3_auroc": val_metrics['level3_auroc'],
                "val/level3_auprc": val_metrics['level3_auprc'],
                
                "train/level1_auroc": train_metrics['level1_auroc'],
                "train/level1_auprc": train_metrics['level1_auprc'],
                "train/level1_brier": train_metrics['level1_brier'],
                "train/level2_auroc": train_metrics['level2_auroc'],
                "train/level2_auprc": train_metrics['level2_auprc'],
                "train/level3_auroc": train_metrics['level3_auroc'],
                "train/level3_auprc": train_metrics['level3_auprc'],
            }

            wandb.log(log_dict)

            # Early stopping
            if early_stopper(args, best_val_auroc, model, epoch):
                stop_flag.fill_(1)
                print(f"⛔ Early stopping triggered at epoch {epoch+1}")

        accelerator.wait_for_everyone()

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
        model.load_state_dict(torch.load(actual_best_model_path, map_location=accelerator.device))
    else:
        accelerator.print(f"⚠️ Best model not found, using current model state")

    # ==================== Test ====================
    test_loss, _, _, _, _, _, _, wandb_test_metrics = test(
        args=args,
        model=model,
        dataloader=test_loader,
        loss_module=loss_module,
        device=accelerator.device,
        accelerator=accelerator,
        dataset=test_loader.dataset
    )

    if accelerator.is_main_process:
        wandb.run.summary.update({
            'final_test/total_loss': test_loss,
            'final_test/level3_auroc': wandb_test_metrics['test/level3_auroc'],
            'final_test/level3_auprc': wandb_test_metrics['test/level3_auprc'],
            'final_test/level1_auroc': wandb_test_metrics['test/level1_auroc'],
            'final_test/level1_auprc': wandb_test_metrics['test/level1_auprc'],
            'final_test/level2_auroc': wandb_test_metrics['test/level2_auroc'],
            'final_test/level2_auprc': wandb_test_metrics['test/level2_auprc'],
        })

    # ==================== UMAP Visualization ====================
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("🖼️ Generating Test UMAP...")
        test_umap_dir = os.path.join(args.umap_save_dir, 'test')
        plot_multitask_umap(
            args=args,
            model=model,
            dataloader=test_loader,
            device=accelerator.device,
            accelerator=accelerator,
            dataset=test_loader.dataset,
            epoch=epoch+1,
            save_dir=test_umap_dir,
            max_samples=None,
            umap_reducers=train_reducers  # Test mode: use train PCA + UMAP
        )
        print("✅ Test UMAP completed!")

    print("\n" + "="*80)
    print("✅ MULTI-TASK TRAINING COMPLETED!")
    print(f"   Best Val Level 3 AUROC: {best_val_auroc:.4f}")
    print(f"   Best Val Level 3 AUPRC: {best_val_auprc:.4f}\n")

    print("   [Test Results - Hierarchical Metrics]")
    print(f"   Level 1 (Edema Detection):        AUROC={wandb_test_metrics['level1_auroc']:.4f}  AUPRC={wandb_test_metrics['level1_auprc']:.4f}")
    print(f"   Level 2 (Subtype Classification): AUROC={wandb_test_metrics['level2_auroc']:.4f}  AUPRC={wandb_test_metrics['level2_auprc']:.4f}")
    print(f"   Level 3 (3-class Combined):       AUROC={wandb_test_metrics['level3_auroc']:.4f}  AUPRC={wandb_test_metrics['level3_auprc']:.4f}")
    print("="*80 + "\n")

    results = {
        'val_level3_auroc': best_val_auroc,
        'val_level3_auprc': best_val_auprc,
        'test_level1_auroc': wandb_test_metrics['level1_auroc'],
        'test_level1_auprc': wandb_test_metrics['level1_auprc'],
        'test_level2_auroc': wandb_test_metrics['level2_auroc'],
        'test_level2_auprc': wandb_test_metrics['level2_auprc'],
        'test_level3_auroc': wandb_test_metrics['level3_auroc'],
        'test_level3_auprc': wandb_test_metrics['level3_auprc']
    }

    if accelerator.num_processes > 1:
        dist.destroy_process_group()

    return results


##################################################################################################
# Stage 1: Representation Pretraining
##################################################################################################
# def train_representation(ts_df, img_df, text_df, demo_df, args):
#     """
#     Stage 1: Representation Learning
#     - No CE Loss, No Classifier training
#     """
#     print("\n" + "="*80)
#     print("STAGE 1: MULTI-TASK REPRESENTATION LEARNING")
#     print("="*80)
#     print(f"Configuration:")
#     print(f"   - Epochs: {args.stage1_epochs}")
#     print(f"   - Learning Rate: {args.stage1_lr}")
#     print(f"   - BCE Loss (Edema): {'Enabled' if args.use_bce else 'Disabled'} (λ={args.bce_weight})")
#     print(f"   - SupCon Loss (Subtype): {'Enabled' if args.use_supcon else 'Disabled'} (λ={args.scl_weight})")
#     print(f"   - TSC Loss: {'Enabled' if args.use_target_supcon else 'Disabled'} (λ={args.target_supcon_weight})")
#     print(f"   - CE Loss: {'Enabled' if args.use_ce else 'Disabled'} (λ={args.ce_weight})")
#     print("="*80 + "\n")

#     # DDP Setup
#     ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
#     accelerator = Accelerator(
#         mixed_precision="bf16",
#         kwargs_handlers=[ddp_kwargs],
#         device_placement=True
#     )
#     accelerator.replace_sampler = False
#     device = accelerator.device

#     if accelerator.is_main_process:
#         print(f"\n{'='*80}")
#         print(f"🚀 [GPU Configuration]")
#         print(f"   Number of GPUs: {accelerator.num_processes}")
#         print(f"   Device: {device}")
#         print(f"   Mixed Precision: bf16")
#         print(f"{'='*80}\n")

#         # wandb.init(
#         #     project=args.project_name,
#         #     name=f"{args.wandb_run_name}_Stage1",
#         #     config=vars(args),
#         #     tags=["stage1", "representation_learning", "TSC"]
#         # )

#     # DataLoader
#     with timer("Dataset 최종 처리 완료"):
#         train_loader, val_loader, _, train_sampler = get_dataloaders(
#             ts_df, img_df, text_df, demo_df, args, accelerator
#         )

#     print("\n[Stage 1] Initializing Model Architecture")

#     encoder = MultiModalEncoder(args, disable_cxr=args.disable_cxr, disable_txt=args.disable_txt)
#     accelerator.print(f"   [모달리티 상태] CXR 사용: {not encoder.disable_cxr}, TEXT 사용: {not encoder.disable_txt}")

#     model = MultiModalContrastiveModel(encoder=encoder,args=args)

#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print(f"\n📊 [Model Parameters - Stage 1]")
#     print(f"   Total: {total_params:,}")
#     print(f"   Trainable: {trainable_params:,}")
#     print(f"   Frozen: {total_params - trainable_params:,}")
#     print(f"   Trainable Ratio: {100 * trainable_params / total_params:.1f}%")

#     # Loss Module
#     loss_module = MultiModalLoss(args, class_weights=None) # CE disabled

#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=args.stage1_lr,
#         weight_decay=1e-4
#     )

#     model, optimizer, loss_module = accelerator.prepare(model, optimizer, loss_module)

#     # ==================== Scheduler Configuration ====================
#     warmup_epochs = 5
#     warmup_scheduler = lr_scheduler.LinearLR(
#         optimizer,
#         start_factor=0.1,
#         total_iters=warmup_epochs
#     )
#     main_scheduler = lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=args.stage1_epochs - warmup_epochs,
#         eta_min=args.stage1_lr * 0.01
#     )
#     scheduler = lr_scheduler.SequentialLR(
#         optimizer,
#         schedulers=[warmup_scheduler, main_scheduler],
#         milestones=[warmup_epochs]
#     )

#     print(f"\n[Scheduler Configuration]")
#     print(f"   Warmup Epochs: {warmup_epochs}")
#     print(f"   Warmup Start Factor: 0.1")
#     print(f"   CosineAnnealing T_max: {args.stage1_lr - warmup_epochs}")
#     print(f"   Minimum LR (eta_min): {args.stage1_lr * 0.01:.2e}")

#     # Early Stopping (현재 Validation 없이 고정된 에포크 학습 중)
#     # early_stopper = stage1_earlystopping(patience=args.patience, start_epoch=0)

#     # best_ssl_loss = float('inf')
#     # stop_flag = torch.zeros(1, dtype=torch.uint8, device=device)
#     local_rank = accelerator.local_process_index

#     # ==================== Training Loop ====================
#     tsc_start_epoch = args.stage1_epochs // 2

#     for epoch in tqdm(range(args.stage1_epochs), total=args.stage1_epochs,
#                     desc=f"[Rank {local_rank}] 🔄 Stage 1 Training",
#                     position=local_rank, leave=True, dynamic_ncols=True):

#         # LR Reset at TSC intervention
#         if args.use_target_supcon and epoch == tsc_start_epoch:
#             accelerator.print(
#             f"\n{'='*60}\n"
#             f"🔁 Phase Shift Detected\n"
#             f"   Target_SupCon activated at epoch {epoch+1}\n"
#             f"   → Resetting LR scheduler (new optimization phase)\n"
#             f"{'='*60}\n"
#         )
            
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = args.stage1_lr

#             warmup_epochs_phase2 = 1

#             warmup_scheduler = lr_scheduler.LinearLR(
#                 optimizer,
#                 start_factor=0.2,
#                 total_iters=warmup_epochs_phase2
#             )

#             main_scheduler = lr_scheduler.CosineAnnealingLR(
#                 optimizer,
#                 T_max=args.stage1_epochs - epoch - warmup_epochs_phase2,
#                 eta_min=args.stage1_lr * 0.01
#             )

#             scheduler = lr_scheduler.SequentialLR(
#                 optimizer,
#                 schedulers=[warmup_scheduler, main_scheduler],
#                 milestones=[warmup_epochs_phase2]
#             )
#         ######################################################################
#         # Loss 누적 변수
#         bce_sum = torch.zeros(1, device=device, dtype=torch.float32)
#         bce_count = torch.zeros(1, device=device, dtype=torch.float32)
#         scl_sum = torch.zeros(1, device=device, dtype=torch.float32)
#         scl_count = torch.zeros(1, device=device, dtype=torch.float32)
#         target_supcon_sum = torch.zeros(1, device=device, dtype=torch.float32)
#         target_supcon_count = torch.zeros(1, device=device, dtype=torch.float32)
#         kcl_sum = torch.zeros(1, device=device, dtype=torch.float32)
#         tsc_sum = torch.zeros(1, device=device, dtype=torch.float32)
#         softmax_prob_sum = 0.0
#         softmax_prob_count = 0

#         optimizer.zero_grad(set_to_none=True)
#         train_sampler.set_epoch(epoch)

#         if accelerator.is_main_process:
#             current_lr = optimizer.param_groups[0]['lr']
#             print(f"\n[Epoch {epoch+1}] Learning Rate: {current_lr:.2e}")

#         # Training
#         for batch_idx, batch in enumerate(tqdm(train_loader, total=len(train_loader),
#                                             desc=f"[Rank {local_rank}] Epoch {epoch+1}/{args.stage1_epochs}",
#                                             position=local_rank, leave=True, dynamic_ncols=True)):
#             with accelerator.accumulate(model):
#                 total_batch_loss, batch_bce, _, batch_scl, batch_target_supcon, batch_kcl, batch_tsc, _, batch_counts = train_batch(
#                     args=args,
#                     model=model,
#                     batch=batch,
#                     loss_module=loss_module,
#                     device=accelerator.device,
#                     accelerator=accelerator,
#                     dataset=train_loader.dataset,
#                     max_length=args.token_max_length,
#                     disable_cxr=args.disable_cxr,
#                     disable_txt=args.disable_txt,
#                     ce_weight=0.0,
#                     current_epoch=epoch,
#                     total_epochs=args.stage1_epochs,
#                 )

#                 accelerator.backward(total_batch_loss)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)

#             window_ct_local = torch.as_tensor(batch_counts['window_count'], device=device, dtype=torch.float32)
#             softmax_prob_sum += batch_counts['softmax_self_prob']
#             softmax_prob_count += 1

#             bce_sum += torch.as_tensor(batch_bce, device=device, dtype=torch.float32) * window_ct_local
#             bce_count += window_ct_local
#             scl_sum += torch.as_tensor(batch_scl, device=device, dtype=torch.float32) * window_ct_local
#             scl_count += window_ct_local
#             target_supcon_sum += torch.as_tensor(batch_target_supcon, device=device, dtype=torch.float32) * window_ct_local
#             target_supcon_count += window_ct_local
#             kcl_sum += torch.as_tensor(batch_kcl, device=device, dtype=torch.float32) * window_ct_local
#             tsc_sum += torch.as_tensor(batch_tsc, device=device, dtype=torch.float32) * window_ct_local

#         # Aggregate across GPUs
#         if dist.is_available() and dist.is_initialized():
#             dist.all_reduce(bce_sum, op=dist.ReduceOp.SUM)
#             dist.all_reduce(bce_count, op=dist.ReduceOp.SUM)
#             dist.all_reduce(scl_sum, op=dist.ReduceOp.SUM)
#             dist.all_reduce(scl_count, op=dist.ReduceOp.SUM)
#             dist.all_reduce(target_supcon_sum, op=dist.ReduceOp.SUM)
#             dist.all_reduce(target_supcon_count, op=dist.ReduceOp.SUM)
#             dist.all_reduce(kcl_sum, op=dist.ReduceOp.SUM)
#             dist.all_reduce(tsc_sum, op=dist.ReduceOp.SUM)

#         bce_avg = (bce_sum / (bce_count + 1e-8)).item()
#         scl_avg = (scl_sum / (scl_count + 1e-8)).item()
#         target_supcon_avg = (target_supcon_sum / (target_supcon_count + 1e-8)).item()
#         kcl_avg = (kcl_sum / (target_supcon_count + 1e-8)).item()
#         tsc_avg = (tsc_sum / (target_supcon_count + 1e-8)).item()

#         bce_weight = args.bce_weight if args.use_bce else 0.0
#         scl_weight = args.scl_weight
#         target_supcon_weight = args.target_supcon_weight

#         bce_contrib = bce_weight * bce_avg
#         scl_contrib = scl_weight * scl_avg
#         tsc_contrib = target_supcon_weight * target_supcon_avg

#         avg_stage1_loss = bce_contrib + scl_contrib + tsc_contrib

#         scheduler.step()

#         softmax_prob_avg = softmax_prob_sum / max(softmax_prob_count, 1)
#         accelerator.print(
#             f"✅ Epoch {epoch+1} - Train Stage1 Loss: {avg_stage1_loss:.4f}\n"
#             f"   BCE (Edema): {bce_avg:.4f}, SupCon (Subtype): {scl_avg:.4f}\n"
#             f"   Target SupCon: {target_supcon_avg:.4f}, KCL: {kcl_avg:.4f}, TSC: {tsc_avg:.4f}\n"
#             f"   Softmax self-prob (pos 0): {softmax_prob_avg:.4f}  "
#         )

#         gc.collect()
#         torch.cuda.empty_cache()

#         # ==================== Embedding Visualization ====================
#         if accelerator.is_main_process and args.use_target_supcon:
#             tsc_activated = (epoch >= args.stage1_epochs // 2)
#             tsc_first_epoch = (epoch == args.stage1_epochs // 2)
#             kcl_only = not tsc_activated

#             # KCL 구간: 5 epoch마다 + 마지막 KCL epoch
#             kcl_last_epoch = (epoch == args.stage1_epochs // 2 - 1)
#             should_visualize_kcl = kcl_only and ((epoch + 1) % 5 == 0 or kcl_last_epoch)

#             # TSC 구간: 첫 epoch + 5 epoch마다 + 마지막 epoch
#             should_visualize_tsc = tsc_activated and (tsc_first_epoch or (epoch + 1) % 5 == 0 or epoch == args.stage1_epochs - 1)

#             if should_visualize_kcl:
#                 vis_save_path = f'{args.umap_save_dir}/stage1_kcl_only_epoch{epoch+1}.png'
#                 print(f"🎨 Generating KCL-only Visualization (Epoch {epoch+1})")
#                 os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
#                 visualize_target_supcon(
#                     model=model,
#                     dataloader=train_loader,
#                     loss_module=loss_module,
#                     device=accelerator.device,
#                     save_path=vis_save_path,
#                     max_samples=5000,
#                     epoch=epoch+1,
#                     use_target=False,
#                 )

#             if should_visualize_tsc:
#                 vis_save_path = f'{args.umap_save_dir}/stage1_target_supcon_epoch{epoch+1}.png'
#                 print(f"🎨 Generating TSC Visualization (Epoch {epoch+1})")
#                 os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
#                 visualize_target_supcon(
#                     model=model,
#                     dataloader=train_loader,
#                     loss_module=loss_module,
#                     device=accelerator.device,
#                     save_path=vis_save_path,
#                     max_samples=3000,
#                     epoch=epoch+1
#                 )
        
#             # wandb.log({
#             #     "epoch": epoch + 1,
#             #     "train_loss": avg_stage1_loss,
#             #     "target_supcon_avg":target_supcon_avg,
#             #     "kcl_avg": kcl_avg,
#             #     "tsc_avg": tsc_avg
#             # })

#     os.makedirs(os.path.dirname(args.stage1_model_path), exist_ok=True)

#     # Save only the encoder
#     unwrapped_model = accelerator.unwrap_model(model)
#     torch.save(unwrapped_model.encoder.state_dict(), args.stage1_model_path)
#     print(f"[Stage 1] ✅ Encoder saved at (Train Loss: {avg_stage1_loss:.4f})")
#     print(f"   Saved path: {args.stage1_model_path}")

#         # ==================== Stage 1 Validation ====================
#         # 일단 validation은 주석 처리함.
#     #     val_stage1_loss, val_ts_recon_avg, val_local_temp_avg, val_scl_avg, val_time_aware_avg = validate_stage1(
#     #         args=args,
#     #         model=model,
#     #         dataloader=val_loader,
#     #         loss_module=loss_module,
#     #         device=accelerator.device,
#     #         accelerator=accelerator,
#     #         dataset=val_loader.dataset,
#     #         epoch=epoch+1,
#     #         disable_cxr=args.disable_cxr,
#     #         disable_txt=args.disable_txt,
#     #         max_length=args.token_max_length
#     #     )

#     #     if accelerator.is_main_process:
#     #         # wandb.log({
#     #         #     "epoch": epoch + 1,
#     #         #     "train_ts_recon": ts_recon_avg,
#     #         #     "train_local_temporal": local_temp_avg,
#     #         #     "train_supcon": scl_avg,
#     #         #     "train_time_aware": time_aware_avg,
#     #         #     "train_stage1_loss": avg_stage1_loss,
#     #         #     "val_ts_recon": val_ts_recon_avg,
#     #         #     "val_local_temporal": val_local_temp_avg,
#     #         #     "val_supcon": val_scl_avg,
#     #         #     "val_time_aware": val_time_aware_avg,
#     #         #     "val_stage1_loss": val_stage1_loss,
#     #         # })

#     #         # Save best model (based on weighted SupCon + Time-Aware loss)
#     #         stage1_val_loss = args.scl_weight * val_scl_avg + args.time_aware_weight * val_time_aware_avg
#     #         if stage1_val_loss < best_ssl_loss:
#     #             best_ssl_loss = stage1_val_loss

#     #             os.makedirs(os.path.dirname(args.stage1_model_path), exist_ok=True)
#     #             torch.save(model.state_dict(), args.stage1_model_path)
#     #             print(f"[Stage 1] ✅ Best model saved (Val Loss: {best_ssl_loss:.4f})")

#     #         # Early stopping
#     #         if early_stopper(args, stage1_val_loss, model, epoch):
#     #             stop_flag.fill_(1)
#     #             print(f"⛔ Early stopping triggered at epoch {epoch+1} (Val Loss: {stage1_val_loss:.4f})")

#     #         # ReduceLROnPlateau scheduler step (validation-based)
#     #         if use_reduce_on_plateau:
#     #             scheduler.step(stage1_val_loss)  # Val SupCon loss 기준
#     #             current_lr = optimizer.param_groups[0]['lr']
#     #             print(f"[Scheduler] Current LR: {current_lr:.2e}")

#     #     # ==================== Visualize Attention (Epoch 0 only) ====================
#     #     if epoch == 0 and accelerator.is_main_process:
#     #         actual_model = accelerator.unwrap_model(model)
#     #         attn = actual_model.ts_centric_fusion.debug_ts_attn  # [B, L, T]

#     #         if attn is not None:
#     #             save_dir = "/home/DAHS1/gangmin/my_research/src/output/latent_attention_map"
#     #             os.makedirs(save_dir, exist_ok=True)

#     #             # Plot only first sample to avoid huge images
#     #             attn_first = attn[:1]  # [1, L, T] - only first sample
#     #             plot_latent_time_attention(
#     #                 attn_first,
#     #                 save_path=f"{save_dir}/latent_attention_timestep_figure.png"
#     #             )
#     #             print(f"✓ Saved attention visualization to {save_dir}/latent_attention_epoch0.png")

#     #     if dist.is_initialized() and dist.get_world_size() > 1:
#     #         gathered_stop_flag = accelerator.gather_for_metrics(stop_flag)
#     #         stop_flag.fill_(gathered_stop_flag.max().item())

#     #     if stop_flag.item() == 1:
#     #         break

#     # accelerator.wait_for_everyone()

#     # ==================== Stage 1 Final UMAP (Val Set) ====================
#     # if accelerator.is_main_process:
#     #     print("\n" + "="*60)
#     #     print("🎨 Generating Stage 1 Final UMAP (Val Set)")
#     #     print("="*60)

#     #     # Load best model from Stage 1
#     #     stage1_best_path = args.stage1_model_path
#     #     if os.path.exists(stage1_best_path):
#     #         model.load_state_dict(torch.load(stage1_best_path, map_location=device))
#     #         print(f"✅ Loaded best Stage 1 model from: {stage1_best_path}")
#     #     else:
#     #         print(f"⚠️ Best model not found at {stage1_best_path}, using current model state")

#     #     model.eval()
#     #     val_embeddings = []
#     #     val_labels = []

#     #     with torch.no_grad():
#     #         for batch in tqdm(val_loader, desc="Collecting val embeddings"):
#     #             img_index_tensor = batch['img_index_tensor']
#     #             txt_index_tensor = batch['text_index_tensor']
#     #             has_cxr = (img_index_tensor != -1).long().to(device, non_blocking=True)
#     #             has_text = (txt_index_tensor != -1).long().to(device, non_blocking=True)

#     #             labels = batch['labels'].to(device)
#     #             window_mask = batch['window_mask'].to(device)
#     #             seq_valid_mask = batch['valid_seq_mask'].to(device)

#     #             demo_features = batch.get('demo_features')
#     #             demo_features = demo_features.to(device, non_blocking=True)

#     #             ts_series, cxr_data, text_data, has_cxr, has_text = prepare_multiview_inputs_v2(
#     #                 batch, device, has_cxr, has_text, val_loader.dataset,
#     #                 disable_cxr=args.disable_cxr, disable_txt=args.disable_txt,
#     #                 max_length=256
#     #             )

#     #             time_steps = batch.get('time_steps', None)
#     #             if time_steps is not None:
#     #                 time_steps = time_steps.to(device, non_blocking=True)

#     #             classification_input, _, logits, _ = model(
#     #                 args, ts_series, cxr_data, text_data, has_cxr, has_text, window_mask, seq_valid_mask, demo_features, time_steps=time_steps
#     #             )

#     #             # Extract valid embeddings and labels
#     #             inference_results = loss_module.inference(classification_input, logits, labels, window_mask)
#     #             window_embeddings = inference_results['window_embeddings']  # [B, W, D]
#     #             window_labels = inference_results['labels']  # [B, W]
#     #             valid_mask = inference_results['window_mask'].bool()  # [B, W]

#     #             masked_embeddings = window_embeddings[valid_mask]
#     #             masked_labels = window_labels[valid_mask]

#     #             label_mask = masked_labels != -1
#     #             if label_mask.any():
#     #                 val_embeddings.append(masked_embeddings[label_mask].cpu())
#     #                 val_labels.append(masked_labels[label_mask].cpu())

#     #     if len(val_embeddings) > 0:
#     #         val_embeddings = torch.cat(val_embeddings, dim=0)
#     #         val_labels = torch.cat(val_labels, dim=0)

#     #         total_samples = val_embeddings.size(0)
#     #         print(f"Collected {total_samples} validation embeddings")

#     #         plot_umap_2d(
#     #             args=args,
#     #             window_embeddings_list=[val_embeddings],
#     #             window_labels_list=[val_labels],
#     #             window_pred_list=None,
#     #             epoch="stage1_final",
#     #             prefix="stage1_val",
#     #             pca_model=None
#     #         )
#     #         print("✅ Stage 1 Final UMAP saved!")
#     #     else:
#     #         print("⚠️  No valid embeddings collected for UMAP")

#     #     print("="*60 + "\n")

#     # # if accelerator.is_main_process and wandb.run is not None:
#     # #     wandb.run.summary['stage1_best_loss'] = best_ssl_loss
#     # #     wandb.finish()

#     wandb.finish()
#     print("✅ STAGE 1 COMPLETED!")
#     print(f"   Model saved at: {args.stage1_model_path}")

#     # Return the trained encoder for direct use in Stage 2 (if running two-stage)
#     return unwrapped_model.encoder


##################################################################################################
# Stage 2: Classification Fine-tuning (Linear Probing or Fine-tuning)
##################################################################################################
# def train_classifier_with_ce(ts_df, img_df, text_df, demo_df, args, pretrained_encoder=None):
#     """
#     Stage 2: Classification Fine-tuning
#     - Load pretrained encoder from Stage 1 (from memory or file)
#     - Encoder → Linear Classifier → CE Loss
#     - Two modes:
#         1. Linear Probing: freeze_encoder_stage2=True (Encoder frozen)
#         2. Full Fine-tuning: freeze_encoder_stage2=False (Encoder trainable)

#     Args:
#         pretrained_encoder: If provided, use this encoder directly (from Stage 1 in two-stage training)
#                            If None, load from args.stage1_model_path
#     """
#     mode = "Linear Probing" if args.freeze_encoder_stage2 else "Fine-tuning"

#     print("\n" + "="*80)
#     print(f"STAGE 2: CLASSIFICATION {mode.upper()}")
#     print("="*80)
#     print(f"Configuration:")
#     print(f"   - Mode: {mode}")
#     print(f"   - Freeze Encoder: {args.freeze_encoder_stage2}")
#     print(f"   - Epochs: {args.stage2_epochs}")
#     print(f"   - CE Weight: {args.ce_weight}")
#     print(f"   - UCL/SCL: Disabled")
#     print(f"   - Pretrained Model: {args.stage1_model_path}")
#     print("="*80 + "\n")

#     ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
#     accelerator = Accelerator(
#         mixed_precision="bf16",
#         kwargs_handlers=[ddp_kwargs],
#         device_placement=True
#     )
#     accelerator.replace_sampler = False
#     device = accelerator.device

#     if accelerator.is_main_process:
#         print(f"\n{'='*60}")
#         print(f"🚀 [GPU Configuration]")
#         print(f"   Number of GPUs: {accelerator.num_processes}")
#         print(f"   Device: {device}")
#         print(f"   Mixed Precision: bf16")
#         print(f"{'='*60}\n")

#         wandb.init(
#             project=args.project_name,
#             name=f"{args.wandb_run_name}_Stage2_{mode.replace(' ', '_')}",
#             config=vars(args),
#             tags=["stage2", "classification"]
#         )

#     # DataLoader
#     with timer("Dataset Loading"):
#         train_loader, val_loader, test_loader, train_sampler = get_dataloaders(
#             ts_df, img_df, text_df, demo_df, args, accelerator
#         )
#         ###################################################################
#         distribution = train_sampler.get_actual_class_distribution()

#         """
#         # Balanced CE
#         Negative: 0.2258
#         Non-cardio: 2.3129
#         Cardio: 0.4613
#         """
#         # class_weights_list = [
#         #     1.0 / (distribution['negative'] + 1e-8),
#         #     1.0 / (distribution['noncardio'] + 1e-8),
#         #     1.0 / (distribution['cardio'] + 1e-8),
#         # ]
#         # class_weights_sum = sum(class_weights_list)
#         # class_weights_list = [w * len(class_weights_list) / class_weights_sum for w in class_weights_list]
#         # computed_class_weights = torch.tensor(class_weights_list, dtype=torch.float32, device=accelerator.device)
#         ###################################################################
#         """
#         # ENS
#         - Class 1만 학습이 너무 잘 되어서 마련한 해결방안
#         beta 0.99999
#         Negative: 0.3835
#         Non-cardio: 2.0526
#         Cardio: 0.5590

#         beta 0.99995
#         Negative: 0.7968
#         Non-cardio: 1.3949
#         Cardio: 0.8083
#         """
#         beta = 0.99996
#         n_neg = distribution['negative_count']
#         n_nc = distribution['noncardio_count']
#         n_c = distribution['cardio_count']

#         samples_per_class = torch.tensor([n_neg, n_nc, n_c], dtype=torch.float32, device=accelerator.device)
#         effective_num = 1.0 - torch.pow(beta, samples_per_class)

#         class_weights = (1.0 - beta) / (effective_num + 1e-8)
#         class_weights = class_weights * (len(class_weights) / class_weights.sum())
#         # print("effective_num: ", effective_num)
#         # print("class_weights: ", class_weights)
#         computed_class_weights = class_weights
#         ###################################################################
#         if accelerator.is_main_process:
#             print(f"\n[Dynamic ENS Class Weights]")
#             print(f"   Negative (class 0): {computed_class_weights[0]:.4f}")
#             print(f"   Non-cardio (class 1): {computed_class_weights[1]:.4f}")
#             print(f"   Cardio (class 2): {computed_class_weights[2]:.4f}")

#     print("\n[Stage 2] Initializing Model Architecture")

#     # Use pretrained encoder from memory (two-stage) or load from file (stage2-only)
#     if pretrained_encoder is not None:
#         print("📦 Using pretrained encoder from Stage 1 (in-memory)")
#         encoder = pretrained_encoder
#         print("✅ Encoder loaded from Stage 1 training session")

#     elif args.stage1_model_path and os.path.exists(args.stage1_model_path):
#         print(f"📂 Loading pretrained encoder from file: {args.stage1_model_path}")

#         encoder = MultiModalEncoder(args, disable_cxr=args.disable_cxr, disable_txt=args.disable_txt)

#         encoder_state_dict = torch.load(args.stage1_model_path, map_location='cpu')
#         encoder.load_state_dict(encoder_state_dict, strict=True)
#         print("✅ Encoder loaded from Stage 1 checkpoint file")

#     else:
#         print("❌ WARNING: No pretrained encoder provided. Creating new encoder from scratch")
#         print(f"   This will result in poor performance (no pretraining)")
#         encoder = MultiModalEncoder(args, disable_cxr=args.disable_cxr, disable_txt=args.disable_txt)

#     model = MultiModalClassificationModel(
#         encoder=encoder,
#         args=args,
#         freeze_encoder=args.freeze_encoder_stage2
#     )

#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print(f"\n📊 [Model Parameters - Stage 2]")
#     print(f"   Total: {total_params:,}")
#     print(f"   Trainable: {trainable_params:,}")
#     print(f"   Frozen: {total_params - trainable_params:,}")
#     print(f"   Trainable Ratio: {100 * trainable_params / total_params:.1f}%")

#     # Loss Module (Contrastive disabled, CE enabled)
#     loss_module = MultiModalLoss(args, class_weights=computed_class_weights)

#     encoder_params = [p for n, p in model.named_parameters() if 'classifier' not in n and p.requires_grad]
#     classifier_params = list(model.classifier.parameters())

#     optimizer = torch.optim.AdamW([
#         {
#             'params': encoder_params,
#             'lr': args.stage2_lr,               # Encoder: 1e-4 (stage2_lr)
#             'weight_decay': 5e-4               
#         },
#         {
#             'params': classifier_params,
#             'lr': args.stage2_lr * 3,           # Classifier: 3e-4 (encoder의 3배)
#             'weight_decay': 1e-4
#         }
#     ])

#     print(f"\n🎯 [Differential Learning Rate]")
#     print(f"   Encoder params: {sum(p.numel() for p in encoder_params):,}")
#     print(f"   Classifier params: {sum(p.numel() for p in classifier_params):,}")

#     model, optimizer, loss_module = accelerator.prepare(model, optimizer, loss_module)

#     # ==================== Scheduler Configuration ====================
#     warmup_epochs = 3
#     warmup_scheduler = lr_scheduler.LinearLR(
#         optimizer,
#         start_factor=0.1,
#         total_iters=warmup_epochs
#     )
#     main_scheduler = lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=args.stage2_epochs - warmup_epochs,
#         eta_min=1e-7
#     )
#     scheduler = lr_scheduler.SequentialLR(
#         optimizer,
#         schedulers=[warmup_scheduler, main_scheduler],
#         milestones=[warmup_epochs]
#     )
#     print(f"\n[Scheduler Configuration]")
#     print(f"   Warmup Epochs: {warmup_epochs}")
#     print(f"   Warmup Start Factor: 0.1 (Encoder: {5e-5 * 0.1:.2e}, Classifier: {1e-4 * 0.1:.2e})")
#     print(f"   CosineAnnealing T_max: {args.stage2_epochs - warmup_epochs}")
#     print(f"   Minimum LR (eta_min): 1e-7")
#     # ================================================================
#     # Stage 2 Early Stopping
#     stage2_best_model_path = os.path.join(args.best_model_dir, "stage2_best_model.pth")
#     early_stopper = stage2_Earlystopping(
#         patience=args.stage2_patience,
#         start_epoch=0,
#         save_path=stage2_best_model_path,
#         experiment_id=args.experiment_id
#     )

#     best_auroc = 0.0
#     best_auprc = 0.0
#     stop_flag = torch.zeros(1, dtype=torch.uint8, device=device)
#     local_rank = accelerator.local_process_index

#     # Training Loop
#     for epoch in tqdm(range(args.stage2_epochs), total=args.stage2_epochs,
#                     desc=f"[Rank {local_rank}] 🔄 Stage 2 Training",
#                     position=local_rank, leave=True, dynamic_ncols=True):

#         ce_sum = torch.zeros(1, device=device, dtype=torch.float32)
#         ce_count = torch.zeros(1, device=device, dtype=torch.float32)

#         train_probs_list = []
#         train_labels_list = []

#         optimizer.zero_grad(set_to_none=True)
#         train_sampler.set_epoch(epoch)

#         if accelerator.is_main_process:
#             encoder_lr = optimizer.param_groups[0]['lr']
#             classifier_lr = optimizer.param_groups[1]['lr']
#             print(f"\n[Epoch {epoch+1}] Learning Rate - Encoder: {encoder_lr:.2e}, Classifier: {classifier_lr:.2e}")


#         for batch_idx, batch in enumerate(tqdm(train_loader, total=len(train_loader),
#                                             desc=f"[Rank {local_rank}] Epoch {epoch+1}/{args.stage2_epochs}",
#                                             position=local_rank, leave=True, dynamic_ncols=True)):
#             with accelerator.accumulate(model):
#                 total_batch_loss, batch_ce, _, _, _, _, batch_outputs, batch_counts = train_batch(
#                     args=args,
#                     model=model,
#                     batch=batch,
#                     loss_module=loss_module,
#                     device=accelerator.device,
#                     accelerator=accelerator,
#                     dataset=train_loader.dataset,
#                     max_length=args.token_max_length,
#                     disable_cxr=args.disable_cxr,
#                     disable_txt=args.disable_txt,
#                     ce_weight=args.ce_weight,           # Stage 2: Use CE
#                 )

#                 accelerator.backward(total_batch_loss)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)

#             # Logging
#             ce_ct_local = torch.as_tensor(batch_counts['ce_count'], device=device, dtype=torch.float32)
#             ce_sum += torch.as_tensor(batch_ce, device=device, dtype=torch.float32) * ce_ct_local
#             ce_count += ce_ct_local

#             with torch.no_grad():
#                 labels = batch_outputs['labels']
#                 logits = batch_outputs['logits']

#                 # 유효한 레이블만 필터링 (label != -1)
#                 valid_mask = (labels != -1)
#                 if valid_mask.any():
#                     valid_labels = labels[valid_mask]
#                     valid_logits = logits[valid_mask]

#                     probs = F.softmax(valid_logits, dim=-1)

#                     train_probs_list.append(probs.detach().cpu())
#                     train_labels_list.append(valid_labels.detach().cpu())

#         if dist.is_available() and dist.is_initialized():
#             dist.all_reduce(ce_sum, op=dist.ReduceOp.SUM)
#             dist.all_reduce(ce_count, op=dist.ReduceOp.SUM)

#         CE_avg = (ce_sum / ce_count).item()

#         scheduler.step()

#         train_metrics = {}
#         if accelerator.is_main_process and len(train_probs_list) > 0:
#             train_probs = torch.cat(train_probs_list, dim=0).numpy()  # [N, num_classes]
#             train_labels = torch.cat(train_labels_list, dim=0).numpy()  # [N]
#             train_preds = train_probs.argmax(axis=-1)

#             # Precision/Recall
#             precision, recall, _, _ = precision_recall_fscore_support(
#                 train_labels, train_preds, average='macro', zero_division=0
#             )
#             train_metrics['precision_macro'] = precision
#             train_metrics['recall_macro'] = recall

#             # AUROC/AUPRC
#             try:
#                 train_labels_binarized = label_binarize(train_labels, classes=list(range(args.num_classes)))
#                 auroc = roc_auc_score(train_labels_binarized, train_probs, average='macro', multi_class='ovr')
#                 auprc = average_precision_score(train_labels_binarized, train_probs, average='macro')
#                 train_metrics['auroc_macro'] = auroc
#                 train_metrics['auprc_macro'] = auprc
#             except:
#                 train_metrics['auroc_macro'] = float('nan')
#                 train_metrics['auprc_macro'] = float('nan')

#             # Per-class accuracy
#             print("\n[Train Accuracy by classes]")
#             for label in range(args.num_classes):
#                 total = (train_labels == label).sum()
#                 correct = ((train_labels == label) & (train_preds == label)).sum()

#                 if total > 0:
#                     acc = 100.0 * correct / total
#                     train_metrics[f'class_{label}_accuracy'] = acc
#                     train_metrics[f'class_{label}_count'] = total
#                     print(f"Label {label}: {correct}/{total} = {acc:.1f}%")
#                 else:
#                     train_metrics[f'class_{label}_accuracy'] = None
#                     train_metrics[f'class_{label}_count'] = 0
#                     print(f"Label {label}: No samples.")

#             accelerator.print(
#                 f"✅ Epoch {epoch+1} - Train CE: {CE_avg:.4f} | "
#                 f"Precision: {train_metrics['precision_macro']:.4f} | "
#                 f"Recall: {train_metrics['recall_macro']:.4f} | "
#                 f"AUROC: {train_metrics['auroc_macro']:.4f} | "
#                 f"AUPRC: {train_metrics['auprc_macro']:.4f}"
#             )
#         else:
#             accelerator.print(f"✅ Epoch {epoch+1} - Train CE Loss: {CE_avg:.4f}")

#         gc.collect()
#         torch.cuda.empty_cache()

#         # Validation
#         val_loss, _, val_metrics, _, _, _, _ = validate(
#             args=args,
#             model=model,
#             dataloader=val_loader,
#             loss_module=loss_module,
#             device=accelerator.device,
#             accelerator=accelerator,
#             dataset=val_loader.dataset,
#             epoch=epoch+1,
#             disable_cxr=args.disable_cxr,
#             disable_txt=args.disable_txt
#         )

#         if val_metrics['auroc_macro'] > best_auroc:
#             best_auroc = val_metrics['auroc_macro']

#         if val_metrics['auprc_macro'] > best_auprc:
#             best_auprc = val_metrics['auprc_macro']

#         if accelerator.is_main_process:
#             accelerator.print(
#                 f"[Epoch {epoch+1}] Val Loss={val_loss:.4f} | "
#                 f"Val AUROC={val_metrics['auroc_macro']:.4f} | "
#                 f"Val AUPRC={val_metrics['auprc_macro']:.4f}"
#             )

#             log_dict = {
#                 "epoch": epoch + 1,
#                 "train_ce": CE_avg,
#                 "val_loss": val_loss,
#                 "val_auroc": val_metrics['auroc_macro'],
#                 "val_auprc": val_metrics['auprc_macro'],
#                 "val_precision": val_metrics['precision_macro'],
#                 "val_recall": val_metrics['recall_macro'],
#             }

#             if train_metrics:
#                 log_dict["train_precision"] = train_metrics['precision_macro']
#                 log_dict["train_recall"] = train_metrics['recall_macro']
#                 log_dict["train_auroc"] = train_metrics['auroc_macro']
#                 log_dict["train_auprc"] = train_metrics['auprc_macro']

#             for k, v in val_metrics.items():
#                 if k.endswith('_accuracy'):
#                     log_dict[f"val_{k}"] = v

#             wandb.log(log_dict)

#             # Early stopping
#             if early_stopper(args, val_metrics['auroc_macro'], model, epoch):
#                 stop_flag.fill_(1)
#                 print(f"⛔ Early stopping triggered at epoch {epoch+1}")

#         accelerator.wait_for_everyone()

#         if dist.is_initialized() and dist.get_world_size() > 1:
#             gathered_stop_flag = accelerator.gather_for_metrics(stop_flag)
#             stop_flag.fill_(gathered_stop_flag.max().item())

#         if stop_flag.item() == 1:
#             break

#     accelerator.wait_for_everyone()

#     # ==================== Load Best Model ====================
#     # early_stopper가 실제로 저장한 경로 사용
#     actual_best_model_path = early_stopper.get_best_model_path()
#     if actual_best_model_path and os.path.exists(actual_best_model_path):
#         accelerator.print(f"✅ Loading best model from: {actual_best_model_path}")
#         model.load_state_dict(torch.load(actual_best_model_path, map_location=accelerator.device))
#     else:
#         accelerator.print(f"⚠️ Best model not found at {actual_best_model_path}, using current model state")

#     # ==================== Stage 2 Final UMAP (Val Set) ====================
#     # if accelerator.is_main_process:
#     #     print("\n" + "="*60)
#     #     print("🎨 Generating Stage 2 Final UMAP (Val Set)")
#     #     print("="*60)

#     #     model.eval()
#     #     val_embeddings = []
#     #     val_labels = []

#     #     with torch.no_grad():
#     #         for batch in tqdm(val_loader, desc="Collecting val embeddings"):
#     #             img_index_tensor = batch['img_index_tensor']
#     #             txt_index_tensor = batch['text_index_tensor']
#     #             has_cxr = (img_index_tensor != -1).long().to(device, non_blocking=True)
#     #             has_text = (txt_index_tensor != -1).long().to(device, non_blocking=True)

#     #             labels = batch['labels'].to(device)
#     #             window_mask = batch['window_mask'].to(device)
#     #             seq_valid_mask = batch['valid_seq_mask'].to(device)

#     #             demo_features = batch.get('demo_features')
#     #             demo_features = demo_features.to(device, non_blocking=True)

#     #             ts_series, cxr_data, text_data, has_cxr, has_text = prepare_multiview_inputs_v2(
#     #                 batch, device, has_cxr, has_text, val_loader.dataset,
#     #                 disable_cxr=args.disable_cxr, disable_txt=args.disable_txt,
#     #                 max_length=256
#     #             )

#     #             time_steps = batch.get('time_steps', None)
#     #             if time_steps is not None:
#     #                 time_steps = time_steps.to(device, non_blocking=True)

#     #             classification_input, _, logits, _ = model(
#     #                 args, ts_series, cxr_data, text_data, has_cxr, has_text, window_mask, seq_valid_mask, demo_features, time_steps=time_steps
#     #             )

#     #             inference_results = loss_module.inference(classification_input, logits, labels, window_mask)
#     #             window_embeddings = inference_results['window_embeddings']  # [B, W, D]
#     #             window_labels = inference_results['labels']  # [B, W]
#     #             valid_mask = inference_results['window_mask'].bool()  # [B, W]

#     #             masked_embeddings = window_embeddings[valid_mask]
#     #             masked_labels = window_labels[valid_mask]

#     #             label_mask = masked_labels != -1
#     #             if label_mask.any():
#     #                 val_embeddings.append(masked_embeddings[label_mask].cpu())
#     #                 val_labels.append(masked_labels[label_mask].cpu())

#     #     if len(val_embeddings) > 0:
#     #         val_embeddings = torch.cat(val_embeddings, dim=0)
#     #         val_labels = torch.cat(val_labels, dim=0)

#     #         total_samples = val_embeddings.size(0)
#     #         print(f"Collected {total_samples} validation embeddings")

#     #         plot_umap_2d(
#     #             args=args,
#     #             window_embeddings_list=[val_embeddings],
#     #             window_labels_list=[val_labels],
#     #             window_pred_list=None,
#     #             epoch="stage2_final",
#     #             prefix="stage2_val",
#     #             pca_model=None
#     #         )
#     #         print("✅ Stage 2 Final UMAP saved!")
#     #     else:
#     #         print("⚠️  No valid embeddings collected for UMAP")

#     #     print("="*60 + "\n")

#     # ==================== Stage 2 Test ====================
#     test_loss, test_metrics, test_window_embeddings, test_window_labels, test_window_preds, _ = test(
#         args=args,
#         model=model,
#         dataloader=test_loader,
#         loss_module=loss_module,
#         device=accelerator.device,
#         accelerator=accelerator,
#         dataset=test_loader.dataset
#     )

#     test_loss = torch.tensor(test_loss, device=accelerator.device)
#     test_loss = accelerator.gather_for_metrics(test_loss).mean().item()

#     accelerator.wait_for_everyone()

#     gathered_test_metrics = {}
#     for k, v in test_metrics.items():
#         if not isinstance(v, torch.Tensor):
#             v = torch.tensor(v, dtype=torch.float32, device=accelerator.device)
#         gathered_test_metrics[k] = accelerator.gather_for_metrics(v).mean().item()
#     test_metrics = gathered_test_metrics

#     # ==================== Stage 2 Test UMAP ====================
#     if accelerator.is_main_process:
#         print("\n" + "="*60)
#         print("🎨 Generating Stage 2 Test UMAP")
#         print("="*60)

#         if len(test_window_embeddings) > 0 and len(test_window_labels) > 0:
#             test_embeddings_cat = torch.cat(test_window_embeddings, dim=0)
#             test_labels_cat = torch.cat(test_window_labels, dim=0)
#             test_preds_cat = torch.cat(test_window_preds, dim=0) if test_window_preds is not None else None

#             total_samples = test_embeddings_cat.size(0)
#             print(f"Collected {total_samples} test embeddings")

#             # Create UMAP save directory
#             test_umap_path = f'{args.umap_save_dir}/stage2_test_final.png'
#             os.makedirs(os.path.dirname(test_umap_path), exist_ok=True)

#             try:
#                 plot_umap_2d(
#                     args=args,
#                     window_embeddings_list=[test_embeddings_cat],
#                     window_labels_list=[test_labels_cat],
#                     window_pred_list=[test_preds_cat] if test_preds_cat is not None else None,
#                     epoch="test_final",
#                     prefix="stage2_test",
#                     pca_model=None
#                 )
#                 print(f"✅ Stage 2 Test UMAP saved to: {test_umap_path}")
#             except Exception as e:
#                 print(f"⚠️ Test UMAP generation failed: {e}")
#         else:
#             print("⚠️ No valid test embeddings collected for UMAP")

#         print("="*60 + "\n")

#     if accelerator.is_main_process:
#         wandb.run.summary['test_precision'] = test_metrics['precision_macro']
#         wandb.run.summary['test_recall'] = test_metrics['recall_macro']
#         wandb.run.summary['test_auroc'] = test_metrics['auroc_macro']
#         wandb.run.summary['test_auprc'] = test_metrics['auprc_macro']
        
#         for k, v in test_metrics.items():
#             if k.endswith('_accuracy'):
#                 wandb.run.summary[f"test_{k}"] = v

#         wandb.finish()

#     print("\n" + "="*80)
#     print("✅ STAGE 2 COMPLETED!")
#     print(f"   Mode: {mode}")
#     print(f"   Best Val AUROC: {best_auroc:.4f}")
#     print(f"   Best Val AUPRC: {best_auprc:.4f}")
#     print()
#     print(f"   Test Precision: {test_metrics['precision_macro']:.4f}")
#     print(f"   Test Recall: {test_metrics['recall_macro']:.4f}")
#     print(f"   Test AUROC: {test_metrics['auroc_macro']:.4f}")
#     print(f"   Test AUPRC: {test_metrics['auprc_macro']:.4f}")
#     print()
#     print(f"   Test Negative Acc (Class 0): {test_metrics.get('class_0_accuracy'):.2f}%")
#     print(f"   Test Non-cardiogenic Acc (Class 1): {test_metrics.get('class_1_accuracy'):.2f}%")
#     print(f"   Test Cardiogenic Acc (Class 2): {test_metrics.get('class_2_accuracy'):.2f}%")

#     results = {
#         'val_auroc': best_auroc,
#         'val_auprc': best_auprc,
#         'test_auroc': test_metrics['auroc_macro'],
#         'test_auprc': test_metrics['auprc_macro']
#     }

#     return results