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
from training.evaluator import test, validate_multitask
from models.main_architecture import MultiModalEncoder, MultiModalMultiTaskModel
from loss.losses import MultiModalLoss
from loss.target_metrics import visualize_target_supcon
from utils.utils import timer, plot_latent_time_attention, Earlystopping
from analysis.umap_multitask import plot_multitask_umap


##################################################################################################
# Model Training Control Center
##################################################################################################
def train_single_stage_multimodal_model(ts_df, img_df, text_df, clinical_prompt_df, args):
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
    print(f"      Regression Loss Scheduled for introduction")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("="*80 + "\n")

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
        print(f"\n{'='*60}")
        print(f"🚀 [GPU Configuration]")
        print(f"   Number of GPUs: {accelerator.num_processes}")
        print(f"   Device: {device}")
        print(f"   Mixed Precision: bf16")
        print(f"{'='*60}\n")

        if wandb_on:
            wandb.init(
                project=args.project_name,
                name=args.wandb_run_name,
                config=vars(args),
                tags=["single_stage", "multi_task"]
            )

    # DataLoader
    with timer("Dataset Loading"):
        train_loader, val_loader, test_loader, train_sampler = get_dataloaders(ts_df, img_df, text_df, clinical_prompt_df, args, accelerator)

    # Create model on CPU - Accelerator will handle device placement
    encoder = MultiModalEncoder(args, disable_cxr=args.disable_cxr, disable_txt=args.disable_txt, disable_prompt=args.disable_prompt)
    model = MultiModalMultiTaskModel(encoder)
    accelerator.print(f"[모달리티 상태] CXR 사용: {not model.encoder.disable_cxr}, TEXT 사용: {not model.encoder.disable_txt}, PROMPT 사용: {not model.encoder.disable_prompt}")

    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    print(f"\n{'='*80}")
    print(f"📊 DETAILED MODEL PARAMETER BREAKDOWN")
    print(f"{'='*80}")

    # 1. Encoder Components
    print(f"\n🔹 [1] Modality-Specific Encoders")
    ts_total, ts_trainable = count_params(model.encoder.ts_encoder)
    print(f"   Time-Series Encoder:")
    print(f"      Total: {ts_total:>12,} | Trainable: {ts_trainable:>12,} ({100*ts_trainable/ts_total:.1f}%)")

    img_total, img_trainable = count_params(model.encoder.img_encoder)
    print(f"   Image Encoder:")
    print(f"      Total: {img_total:>12,} | Trainable: {img_trainable:>12,} ({100*img_trainable/img_total:.1f}%)")

    txt_total, txt_trainable = count_params(model.encoder.text_encoder)
    print(f"   Text Encoder:")
    print(f"      Total: {txt_total:>12,} | Trainable: {txt_trainable:>12,} ({100*txt_trainable/txt_total:.1f}%)")

    # 2. Fusion Module
    print(f"\n🔹 [2] Multimodal Fusion Module")
    fusion_total, fusion_trainable = count_params(model.encoder.ts_centric_fusion)
    print(f"   TS-Centric Cross-Attention:")
    print(f"      Total: {fusion_total:>12,} | Trainable: {fusion_trainable:>12,} ({100*fusion_trainable/fusion_total:.1f}%)")

    # Detailed fusion components
    latent_params = model.encoder.ts_centric_fusion.latent_init.numel()
    # prompt_ca_total, prompt_ca_trainable = count_params(model.encoder.ts_centric_fusion.ctx_cross_attn)
    ts_ca_total, ts_ca_trainable = count_params(model.encoder.ts_centric_fusion.ts_cross_attn)
    img_ca_total, img_ca_trainable = count_params(model.encoder.ts_centric_fusion.img_cross_attn)
    txt_ca_total, txt_ca_trainable = count_params(model.encoder.ts_centric_fusion.text_cross_attn)
    tsmixer_total, tsmixer_trainable = count_params(model.encoder.ts_centric_fusion.tsmixer)

    print(f"      ├─ Latent Array Init:     {latent_params:>12,}")
    # print(f"      ├─ Prompt Cross-Attn:     {prompt_ca_trainable:>12,}")
    print(f"      ├─ TS Cross-Attn:         {ts_ca_trainable:>12,}")
    print(f"      ├─ IMG Cross-Attn:        {img_ca_trainable:>12,}")
    print(f"      ├─ TXT Cross-Attn:        {txt_ca_trainable:>12,}")
    print(f"      └─ TSMixer:               {tsmixer_trainable:>12,}")

    # 3. Attention Pooling
    print(f"\n🔹 [3] Attention Pooling")
    pool_total, pool_trainable = count_params(model.encoder.attention_pooling)
    print(f"   Attention Pooling Layer:")
    print(f"      Total: {pool_total:>12,} | Trainable: {pool_trainable:>12,} ({100*pool_trainable/pool_total:.1f}%)")

    # 4. Task-Specific Heads
    print(f"\n🔹 [4] Task-Specific Prediction Heads")
    edema_total, edema_trainable = count_params(model.edema_classifier)
    print(f"   Binary Classifier (Edema Detection):")
    print(f"      Total: {edema_total:>12,} | Trainable: {edema_trainable:>12,} ({100*edema_trainable/edema_total:.1f}%)")

    subtype_total, subtype_trainable = count_params(model.subtype_classifier)
    print(f"   Subtype Classifier (Cardiogenic/Non-cardiogenic):")
    print(f"      Total: {subtype_total:>12,} | Trainable: {subtype_trainable:>12,} ({100*subtype_trainable/subtype_total:.1f}%)")

    regression_total, regression_trainable = count_params(model.regression_head)
    print(f"   Regression Head:")
    print(f"      Total: {regression_total:>12,} | Trainable: {regression_trainable:>12,} ({100*regression_trainable/regression_total:.1f}%)")

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

    # Loss Module
    loss_module = MultiModalLoss(args, class_weights=None)

    # ==================== Optimizer (5-Tier Differential Learning Rate) ====================
    base_lr = args.single_learning_rate

    lora_params, cross_attn_params, core_params = [], [], []

    # Edema classifier
    edema_params = list(model.edema_classifier.parameters())

    # Subtype classifier
    subtype_params = list(model.subtype_classifier.parameters())

    # Regression head
    regression_params = list(model.regression_head.parameters())

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip heads
        if 'subtype_classifier' in name or 'edema_classifier' in name or 'regression_head' in name:
            continue

        # Cross-Attention parameters
        if 'cross_attn' in name or 'norm_cross' in name or 'ls_cross' in name:
            cross_attn_params.append(param)

        # LoRA parameters (CXFormer + BioClinicalBERT)
        elif ('lora_A' in name or 'lora_B' in name):
            lora_params.append(param)

        # TS Encoder, Fusion, Pooling
        else:
            core_params.append(param)

    param_groups = [
        # LoRA
        {
            'params': lora_params,
            'lr': base_lr * 0.5,
            'weight_decay': 1e-4
        },
        # Core Trainable Components
        {
            'params': core_params,
            'lr': base_lr * 1.0,
            'weight_decay': 1e-4
        },
        # Edema Classifier 
        {
            'params': edema_params,
            'lr': base_lr * 1.0,
            'weight_decay': 1e-4
        },
        # Subtype Classifier
        {
            'params': subtype_params,
            'lr': base_lr * 0.7,
            'weight_decay': 1e-3
        },
        # Regression
        {
            'params': regression_params,
            'lr': base_lr * 0.8,
            'weight_decay': 1e-3
        }
    ]

    optimizer = torch.optim.AdamW(param_groups)

    # Prepare model, optimizer, loss_module, AND dataloaders with Accelerator
    model, optimizer, loss_module, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, loss_module, train_loader, val_loader, test_loader
    )

    # ==================== Scheduler Configuration ====================
    warmup_epochs = 3

    # Warmup scheduler
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )

    # ReduceLROnPlateau: Validation loss 기반 adaptive lr 조절
    main_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Minimize validation loss
        factor=0.5,           # lr을 절반으로 감소
        patience=5,           # 5 epoch 동안 개선 없으면 lr 감소
        min_lr=5e-6           # 최소 lr 제한
    )

    print(f"\n📅 [Scheduler Configuration]")
    print(f"   Warmup Epochs: {warmup_epochs}")
    print(f"   Warmup Start Factor: 0.1")
    print(f"   Main Scheduler: ReduceLROnPlateau")
    print(f"   ├─ Mode: min (validation loss)")
    print(f"   ├─ Factor: 0.5")
    print(f"   ├─ Patience: 5 epochs")
    print(f"   └─ Min LR: 1e-6")

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
        mse_sum = torch.zeros(1, device=device, dtype=torch.float32)
        mse_count = torch.zeros(1, device=device, dtype=torch.float32)

        train_edema_preds_list = []
        train_edema_labels_list = []
        train_subtype_preds_list = []
        train_subtype_labels_list = []

        optimizer.zero_grad(set_to_none=True)
        train_sampler.set_epoch(epoch)

        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch+1}] Learning Rate: {current_lr:.2e}")

        ####################################################################################
        # Dynamic loss weight adjustment to prevent Level 2 task's overfitting
        if epoch < 8:
            dynamic_bce_weight = args.bce_weight
            dynamic_ce_weight = args.ce_weight
            dynamic_mse_weight = args.mse_weight
        else:
            dynamic_bce_weight = args.bce_weight * 1.5  # Increase Level 1 focus (Edema detection)
            dynamic_ce_weight = args.ce_weight * 0.3    # Reduce Level 2 (prevent overfitting)
            dynamic_mse_weight = args.mse_weight
        ####################################################################################

        if accelerator.is_main_process and epoch in [0, 10]:
            print(f"   📊 Loss Weights - BCE: {dynamic_bce_weight:.2f}, CE: {dynamic_ce_weight:.2f}, MSE: {dynamic_mse_weight:.2f}")

        # Training
        for batch_idx, batch in enumerate(tqdm(train_loader, total=len(train_loader),
                                                desc=f"[Rank {local_rank}] Epoch {epoch+1}/{args.single_stage_epochs}",
                                                position=local_rank, leave=True, dynamic_ncols=True)):

            # ===== Spatial Region Weights 로깅 (100번째 배치마다) =====
            if batch_idx % 100 == 0 and accelerator.is_main_process:
                try:
                    unwrapped_model = accelerator.unwrap_model(model)
                    if hasattr(unwrapped_model, 'encoder') and hasattr(unwrapped_model.encoder, 'spatial_pooling') and hasattr(unwrapped_model.encoder.spatial_pooling, 'region_logits'):

                        # Region weights 계산
                        w = unwrapped_model.encoder.spatial_pooling.region_logits.softmax(dim=0)
                        labels = ["cls", "center", "UL", "UR", "LL", "LR"]

                        if wandb.run is not None:
                            region_weights = {
                                f"spatial_weights/{label}": val.item()
                                for label, val in zip(labels, w)
                            }
                            global_step = epoch * len(train_loader) + batch_idx
                            wandb.log(region_weights, step=global_step)

                        weights_str = ", ".join([f"{l}={v.item():.3f}" for l, v in zip(labels, w)])
                        print(f"\n[Epoch {epoch+1}, Batch {batch_idx}] Spatial Weights: {weights_str}\n")

                except Exception:
                    pass

            with accelerator.accumulate(model):
                total_batch_loss, batch_bce, batch_ce, batch_mse, batch_outputs, batch_counts = train_batch(
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
                    bce_weight=dynamic_bce_weight,
                    ce_weight=dynamic_ce_weight,
                    mse_weight=dynamic_mse_weight,
                )

                accelerator.backward(total_batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Get loss-specific sample counts
            bce_ct_local = torch.as_tensor(batch_counts['bce_count'], device=device, dtype=torch.float32)
            ce_ct_local = torch.as_tensor(batch_counts['ce_count'], device=device, dtype=torch.float32)
            mse_ct_local = torch.as_tensor(batch_counts['mse_count'], device=device, dtype=torch.float32)

            # Accumulate losses weighted by their actual sample counts
            bce_sum += torch.as_tensor(batch_bce, device=device, dtype=torch.float32) * bce_ct_local
            bce_count += bce_ct_local
            ce_sum += torch.as_tensor(batch_ce, device=device, dtype=torch.float32) * ce_ct_local
            ce_count += ce_ct_local
            mse_sum += torch.as_tensor(batch_mse, device=device, dtype=torch.float32) * mse_ct_local
            mse_count += mse_ct_local

            with torch.no_grad():
                edema_logits = batch_outputs['edema_logits'].squeeze(-1)   # [B, W]
                subtype_logits = batch_outputs['subtype_logits']           # [B, W, 2]
                edema_labels = batch_outputs['edema_labels']               # [B, W]
                subtype_labels = batch_outputs['subtype_labels']           # [B, W]
                window_mask = batch['window_mask']                         # [B, W]

                valid_mask = window_mask.bool()                            # [B, W]

                p_pos = torch.sigmoid(edema_logits)                        # [B, W] P(edema=1)
                p_sub = torch.softmax(subtype_logits, dim=-1)              # [B, W, 2] P(NCPE|pos), P(CPE|pos)

                p_pos_valid = p_pos[valid_mask]                            # [Nwin]
                p_sub_valid = p_sub[valid_mask]                            # [Nwin, 2]
                edema_valid = edema_labels[valid_mask]                     # [Nwin]
                subtype_valid = subtype_labels[valid_mask]                 # [Nwin]

                train_edema_preds_list.append(p_pos_valid.detach().cpu())
                train_subtype_preds_list.append(p_sub_valid.detach().cpu())
                train_edema_labels_list.append(edema_valid.detach().cpu())
                train_subtype_labels_list.append(subtype_valid.detach().cpu())
        
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bce_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(bce_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(ce_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ce_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse_count, op=dist.ReduceOp.SUM)

        bce_avg = (bce_sum / (bce_count + 1e-8)).item()
        ce_avg = (ce_sum / (ce_count + 1e-8)).item()
        mse_avg = (mse_sum / (mse_count + 1e-8)).item()

        bce_contrib = args.bce_weight * bce_avg
        ce_contrib = args.ce_weight * ce_avg
        mse_contrib = args.mse_weight * mse_avg
        avg_total_loss = bce_contrib + ce_contrib + mse_contrib

        if epoch < warmup_epochs:
            warmup_scheduler.step()

        # Gather train predictions from all GPUs
        if accelerator.num_processes > 1:
            local_preds = {
                'p_pos': [p.cpu() for p in train_edema_preds_list],
                'p_sub': [p.cpu() for p in train_subtype_preds_list],
                'edema': [e.cpu() for e in train_edema_labels_list],
                'subtype': [s.cpu() for s in train_subtype_labels_list]
            }

            # Gather to rank 0 only
            if accelerator.is_main_process:
                gathered_preds = [None] * accelerator.num_processes
                dist.gather_object(local_preds, gathered_preds, dst=0)

                # Combine all predictions from all GPUs
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
                # GPU 1: Send data to rank 0 and set to None
                dist.gather_object(local_preds, dst=0)
                p_pos_all = None
                p_sub_all = None
                edema_all = None
                subtype_all = None

            # Synchronize: GPU 1 waits for GPU 0 to finish metric computation
            dist.barrier()
        else:
            # Single GPU
            if len(train_edema_preds_list) > 0:
                p_pos_all = torch.cat(train_edema_preds_list, dim=0).numpy()
                p_sub_all = torch.cat(train_subtype_preds_list, dim=0).numpy()
                edema_all = torch.cat(train_edema_labels_list, dim=0).numpy()
                subtype_all = torch.cat(train_subtype_labels_list, dim=0).numpy()
            else:
                p_pos_all = None

        # Train metrics - Multi-task learning
        train_metrics = {}
        if accelerator.is_main_process and p_pos_all is not None and len(p_pos_all) > 0:

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
            """
            3-class GT is determined for samples:
            - edema==0 -> Neg(0)
            - edema==1 & subtype==0 -> NCPE(1)
            - edema==1 & subtype==1 -> CPE(2)
            """
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
            print(f"      MSE (Subtype): {mse_avg:.4f} → Weighted: {mse_contrib:.4f} (λ={args.mse_weight})")

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
        val_loss, val_bce_avg, val_ce_avg, val_mse_avg, val_metrics = validate_multitask(
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

        # ReduceLROnPlateau scheduler step
        if epoch >= warmup_epochs:
            main_scheduler.step(val_loss)
            if accelerator.is_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   📉 ReduceLROnPlateau - Current LR: {current_lr:.2e}, Val Loss: {val_loss:.4f}")

        # Early stopping based on Level 1 AUROC (Edema Detection)
        if accelerator.is_main_process and val_metrics:
            if val_metrics['level1_auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['level1_auroc']

            if val_metrics['level1_auprc'] > best_val_auprc:
                best_val_auprc = val_metrics['level1_auprc']

        # ==================== Multi-Task UMAP Visualization ====================
        # if accelerator.is_main_process and ((epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == args.single_stage_epochs):
        #     print("🖼️ Generating Training UMAP...")
        #     train_umap_dir = os.path.join(args.umap_save_dir, 'train')
        #     train_reducers = plot_multitask_umap(
        #         args=args,
        #         model=model,
        #         dataloader=train_loader,
        #         device=accelerator.device,
        #         accelerator=accelerator,
        #         dataset=train_loader.dataset,
        #         epoch=epoch+1,
        #         save_dir=train_umap_dir,
        #         max_samples=40000,
        #         umap_reducers=None
        #     )
        #     print("✅ Training UMAP completed!")

        #     # Validation UMAP (transform using train PCA + UMAP coordinate system)
        #     print("🖼️ Generating Validation UMAP...")
        #     val_umap_dir = os.path.join(args.umap_save_dir, 'val')
        #     plot_multitask_umap(
        #         args=args,
        #         model=model,
        #         dataloader=val_loader,
        #         device=accelerator.device,
        #         accelerator=accelerator,
        #         dataset=val_loader.dataset,
        #         epoch=epoch+1,
        #         save_dir=val_umap_dir,
        #         max_samples=None,
        #         umap_reducers=train_reducers  # Val mode: use train PCA + UMAP
        #     )
        #     print("✅ Validation UMAP completed!")

        if accelerator.is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                #################### Loss components ####################
                "train/total_loss": avg_total_loss,
                "train/bce_loss": bce_avg,
                "train/ce_loss": ce_avg,
                "train/mse_loss": mse_avg,

                "val/total_loss": val_loss,
                "val/bce_loss": val_bce_avg,
                "val/ce_loss": val_ce_avg,
                "val/mse_loss": val_mse_avg,

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

            if wandb_on:
                wandb.log(log_dict)

            # Early stopping
            print(f"DEBUG [Rank 0]: Before early_stopper")
            if early_stopper(args, best_val_auroc, model, epoch, accelerator):
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
        checkpoint = torch.load(actual_best_model_path, map_location=accelerator.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(state_dict)
    else:
        accelerator.print(f"⚠️ Best model not found, using current model state")

    # ==================== Test ====================
    test_loss, _, _, _, _, wandb_test_metrics = test(
        args=args,
        model=model,
        dataloader=test_loader,
        loss_module=loss_module,
        device=accelerator.device,
        accelerator=accelerator,
        dataset=test_loader.dataset
    )

    if wandb_on:
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
    # if accelerator.is_main_process:
    #     print("\n" + "="*80)
    #     print("🖼️ Generating Test UMAP...")
    #     test_umap_dir = os.path.join(args.umap_save_dir, 'test')
    #     plot_multitask_umap(
    #         args=args,
    #         model=model,
    #         dataloader=test_loader,
    #         device=accelerator.device,
    #         accelerator=accelerator,
    #         dataset=test_loader.dataset,
    #         epoch=epoch+1,
    #         save_dir=test_umap_dir,
    #         max_samples=None,
    #         umap_reducers=train_reducers  # Test mode: use train PCA + UMAP
    #     )
        # print("✅ Test UMAP completed!")

    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("✅ MULTI-TASK TRAINING COMPLETED!")
        print(f"   Best Val Level 1 AUROC (Edema Detection): {best_val_auroc:.4f}")
        print(f"   Best Val Level 1 AUPRC (Edema Detection): {best_val_auprc:.4f}\n")

        if wandb_test_metrics:
            print("   [Test Results - Hierarchical Metrics]")
            print(f"   Level 1 (Edema Detection):        AUROC={wandb_test_metrics['test/level1_auroc']:.4f}  AUPRC={wandb_test_metrics['test/level1_auprc']:.4f}")
            print(f"   Level 2 (Subtype Classification): AUROC={wandb_test_metrics['test/level2_auroc']:.4f}  AUPRC={wandb_test_metrics['test/level2_auprc']:.4f}")
            print(f"   Level 3 (3-class Combined):       AUROC={wandb_test_metrics['test/level3_auroc']:.4f}  AUPRC={wandb_test_metrics['test/level3_auprc']:.4f}")
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
            'test_level3_auroc': wandb_test_metrics['test/level3_auroc'],
            'test_level3_auprc': wandb_test_metrics['test/level3_auprc']
        }

    # Cleanup distributed resources
    if dist.is_initialized():
        dist.destroy_process_group()

    return results