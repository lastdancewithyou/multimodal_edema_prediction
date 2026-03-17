import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    # wandb
    parser.add_argument('--project_name', type=str, default="Multi_task", help="Wandb project name")
    parser.add_argument('--experiment_id', type=str, default="29", help="Experiment ID")
    parser.add_argument('--run_name', type=str, default=None)

    # Modality Selection
    parser.add_argument("--disable_cxr", type=bool, default=False, help="이미지 모달리티 활성화 여부")
    parser.add_argument("--disable_txt", type=bool, default=False, help="텍스트 모달리티 활성화 여부")
    parser.add_argument('--use_demographic', type=bool, default=True, help='Use demographic information')

    parser.add_argument('--img_to_3ch', type=bool, default=False, help='Convert grayscale to 3-channel by repeating channel 1 (for CXFormer/ResNet)')
    # parser.add_argument('--use_img_augmentation', type=bool, default=False, help='Use per-window image augmentation for diversity') # False - v2 / True - v3

    # dataset & sampler
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)

    parser.add_argument('--train_ratio', type=float, default=0.75, help='Train dataset ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation dataset ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes') # Negative, Non-cardio, Cardio

    # Early prediction
    parser.add_argument('--prediction_horizon', type=int, default=8, help='Hours ahead to predict for early prediction')
    parser.add_argument('--window_size', type=int, default=24, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=1, help='Sliding window moving stride')

    #################################### Multi-task Learning ####################################
    # Binary Cross-Entropy
    parser.add_argument('--use_bce', type=bool, default=True, help='Enable BCE loss for edema detection')
    parser.add_argument('--bce_weight', type=float, default=1.0, help='Binary cross-entropy loss weight')

    # Cross-Entropy Loss
    parser.add_argument('--use_ce', type=bool, default=True, help='Enable cross-entropy loss for classification')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='Cross-entropy loss weight')

    # Temporal InfoNCE Loss
    parser.add_argument('--use_temporal_ucl', type=bool, default=False)
    parser.add_argument('--ucl_weight', type=float, default=0.1, help='Unsupervised contrastive loss weight')
    parser.add_argument('--ucl_beta', type=float, default=1.0, help='Beta for unsupervised contrastive loss')
    parser.add_argument('--ucl_temperature', type=float, default=0.1, help='Temperature for unsupervised contrastive loss')
    
    # SupCon Loss
    parser.add_argument('--use_supcon', type=bool, default=False)
    parser.add_argument('--scl_weight', type=float, default=0.1, help='Supervised contrastive loss weight')
    parser.add_argument('--scl_temperature', type=float, default=0.3, help='Temperature for supervised contrastive loss')

    # InFoNCE Loss
    parser.add_argument('--use_infonce', type=bool, default=False)
    parser.add_argument('--infonce_weight', type=float, default=0.1, help='InfoNCE loss weight')
    parser.add_argument('--infonce_temperature', type=float, default=0.1, help='Temperature for InfoNCE loss')
    ###############################################################################################

    # Single-Stage Training
    parser.add_argument('--single_stage_epochs', type=int, default=30, help='Number of epochs for single-stage training')
    parser.add_argument('--single_learning_rate', type=float, default=1e-4, help='Learning rate at single training stage')
    parser.add_argument('--single_patience', type=int, default=5, help='Early stopping patience') # Stage 1 early stopping patience

    # Gradient Accumulation for Multi-GPU
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation steps - 멀티 GPU 안정성 향상')

    # model
    ## time-series modal
    parser.add_argument('--ts_encoder_input_size', type=int, default=29, help="Input size for TF") # default=Variable 28 + observed_mask 28
    parser.add_argument('--ts_encoder_hidden_size', type=int, default=512, help="Hidden size for TF")
    parser.add_argument('--ts_encoder_num_layers', type=int, default=2, help="The number of layers in TF")

    ## cxr modal
    parser.add_argument('--cxr_input_size', type=int, default=224, help='CXR input image size')

    ## text modal
    parser.add_argument('--token_max_length', type=int, default=512, help="max length of tokens")

    # cross attention
    parser.add_argument('--num_latents', type=int, default=6, help='number of rows in latent matrix of cross attention module')
    parser.add_argument('--num_iterations', type=int, default=2, help='cross attention iteration number')

    # Visualization
    ## UMAP
    parser.add_argument('--pca_components', type=int, default=32, help='Number of PCA components (recommended: projection_dim // 4)')
    parser.add_argument('--umap_n_neighbors', type=int, default=10, help='UMAP n_neighbors')
    parser.add_argument('--umap_min_dist', type=float, default=0.2, help='UMAP min_dist')
    parser.add_argument('--umap_metric', type=str, default='cosine', help='UMAP metric (default: euclidean)')
    parser.add_argument('--umap_save_dir', type=str, default=None, help='UMAP save directory')

    ## Label Smoothing
    parser.add_argument('--use_label_smooth', type=bool, default=True, help='Use label smoothing for regularization')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')

    ## save_path
    parser.add_argument('--best_model_dir', type=str, default=None, help='Directory to save best model checkpoint')

    ##################################################################################################################################
    # Two-Stage Training Controller
    parser.add_argument('--use_two_stage', type=bool, default=False, help='Use two-stage training: contrastive pretraining + classification')
    parser.add_argument('--stage1_only', type=bool, default=False, help='Run only stage 1 (contrastive pretraining)')
    parser.add_argument('--stage2_only', type=bool, default=False, help='Run only stage 2 (classification from pretrained)')
    parser.add_argument('--stage1_epochs', type=int, default=40, help='Number of epochs for stage 1 (contrastive pretraining)')
    parser.add_argument('--stage2_epochs', type=int, default=30, help='Number of epochs for stage 2 (classification)')
    parser.add_argument('--stage1_lr', type=float, default=1e-4, help='Learning rate for stage 1')
    parser.add_argument('--stage2_lr', type=float, default=5e-5, help='Learning rate for stage 2')
    parser.add_argument('--save_stage1_model', type=bool, default=True, help='Save stage 1 model checkpoint')
    parser.add_argument('--stage1_model_path', type=str, 
                        default=None,
                        # default="/home/DAHS1/gangmin/my_research/src/output/stage1_models/experiment_33_stage1.pth", 
                        help='Path to save/load stage 1 model'
                        )

    # # Stage 2 Training Strategy: Linear Probing vs Fine-tuning
    # parser.add_argument('--freeze_encoder_stage2', type=bool, default=False, help='Freeze encoder in stage 2 (Linear Probing). If not set, fine-tune encoder.')
    # parser.add_argument('--stage2_patience', type=int, default=5, help='Early stopping patience for stage 2')
    ##################################################################################################################################

    args = parser.parse_args([])

    if args.use_two_stage:
        args.wandb_run_name = f"{args.experiment_id}: [GPU 0] temp=0.7/Optimal_hyperparam_search/Img_Text_Tuning/Fine-tuning"
    # single stage
    else:
        args.wandb_run_name = f"{args.experiment_id}: True_Normalized_img(224by224)/Performance_Restoration"

    # ===================================================================================================

    if args.run_name is None:
        args.run_name = f"experiment_{args.experiment_id}"

    if args.best_model_dir is None:
        args.best_model_dir = f'./output/checkpoints/{args.run_name}'

    if args.umap_save_dir is None:
        args.umap_save_dir = f'./output/umap/{args.run_name}'
    
    if args.stage1_model_path is None:
        args.stage1_model_path = f'./output/stage1_models/{args.run_name}_stage1.pth'

    # Stage 2 단독 학습시에만 실제 저장된 모델 경로로 변경
    if args.stage2_only:
        actual_stage1_path = f'./output/stage1_models/experiment_{args.experiment_id}_stage1.pth'
        if os.path.exists(actual_stage1_path):
            args.stage1_model_path = actual_stage1_path
            print(f"✅ [Stage 2 Only] Using existing Stage 1 model: {actual_stage1_path}")
        else:
            print(f"⚠️  [Stage 2 Only] Stage 1 model not found at: {actual_stage1_path}")
            print(f"    Will try default path: {args.stage1_model_path}")

    # if args.target_path is None and args.use_target_supcon:
    #     args.target_path = f'./output/targets/optimal_target_{args.num_classes}_{args.head_hidden_dim2}.npy'

    return args