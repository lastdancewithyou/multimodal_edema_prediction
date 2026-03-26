import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_arguments():
    parser = argparse.ArgumentParser()

    # wandb
    parser.add_argument('--project_name', type=str, default="Model Novelty development", help="Wandb project name")
    parser.add_argument('--experiment_id', type=str, default="23", help="Experiment ID")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--wandb_on', type=bool, default=True, help='Enable Weights & Biases logging')

    # Modality Selection (True가 해당 모달리티 사용 중지)
    parser.add_argument("--disable_prompt", type=bool, default=False, help="prompt 활성화 여부")
    parser.add_argument('--use_clinical_prompt', type=bool, default=True, help='Use demographic information')
    parser.add_argument("--disable_cxr", type=bool, default=False, help="이미지 모달리티 활성화 여부")
    parser.add_argument("--disable_txt", type=bool, default=False, help="텍스트 모달리티 활성화 여부")

    parser.add_argument('--img_to_3ch', type=bool, default=True, help='Convert grayscale to 3-channel by repeating channel 1 (for CXFormer/ResNet)')

    # Fusion architecture ablation
    parser.add_argument('--use_segmented_attention', type=bool, default=True,
                        help='Use segmented temporal attention (True) vs full global attention (False) for TS fusion')

    # dataset & sampler
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--train_ratio', type=float, default=0.75, help='Train dataset ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation dataset ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes') # Negative, Non-cardio, Cardio

    # Early prediction
    parser.add_argument('--prediction_horizon', type=int, default=8, help='Hours ahead to predict for early prediction')
    parser.add_argument('--window_size', type=int, default=24, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=1, help='Sliding window moving stride')
    parser.add_argument('--use_last_point_only', type=bool, default=False,
                        help='시계열 모델의 당위성을 입증하기 위한 가장 최근 데이터만으로 예측을 수행하는 컨트롤 파라미터')

    #################################### Multi-task Learning ####################################
    # Binary Cross-Entropy
    parser.add_argument('--use_bce', type=bool, default=True, help='Enable BCE loss for edema detection')
    parser.add_argument('--bce_weight', type=float, default=1.0, help='Binary cross-entropy loss weight')

    # Cross-Entropy Loss
    parser.add_argument('--use_ce', type=bool, default=True, help='Enable cross-entropy loss for classification')
    parser.add_argument('--ce_weight', type=float, default=0.4, help='Cross-entropy loss weight')

    # Regression Loss
    parser.add_argument('--mse_weight', type=float, default=0.1, help='MSE Loss weight')

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
    parser.add_argument('--single_stage_epochs', type=int, default=50, help='Number of epochs for single-stage training')
    parser.add_argument('--single_learning_rate', type=float, default=1e-4, help='Learning rate at single training stage')
    parser.add_argument('--single_patience', type=int, default=7, help='Early stopping patience') # Stage 1 early stopping patience

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

    args = parser.parse_args([])

    # Single-stage training configuration
    args.wandb_run_name = f"{args.experiment_id}: w/o_CXR_FFN+LN_Tuning/time2vec_ts/NoContextEmbed/Full_modal/Main_performance_to_Level1_AUROC/SpatialModeling/NoTxt"

    # ===================================================================================================

    if args.run_name is None:
        args.run_name = f"experiment_{args.experiment_id}"

    if args.best_model_dir is None:
        args.best_model_dir = f'./output/checkpoints/{args.run_name}'

    if args.umap_save_dir is None:
        args.umap_save_dir = f'./output/umap/{args.run_name}'


    return args