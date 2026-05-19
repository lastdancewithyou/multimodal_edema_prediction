import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_arguments():
    parser = argparse.ArgumentParser()

    # wandb
    parser.add_argument('--project_name', type=str, default="Soft_Laebeling", help="Wandb project name")
    parser.add_argument('--experiment_id', type=str, default="40", help="Experiment ID")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--wandb_on', type=bool, default=True, help='Enable Weights & Biases logging')

    # Modality Selection (True가 해당 모달리티 사용 중지)
    parser.add_argument("--disable_cxr", type=bool, default=False, help="이미지 모달리티 활성화 여부")
    parser.add_argument("--disable_txt", type=bool, default=True, help="텍스트 모달리티 활성화 여부")

    # Fusion architecture ablation
    parser.add_argument('--use_segmented_attention', type=bool, default=True, help='Use segmented temporal attention (True) vs full global attention (False) for TS fusion')

    # dataset & sampler
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')

    # Early prediction
    parser.add_argument('--prediction_horizon', type=int, default=0, help='Hours ahead to predict for early prediction')
    parser.add_argument('--window_size', type=int, default=24, help='Sliding window size')
    parser.add_argument('--train_stride', type=int, default=4, help='Sliding window moving stride')
    parser.add_argument('--eval_stride', type=int, default=1, help='Sliding window moving stride')

    #################################### Multi-task Learning ####################################
    # Binary Cross-Entropy Loss
    parser.add_argument('--bce_weight', type=float, default=1.0, help='Binary cross-entropy loss weight')

    # Sub-task BCE (Cardiomegaly / Pneumonia)
    parser.add_argument('--cardio_weight', type=float, default=0.0, help='Cardiomegaly sub-task BCE weight')
    parser.add_argument('--pneumo_weight', type=float, default=0.0, help='Pneumonia sub-task BCE weight')
    parser.add_argument('--info_weight', type=float, default=0.1, help='Weight for Text Alignment InfoNCE loss')

    parser.add_argument('--unsupervised_weight', type=float, default=0.0, help='Unsupervised contrastive loss weight (0 to disable)')
    parser.add_argument('--contrast_temperature', type=float, default=0.1, help='Contrastive temperature')

    # Augmentation
    # parser.add_argument('--feature_mask_prob', type=float, default=0.2, help='Per-cell masking probability for two-view augmentation (applies only where flag==1)')

    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--encoder_lr', type=float, default=5e-5, help='Learning rate for encoder')
    parser.add_argument('--head_lr', type=float, default=1e-4, help='Learning rate for task readout heads')

    # model
    ## time-series modal
    parser.add_argument('--ts_encoder_input_size', type=int, default=45, help="Input size for TF")
    parser.add_argument('--ts_encoder_hidden_size', type=int, default=256, help="Hidden size for TF")
    parser.add_argument('--ts_encoder_num_layers', type=int, default=2, help="The number of layers in TF")

    ## cxr modal (사전 임베딩 차원)
    parser.add_argument('--cxr_input_size', type=int, default=224, help='CXR input image size (legacy, unused in pre-embedding pipeline)')
    parser.add_argument('--img_emb_dim', type=int, default=768, help='Pre-computed RadDino embedding dim')
    parser.add_argument('--seg_emb_dim', type=int, default=32, help='Pre-computed HybridGNet segment embedding dim')
    parser.add_argument('--img_shared_dim', type=int, default=256, help='Shared latent dim for image-segment cross-attention')
    # (Future) parser.add_argument('--lesion_emb_dim', type=int, default=512, help='Pre-computed lesion embedding dim')

    ## text modal (사전 임베딩 차원)
    # parser.add_argument('--token_max_length', type=int, default=512, help="max length of tokens (legacy, unused)")
    parser.add_argument('--text_emb_dim', type=int, default=768, help='Pre-computed text embedding dim (Bio_ClinicalBERT CLS)')

    # cross attention
    parser.add_argument('--num_latents', type=int, default=6, help='number of rows in latent matrix of cross attention module')
    parser.add_argument('--num_iterations', type=int, default=1, help='cross attention iteration number')

    # Visualization
    ## UMAP
    parser.add_argument('--pca_components', type=int, default=32, help='Number of PCA components (recommended: projection_dim // 4)')
    parser.add_argument('--umap_n_neighbors', type=int, default=10, help='UMAP n_neighbors')
    parser.add_argument('--umap_min_dist', type=float, default=0.2, help='UMAP min_dist')
    parser.add_argument('--umap_metric', type=str, default='cosine', help='UMAP metric (default: euclidean)')
    parser.add_argument('--umap_save_dir', type=str, default=None, help='UMAP save directory')

    ## Label Smoothing
    parser.add_argument('--use_label_smooth', type=bool, default=True, help='Use label smoothing for regularization')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')

    ## save_path
    parser.add_argument('--best_model_dir', type=str, default=None, help='Directory to save best model checkpoint')
    ##################################################################################################################################

    args = parser.parse_args([])

    args.wandb_run_name = f"{args.experiment_id}: [Single GPU] Reform_V5_iter=1"

    if args.run_name is None:
        args.run_name = f"experiment_{args.experiment_id}"

    if args.best_model_dir is None:
        args.best_model_dir = f'./output/checkpoints/{args.run_name}'

    if args.umap_save_dir is None:
        args.umap_save_dir = f'./output/umap/{args.run_name}'

    return args