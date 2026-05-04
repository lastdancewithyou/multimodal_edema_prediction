import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_arguments():
    parser = argparse.ArgumentParser()

    # wandb
    parser.add_argument('--project_name', type=str, default="Deciding on Learning Methods", help="Wandb project name")
    parser.add_argument('--experiment_id', type=str, default="10", help="Experiment ID")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--wandb_on', type=bool, default=False, help='Enable Weights & Biases logging')

    # Modality Selection (True가 해당 모달리티 사용 중지)
    parser.add_argument("--disable_prompt", type=bool, default=True, help="prompt 활성화 여부")
    parser.add_argument('--use_clinical_prompt', type=bool, default=True, help='Use demographic information')
    parser.add_argument("--disable_cxr", type=bool, default=False, help="이미지 모달리티 활성화 여부")
    parser.add_argument("--disable_txt", type=bool, default=False, help="텍스트 모달리티 활성화 여부")

    parser.add_argument('--img_to_3ch', type=bool, default=False, help='Convert grayscale to 3-channel by repeating channel 1 (for CXFormer/ResNet)')

    # Fusion architecture ablation
    parser.add_argument('--use_segmented_attention', type=bool, default=True, help='Use segmented temporal attention (True) vs full global attention (False) for TS fusion')

    # dataset & sampler
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')

    # Early prediction
    parser.add_argument('--prediction_horizon', type=int, default=0, help='Hours ahead to predict for early prediction')
    parser.add_argument('--window_size', type=int, default=24, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=1, help='Sliding window moving stride')

    #################################### Multi-task Learning ####################################
    # Binary Cross-Entropy Loss
    parser.add_argument('--bce_weight', type=float, default=0.6, help='Binary cross-entropy loss weight')

    # Cross-Entropy Loss
    parser.add_argument('--ce_weight', type=float, default=0.4, help='Cross-entropy loss weight')

    # Virtual Prompt Alignment Loss
    parser.add_argument('--align_weight', type=float, default=0.1, help='Virtual prompt alignment loss weight')

    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='Learning rate for encoder')
    parser.add_argument('--head_lr', type=float, default=5e-4, help='Learning rate for task readout heads')

    # model
    ## time-series modal
    parser.add_argument('--ts_encoder_input_size', type=int, default=47, help="Input size for TF")
    parser.add_argument('--ts_encoder_hidden_size', type=int, default=512, help="Hidden size for TF")
    parser.add_argument('--ts_encoder_num_layers', type=int, default=3, help="The number of layers in TF")

    ## cxr modal
    parser.add_argument('--cxr_input_size', type=int, default=224, help='CXR input image size')

    ## text modal
    parser.add_argument('--token_max_length', type=int, default=512, help="max length of tokens")

    # cross attention
    parser.add_argument('--num_latents', type=int, default=6, help='number of rows in latent matrix of cross attention module')
    parser.add_argument('--num_iterations', type=int, default=2, help='cross attention iteration number')

    # DLinear
    # parser.add_argument('--seq_len', type=int, default=24, help="window size")
    # parser.add_argument('--enc_in', type=int, default=58, help="입력 변수의 총 개수")
    # parser.add_argument('--individual', type=bool, default=False, help="변수의 독립 가중치 여부")

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

    args.wandb_run_name = f"{args.experiment_id}: Pre-trained_ts_enc/TS+CXR_Text_align_scheduled_routing"

    if args.run_name is None:
        args.run_name = f"experiment_{args.experiment_id}"

    if args.best_model_dir is None:
        args.best_model_dir = f'./output/checkpoints/{args.run_name}'

    if args.umap_save_dir is None:
        args.umap_save_dir = f'./output/umap/{args.run_name}'

    return args