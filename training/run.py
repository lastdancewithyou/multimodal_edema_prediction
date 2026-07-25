import argparse

from dateutil import parser

PRETRAINED_PATH = "./models/PatchTST_self_supervised/saved_models/icu/masked_patchtst/based_model/"

def parse_arguments():
    parser = argparse.ArgumentParser()

    # wandb
    parser.add_argument('--project_name', type=str, default="Edema_Final", help="Wandb project name")
    parser.add_argument('--experiment_id', type=str, default="64", help="Experiment ID")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--wandb_on', type=bool, default=False, help='Enable Weights & Biases logging')

    # Modality Selection (True가 해당 모달리티 사용 중지)
    parser.add_argument("--disable_cxr", type=bool, default=False, help="이미지 모달리티 활성화 여부")
    parser.add_argument("--disable_txt", type=bool, default=False, help="텍스트 모달리티 활성화 여부")

    # Fusion architecture ablation
    parser.add_argument('--use_segmented_attention', type=bool, default=True, help='Use segmented temporal attention (True) vs full global attention (False) for TS fusion')
    parser.add_argument('--causal_attn_mask', type=int, default=0,
                        help='1: causal 적용 / 0: causal 미적용')

    # dataset & sampler
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')

    # Window
    parser.add_argument('--window_size', type=int, default=48)                                             # 48, 96
    parser.add_argument('--train_stride', type=int, default=8, help='Sliding window moving stride')        # 8, 16
    parser.add_argument('--eval_stride', type=int, default=1, help='Sliding window moving stride')

    # CXR Flag 1인 시점만 학습 수행
    parser.add_argument('--train_on_cxr_only', type=int, default=0,
                        help='1: train only on windows whose label slot has cxr_flag==1 '
                            '(align train/eval label source, much smaller train set). '
                            '0: all valid windows (soft-label trajectory training).')

    #################################### Training ####################################
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=6,
                        help='Linear LR warmup epochs before cosine annealing (0 to disable)')
    parser.add_argument('--warmup_start_factor', type=float, default=0.01,
                        help='Starting LR factor at epoch 0 during warmup (e.g. 0.01 = 1% of target LR)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')

    # Learning rate
    parser.add_argument('--encoder_lr', type=float, default=8e-5,
                        help='Learning rate for fusion (v7), img/text_pre_proj, token_type_emb')
    parser.add_argument('--ts_encoder_lr', type=float, default=5e-4,
                        help='Learning rate for PatchTST backbone (no-op when frozen)')
    parser.add_argument('--ts_proj_lr', type=float, default=5e-5,
                        help='Learning rate for ts_pre_proj (fresh-init 큰 bottleneck MLP, 별도 그룹)')
    parser.add_argument('--text_proj_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=1e-4,
                        help='Learning rate for priv_readout + deploy_readout heads')
    parser.add_argument('--priv_lr_boost', type=float, default=6.0,
                        help='Priv encoder LR multiplier (Separated 구조에서 gradient 부족 보상). '
                             'Shared 구조는 encoder가 ∇BCE_priv + ∇BCE_deploy 두 신호 받지만 '
                             'Separated에서 priv는 ∇BCE_priv 1개만 → LR×boost로 update 크기 보상. '
                             '2.0~4.0 권장.')

    # model
    ## time-series modal
    parser.add_argument('--ts_scaler_path', type=str, default='/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/patchtst_scaler.pkl')
    parser.add_argument('--ts_encoder_input_size', type=int, default=21, help="Input size for TF")
    parser.add_argument('--ts_encoder_hidden_size', type=int, default=256, help="Hidden size for TF")
    parser.add_argument('--ts_encoder_num_layers', type=int, default=3, help="The number of layers in TF")

    ## PatchTST SSL-pretrained TS encoder
    parser.add_argument('--patch_len', type=int, default=8,
                        help='PatchTST patch length (must match pretraining)')
    parser.add_argument('--stride',    type=int, default=4,
                        help='PatchTST patch stride (must match pretraining)')
    parser.add_argument('--patchtst_pretrained_path', type=str,
                        # default= PRETRAINED_PATH + 'patchtst_pretrained_cw48_patch8_stride4_epochs50_mask0.5_model1.pth',
                        default = "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/models/PatchTST_self_supervised/saved_models/patchtst_pretrained_multiple_length.pth"
    )
    parser.add_argument('--patchtst_freeze', type=int, default=1,
                        help='1: freeze PatchTST backbone, 0: finetune')
    # 학습시키니까 더 악영향 줌.
    parser.add_argument('--patchtst_unfreeze_last_n', type=int, default=0,
                        help='Partial unfreeze: 마지막 N개 transformer layer만 학습. '
                             '0=완전 freeze (기존). 1~2 권장 (text supervision 흡수용). '
                             'freeze=1 일 때만 적용됨.')
    
    parser.add_argument('--var_pool_type', type=str, default='linear',
                        choices=['mlp', 'linear'],
                        help='var_pool architecture: mlp (Linear→GELU→Dropout→Linear, ~1.5M) '
                            'or linear (single Linear, ~0.7M) for probing diagnostics')
    parser.add_argument('--patchtst_d_model', type=int, default=128,
                        help='PatchTST hidden dim (must match pretraining)')
    parser.add_argument('--patchtst_n_heads', type=int, default=16,
                        help='PatchTST attention heads (must match pretraining)')
    parser.add_argument('--patchtst_n_layers', type=int, default=3,
                        help='PatchTST encoder layers (must match pretraining)')

    ## cxr modal
    parser.add_argument('--img_emb_dim', type=int, default=768, help='Pre-computed RadDino Output embedding dim')

    ## text modal
    parser.add_argument('--text_emb_dim', type=int, default=768, help='Bio_ClinicalBERT hidden dim')
    # parser.add_argument('--text_model_path', type=str,
    #                     default='/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/models/bio_clinicalbert_safetensors',
    #                     help='Bio_ClinicalBERT pretrained 경로 (모델 내 frozen forward)')
    parser.add_argument('--text_model_path', type=str, default="NeuML/pubmedbert-base-embeddings") # PubMedBERT
    parser.add_argument('--text_max_tokens', type=int, default=128,
                        help='저장된 input_ids 길이 (전처리 시 max_length와 동일해야 함)')
    parser.add_argument('--text_attn_mask_prob', type=float, default=0.0,
                        help='Attention-based text masking 비율 (학습 중만). '
                             'BERT CLS attention 상위 K% 토큰을 downstream bank에서 제외. '
                             '0이면 비활성. 5~20% 권장.')

    # fusion type
    parser.add_argument('--fusion_type', type=str, default='cross_attn',
                        choices=['cross_attn', 'simple_concat'],
                        help='cross_attn: TimeSeriesCentricCrossAttention_v6 (현재 default). '
                             'simple_concat: per-modality pool → concat → MLP (Linear probe와 비슷한 단순 fusion).')

    # Fusion shared projection dim
    parser.add_argument('--align_d_model', type=int, default=256,
                        help='Shared projection dim for fusion (TS / CXR / Text → d).')

    # LUPI (Learning Under Privileged Information) — dual-stream distillation
    parser.add_argument('--lupi_fd_weight', type=float, default=0.5,
                        help='Feature distillation (fused, cosine distance): mean(1 - cos(fused_deploy, fused_priv))')
    
    parser.add_argument('--lupi_rd_weight', type=float, default=0.5,
                        help='Readout distillation 가중치 (RD 모드든 CMC 모드든 이 weight로 들어감)')

    parser.add_argument('--rd_mode', type=str, default='cmc', choices=['rd', 'cmc'],
                        help='rd: 기존 cosine + smooth_l1 absolute alignment. '
                             'cmc: cross-modal contrastive (InfoNCE between feat_deploy ↔ feat_priv.detach()).')

    parser.add_argument('--cmc_temperature', type=float, default=0.1,
                        help='CMC InfoNCE temperature τ (0.07~0.2 권장)')

    parser.add_argument('--cmc_symmetric', type=int, default=1,
                        help='1: 양방향 InfoNCE (deploy→priv + priv→deploy)/2.  0: deploy→priv 단방향.')
    
    parser.add_argument('--lupi_kd_weight', type=float, default=0.5,
                        help='Logit distillation (soft-target BCE on binary logits)')
    

    parser.add_argument('--lupi_cov_weight', type=float, default=0.01,
                        help='Covariance regularization on fused_deploy pooled features')

    parser.add_argument('--subtype_weight', type=float, default=0.5,
                        help='Masked soft cross-entropy on subtype head (deploy path); '
                             'targets = [p_mixed, p_ncpe, p_cpe], mask = subtype_mask')

    parser.add_argument('--priv_text_dropout_prob', type=float, default=0.0)

    # cross attention
    parser.add_argument('--num_latents', type=int, default=8, help='number of rows in latent matrix of cross attention module')
    parser.add_argument('--num_iteration', type=int, default=1, help='cross attention iteration number')

    # Visualization
    ## UMAP
    parser.add_argument('--pca_components', type=int, default=64, help='Number of PCA components (recommended: projection_dim // 4)')
    parser.add_argument('--umap_n_neighbors', type=int, default=10, help='UMAP n_neighbors')
    parser.add_argument('--umap_min_dist', type=float, default=0.2, help='UMAP min_dist')
    parser.add_argument('--umap_metric', type=str, default='cosine', help='UMAP metric (default: euclidean)')
    parser.add_argument('--umap_save_dir', type=str, default=None, help='UMAP save directory')

    ## save_path
    parser.add_argument('--best_model_dir', type=str, default=None, help='Directory to save best model checkpoint')
    ## load_path
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load model checkpoint (for resuming or inference)')

    ##################################################################################################################################

    args = parser.parse_args([])

    args.wandb_run_name = f"{args.experiment_id}: COV on"

    if args.run_name is None:
        args.run_name = f"experiment_{args.experiment_id}"

    if args.best_model_dir is None:
        args.best_model_dir = f'./output/checkpoints/{args.run_name}'

    if args.umap_save_dir is None:
        args.umap_save_dir = f'./output/umap/{args.run_name}'

    return args