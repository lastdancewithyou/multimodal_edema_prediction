set -e

# ── Teacher 학습 (dual, 새 residual fusion) ─────────────────────────────────
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file /home/DAHS1/.cache/huggingface/accelerate/config_single_gpu_0.yaml \
  main_train_teacher_duett.py \
  --eval_train_batches 50 \
  --perceiver_type dual_patch \
  --freeze_duett \
  --aux_img_alpha 0.5 \
  --aux_ts_alpha 0.5 \
  --aux_fus_alpha 1.0 \
  --query_lr_mult 0.5 \
  --correction_lr_mult 1.0 \
  --aux_residual_alpha 0.0 \
  --wandb_disabled

  # alpha 0.0_0.5_1.0
  # alpha 0.5_0.5_1.0

  # --wandb_disabled \
  # --aug_noise 0.05 --aug_mask 0.15 \
  # --d_latent 128
  # --label_weights 1.0,1.0,1.0 \
  # --perceiver_type dual


# ── Correction-only Linear Probing ────────────────────────
# LP_CKPT="/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/checkpoints_duett/teacher/20260728_182107_correction_lr_mult=8.0_eval_train_batches=100_freeze_duett=True_perceiver_type=dual_patch_query_lr_mult=0.5/best.pt"
# CUDA_VISIBLE_DEVICES=0 accelerate launch \
#   --config_file /home/DAHS1/.cache/huggingface/accelerate/config_single_gpu_0.yaml \
#   main_train_teacher_duett.py \
#   --perceiver_type dual_patch \
#   --freeze_duett \
#   --lp_only_correction \
#   --lp_ckpt "$LP_CKPT" \
#   --lp_beta_l2 0.0 \
#   --lp_corr_l2 0.0 \
#   --lp_correction_dropout 0.0 \
#   --aux_img_alpha 0.0 \
#   --aux_ts_alpha 0.0 \
#   --aux_fus_alpha 0.3 \
#   --correction_lr_mult 1.0 \
#   --query_lr_mult 0.0 \
#   --epochs 15 \
#   --patience 5 \
#   --eval_train_batches 100 \
#   --wandb_disabled

  # --lp_beta_l2 1e-3 \
  # --lp_corr_l2 1e-2 \
  # --lp_correction_dropout 0.3 \


# ── Student 학습 ─────
# TEACHER_CKPT=
# CUDA_VISIBLE_DEVICES=0 accelerate launch \
#   --config_file /home/DAHS1/.cache/huggingface/accelerate/config_single_gpu_0.yaml \
#   main_train_student_duett.py \
#   --teacher_ckpt "$TEACHER_CKPT" \
#   --kd_alpha 0.5 \
#   --kd_T 4.0 \
#   --student_pool mean
