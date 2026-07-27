set -e

# ── Teacher 학습 (dual, 새 residual fusion) ─────────────────────────────────
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file /home/DAHS1/.cache/huggingface/accelerate/config_single_gpu_0.yaml \
  main_train_teacher_duett.py \
  --eval_train_batches 100 \
  --perceiver_type dual_patch \
  --freeze_duett \
  --aux_img_alpha 0.2 \
  --aux_ts_alpha 0.1 \d
  --aux_fus_alpha 1.0 \
  --query_lr_mult 0.5 \
  --correction_lr_mult 3.0


  # --wandb_disabled \
  # --aug_noise 0.05 --aug_mask 0.15 \
  # --d_latent 128
  # --label_weights 1.0,1.0,1.0 \
  # --perceiver_type dual



# ── Student 학습 ─────
# TEACHER_CKPT=
# CUDA_VISIBLE_DEVICES=0 accelerate launch \
#   --config_file /home/DAHS1/.cache/huggingface/accelerate/config_single_gpu_0.yaml \
#   main_train_student_duett.py \
#   --teacher_ckpt "$TEACHER_CKPT" \
#   --kd_alpha 0.5 \
#   --kd_T 4.0 \
#   --student_pool mean