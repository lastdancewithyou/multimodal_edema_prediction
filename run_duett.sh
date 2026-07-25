set -e

# ── Teacher 학습 (dual, 새 residual fusion) ─────────────────────────────────
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file /home/DAHS1/.cache/huggingface/accelerate/config_single_gpu_0.yaml \
  main_train_teacher_duett.py \
  --eval_train_batches 100 \
  --perceiver_type dual_patch \
  --freeze_duett

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


멀티 헤드 개수 2개로 줄지 않았는지 체크해야 함
64 128차원ㅇ네서의 시각화도 뽑아 봐야 함.
쿼리 random 값도 기존 0.02에서 0.1 됐는데 어떻게 처리할 것인지에 대한 고민 필수적임.