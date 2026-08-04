set -e

# Encoder-only intervention: keep the existing image, loss, query, and residual
# fusion settings fixed; replace only DuETT's contextual-hour readout with
# variable-first local trajectory tokens.
python analysis/smoke_test_trajectory_encoder.py

CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file /home/DAHS1/.cache/huggingface/accelerate/config_single_gpu_0.yaml \
  main_train_teacher_duett.py \
  --eval_train_batches 100 \
  --perceiver_type dual_patch \
  --ts_encoder trajectory_local \
  --trajectory_d_model 128 \
  --trajectory_gru_layers 1 \
  --trajectory_windows 6,12,24 \
  --freeze_duett \
  --aux_img_alpha 0.5 \
  --aux_ts_alpha 0.5 \
  --aux_fus_alpha 1.0 \
  --query_lr_mult 0.5 \
  --correction_lr_mult 1.0 \
  --aux_residual_alpha 0.0 \
  --wandb_disabled
