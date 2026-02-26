set -e

# cd my_research/clinical_multimodal_learning/

# GPU 2개
# accelerate launch \
#   --config_file /home/DAHS1/.cache/huggingface/accelerate/default_config.yaml \
#   --num_processes 2 \
#   main_train.py

# GPU 1개 (Rank 0)
# CUDA_VISIBLE_DEVICES=0 accelerate launch \
#   --config_file /home/DAHS1/.cache/huggingface/accelerate/config_single_gpu_0.yaml \
#   main_train.py

# # GPU 1개 (Rank 1)
# CUDA_VISIBLE_DEVICES=1 accelerate launch \
#   --config_file /home/DAHS1/.cache/huggingface/accelerate/config_single_gpu_1.yaml \
#   main_train.py