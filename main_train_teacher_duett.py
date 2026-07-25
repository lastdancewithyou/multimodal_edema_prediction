"""Entry point: train the DuETT-KD Teacher (2-modal, privileged).

Usage:
    accelerate launch main_train_teacher_duett.py --duett_ckpt <path> [flags]
"""
from training_duett.run import parse_teacher_args
from training_duett.trainer import train_teacher


if __name__ == "__main__":
    args = parse_teacher_args()
    train_teacher(args)
