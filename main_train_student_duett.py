"""Entry point: train the DuETT-KD Student (1-modal, distilled from Teacher).

Usage:
    accelerate launch main_train_student_duett.py \\
        --duett_ckpt <path> --teacher_ckpt <path> [--kd_alpha 0.5 --kd_T 4.0]
"""
from training_duett.run import parse_student_args
from training_duett.trainer import train_student


if __name__ == "__main__":
    args = parse_student_args()
    train_student(args)