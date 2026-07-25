from training_duett.run import parse_teacher_args
from training_duett.trainer import train_teacher


if __name__ == "__main__":
    args = parse_teacher_args()
    train_teacher(args)
