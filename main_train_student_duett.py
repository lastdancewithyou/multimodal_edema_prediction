from training_duett.run import parse_student_args
from training_duett.trainer import train_student


if __name__ == "__main__":
    args = parse_student_args()
    train_student(args)