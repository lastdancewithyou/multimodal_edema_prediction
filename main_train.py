import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"

import time
import traceback
import pandas as pd
import torch
import torch.distributed as dist
import wandb

from training.trainer import train_single_stage_multimodal_model
from training.run import parse_arguments
from utils.utils import set_seed


def load_preprocessed_data():
    """
    전처리 완료된 데이터 로드
    """
    data_dir = '/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/2rd_preprocessed_data/processed/'
    train_df = pd.read_feather(data_dir + 'train_multimodal_20260427.ftr')
    val_df = pd.read_feather(data_dir + 'val_multimodal_20260427.ftr')
    test_df = pd.read_feather(data_dir + 'test_multimodal_20260427.ftr')

    print(f"\n[Data Loading] Train: {len(train_df):,} rows, Val: {len(val_df):,} rows, Test: {len(test_df):,} rows")

    return train_df, val_df, test_df


def main():
    args = parse_arguments()

    set_seed(args.random_seed)

    train_df, val_df, test_df = load_preprocessed_data()

    start_time = time.time()

    try:
        print(f"\n{'='*80}")
        print("Starting End-to-End MultiModal Multi-Task Training")
        print(f"{'='*80}\n")
        train_single_stage_multimodal_model(train_df, val_df, test_df, args)

        print("✅ Training completed successfully.")

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TRAINING FAILED")
        print("="*80)
        traceback.print_exc()
        print("="*80)

        is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

        if is_main_process:
            error_msg = (
                f"**Experiment:** {args.experiment_id}\n\n"
                f"**Error Type:** `{type(e).__name__}`\n\n"
                f"**Error Message:** {str(e)}\n\n"
                f"**Full Traceback:**\n```\n{traceback.format_exc()}\n```"
            )

            try:
                wandb.alert(
                    title=f"🚨 Training Failed - Experiment {args.experiment_id}",
                    text=error_msg,
                    level=wandb.AlertLevel.ERROR
                )
                print("\n✅ Error alert sent to WandB (and Slack if integrated)")
            except Exception as alert_error:
                print(f"\n⚠️  Failed to send WandB alert: {alert_error}")

        raise

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("✅ Distributed process group cleaned up")

        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        print(f"\nTotal time spent on the experiment: {h}h {m}m {s}s")

if __name__ == "__main__":
    main()