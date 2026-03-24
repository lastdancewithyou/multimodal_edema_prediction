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

root_dir = '/home/DAHS1/gangmin/my_research/'

def load_preprocessed_data():
    # 5 days
    # ts_df = pd.read_feather(root_dir + 'src/test/final_ts_dataset_2026_0223.ftr') # No observed_mask, sentinel=-2
    ts_df = pd.read_feather(root_dir + 'src/test/final_ts_dataset_20260324.ftr') # score_diff_normalized col add

    # img_df = pd.read_feather(root_dir + 'src/test/total_cxr_df_5days_20260128.ftr')
    img_df = pd.read_feather(root_dir + 'src/test/total_cxr_df_5days_20260316.ftr')

    text_df = pd.read_feather(root_dir + 'src/test/final_text_df_20260128.ftr')

    clinical_prompt_df = pd.read_feather(root_dir + "clinical_multimodal_learning/data/clinical_prompt_df.ftr")
    return ts_df, img_df, text_df, clinical_prompt_df


def main():
    args = parse_arguments()

    set_seed(args.random_seed)

    ts_df, img_df, text_df, clinical_prompt_df = load_preprocessed_data()

    if not args.use_clinical_prompt:
        print("Prompt data disabled.")
        clinical_prompt_df = None
    else:
        print("Prompt Data enabled.")

    start_time = time.time()

    try:
        train_single_stage_multimodal_model(ts_df, img_df, text_df, clinical_prompt_df, args)
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