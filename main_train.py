import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"

import time
import pandas as pd
import torch.distributed as dist

from training.trainer import train_multimodal_model
from training.run import parse_arguments
from utils import set_seed

root_dir = '/home/DAHS1/gangmin/my_research/'

def load_preprocessed_data():
    demo_df = pd.read_feather(root_dir + 'processed/final_demo_df.ftr')

    # 5 days
    # ts_df = pd.read_feather(root_dir + 'src/test/final_ts_dataset_1228.ftr') # Min-max scaling
    # ts_df = pd.read_feather(root_dir + 'src/test/final_ts_dataset_2026_0126.ftr') # Feature-specific scaling
    ts_df = pd.read_feather(root_dir + 'src/test/final_ts_dataset_2026_0223.ftr') # No observed_mask, sentinel=-2

    img_df = pd.read_feather(root_dir + 'src/test/total_cxr_df_5days_20260128.ftr')
    text_df = pd.read_feather(root_dir + 'src/test/final_text_df_20260128.ftr')
    return ts_df, img_df, text_df, demo_df

def main():
    args = parse_arguments()

    set_seed(args.random_seed)

    ts_df, img_df, text_df, demo_df = load_preprocessed_data()

    if not args.use_demographic:
        print("Demographic features disabled. Setting demo_df to None.")
        demo_df = None
    else:
        print("Demographic features enabled.")

    start_time = time.time()
    try:
        train_multimodal_model(ts_df, img_df, text_df, demo_df, args)
        print("✅ Training completed successfully.")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

    finally:
        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        print(f"\nTotal time spent on the experiment: {h}h {m}m {s}s")

if __name__ == "__main__":
    main()