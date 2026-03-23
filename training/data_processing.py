import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import hashlib
from tqdm.auto import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T

from utils.utils import timer, seed_worker

# IMAGE_DIR
# CACHED_IMAGE_DIR = "/home/DAHS1/gangmin/my_research/CXR/pt_20260128/" # before edge cut
# CACHED_IMAGE_DIR = "/home/DAHS1/gangmin/my_research/CXR/cached_images_20260316" # Wrong cropped_outline_image (224 by 224)

CACHED_IMAGE_DIR = "/home/DAHS1/gangmin/my_research/CXR/cached_images_224_0317" # cropped_outline_image (256 by 256)
# CACHED_IMAGE_DIR = "/home/DAHS1/gangmin/my_research/CXR/cached_images_256_0317" # cropped_outline_image (256 by 256)


class SCL_Multi_Dataset(Dataset):
    def __init__(self, args, merged_df):
        self.args = args
        self.window_size = args.window_size
        self.stride = args.stride
        self.prediction_horizon = args.prediction_horizon

        self.merged_df = merged_df
        self.stay_groups = self.merged_df.groupby('stay_id') # stay_id 식별자를 기준으로 grouping
        self.stay_ids = list(self.stay_groups.groups.keys())

        # Exclude clinical_prompt and prompt_id from time-series features
        exclude_cols = ['hadm_id', 'stay_id', 'hour_slot', 'Edema', 'subtype_label',
                        'cxr_flag', 'hash_path', 'text_flag', 'tokenized_text',
                        'clinical_prompt', 'prompt_id']
        all_feature_cols = [col for col in self.merged_df.columns if col not in exclude_cols]

        self.ts_features = all_feature_cols  # Includes both features and observed_mask
        print(f"[Dataset] Total features (including observed_mask): {len(self.ts_features)}")

        self.label_metadata = self._extract_label_metadata() # valid한 window만 학습에 사용할 수 있도록 stay_id별로 생성 가능한 label series를 미리 계산함.

        # ========== Valid window filtering ==========
        # 최소 1개 이상의 window를 생성할 수 있는 환자만 유지하여 너무 짧은 ICU Stay는 학습에서 제외함.
        valid_stay_ids = []
        valid_label_metadata = []
        new_stay_index_map = {}

        for idx, meta in enumerate(self.label_metadata):
            stay_id = meta["stay_id"]
            # Use edema_label_series for window count (main task)
            if "edema_label_series" in meta:
                window_count = len(meta["edema_label_series"])
            else:
                # Fallback to legacy label_series for backward compatibility
                window_count = len(meta.get("label_series", []))

            if window_count > 0:
                new_stay_index_map[stay_id] = len(valid_stay_ids)
                valid_stay_ids.append(stay_id)
                valid_label_metadata.append(meta)

        self.stay_ids = valid_stay_ids
        self.label_metadata = valid_label_metadata
        self.stay_index_map = new_stay_index_map

        # ========== image / text / clinical_prompt mapping 사전 구축 ==========
        # collate_fn에서 배치를 구성할 때, 중복 이미지/텍스트/프롬프트를 제거하고 unique한 것만 인코딩하기 위함
        self.image_map = {}
        self.text_map = {}
        self.prompt_map = {}

        for stay_id in self.stay_ids:
            stay_data = self.stay_groups.get_group(stay_id).sort_values('hour_slot')

            # cxr_flag == 1인 hour_slot만 매핑에 추가함.
            img_dict = {t: path for t, path, flag in zip(
                stay_data['hour_slot'],
                stay_data['hash_path'],
                stay_data['cxr_flag']
            ) if flag == 1}

            # text_flag == 1인 hour_slot만 매핑에 추가함 (radiology report)
            text_dict = {t: token for t, token, flag in zip(
                stay_data['hour_slot'],
                stay_data['tokenized_text'],
                stay_data['text_flag']
            ) if flag == 1}

            # prompt_id를 키로 clinical_prompt 매핑 (환자별 고유 ID 기반, -1 포함)
            prompt_dict = {}
            for _, row in stay_data.iterrows():
                pid = int(row['prompt_id'])  # -1 포함하여 모든 prompt_id 저장
                if pid not in prompt_dict:  # 중복 방지
                    prompt_dict[pid] = row['clinical_prompt']

            self.image_map[stay_id] = img_dict
            self.text_map[stay_id] = text_dict
            self.prompt_map[stay_id] = prompt_dict

        # ========== Image caching ==========
        self.image_cache = {}

        self.to_3ch = args.img_to_3ch # DenseNet: 1채널 grayscale, ResNet: 3채널 RGB (channel 맞춰줘야 함.)
        print(f"[Dataset] Image preprocessing config: to_3ch={self.to_3ch}")

    def load_image_cached(self, hash_filename):
        if not hash_filename or not isinstance(hash_filename, str) or hash_filename.strip() == "":
            raise RuntimeError(
                f"[ERROR] Invalid image filename: {hash_filename}\n"
            )

        if hash_filename in self.image_cache:
            return self.image_cache[hash_filename]

        file_path = os.path.join(CACHED_IMAGE_DIR, hash_filename)
        if not os.path.exists(file_path):
            raise RuntimeError(
                f"[ERROR] Image file does not exist: {file_path}\n"
            )

        try:
            # Load preprocessed tensor: [1, 224, 224]
            img_tensor = torch.load(file_path)

        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Failed to load image tensor from: {file_path}\n"
            )

        # shape 확인용
        if img_tensor.ndim != 3:
            raise RuntimeError(
                f"[ERROR] Invalid image tensor shape: {img_tensor.shape}\n"
            )

        # RGB 인코더 사용 시 1채널 → 3채널 변환
        if self.to_3ch and img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        # Cache 저장 후 반환
        self.image_cache[hash_filename] = img_tensor
        return img_tensor

    def __getitem__(self, idx):
        """
        개별 환자 데이터를 window로 slicing하여 반환함.

        1. 환자의 전체 시계열 데이터를 불러옴.
        2. Sliding window 방식으로 window 생성함.
        3. 각 window마다 prediction_horizon 이후의 label을 할당함.
        4. 이미지와 텍스트는 이 단계에서는 hour_slot index만 저장하고, collate_fn에서 실제 데이터를 불러옴.
        """
        stay_id = self.stay_ids[idx]
        stay_data = self.stay_groups.get_group(stay_id).sort_values('hour_slot')

        hour_slots = stay_data['hour_slot'].to_numpy()
        edema_labels = stay_data['Edema'].to_numpy()  # Binary edema label (0, 1, or -1)
        subtype_labels = stay_data['subtype_label'].to_numpy()  # Subtype label (1, 2, or -1) - will be converted to (0, 1, or -1)
        # Time-series features 추출 (observed_mask 포함)
        ts_feature_values = torch.tensor(stay_data[self.ts_features].astype(np.float32).to_numpy(), dtype=torch.float32)

        # Clinical prompt_id 시리즈 (환자별 고유 ID)
        prompt_ids = stay_data['prompt_id'].to_numpy()

        # 각 hour_slot에 이미지/텍스트/프롬프트가 있으면 해당 인덱스, 없으면 -1
        img_index_series = [t if t in self.image_map[stay_id] else -1 for t in hour_slots]
        text_index_series = [t if t in self.text_map[stay_id] else -1 for t in hour_slots]
        prompt_index_series = [int(pid) if pd.notna(pid) and pid != -1 and int(pid) in self.prompt_map[stay_id] else -1
                               for pid in prompt_ids]

        sequence_series, edema_label_series, subtype_label_series, has_cxr_series, has_text_series, valid_mask_series = [], [], [], [], [], []
        L = len(stay_data)

        # ========== Sliding Window 생성 ==========
        if L >= self.window_size + self.prediction_horizon:

            max_start_idx = L - self.window_size - self.prediction_horizon  # Window의 시점 후 prediction_horizon 이후에도 라벨이 존재해야 함

            # stride에 따른 window 생성
            for i in range(0, max_start_idx + 1, self.stride):
                window_hours = hour_slots[i:i + self.window_size]
                window_ts = ts_feature_values[i:i + self.window_size]
                window_img = img_index_series[i:i + self.window_size]
                window_text = text_index_series[i:i + self.window_size]
                window_prompt = prompt_index_series[i:i + self.window_size]

                window_sequence = [
                    {
                        'time_step': int(window_hours[j]),
                        'ts_features': window_ts[j],  # Includes observed_mask_* as regular features
                        'img_index': window_img[j],
                        'txt_index': window_text[j],
                        'prompt_index': window_prompt[j]  # Clinical prompt ID
                    }
                    for j in range(self.window_size)
                ]

                sequence_series.append(window_sequence)
                has_cxr_series.append([int(x != -1) for x in window_img])       # 1 if image exists
                has_text_series.append([int(x != -1) for x in window_text])     # 1 if text exists
                valid_mask_series.append([1] * self.window_size)

                # 라벨 할당 (both edema and subtype)
                label_idx = i + self.window_size + self.prediction_horizon - 1
                if label_idx < len(edema_labels):
                    future_edema = edema_labels[label_idx]
                    future_subtype = subtype_labels[label_idx]
                    if np.isnan(future_edema):
                        future_edema = -1
                    if np.isnan(future_subtype):
                        future_subtype = -1
                    # Convert subtype labels from {1, 2} to {0, 1} for CE loss
                    elif future_subtype in [1, 2]:
                        future_subtype = future_subtype - 1
                else:
                    future_edema = -1
                    future_subtype = -1
                edema_label_series.append(future_edema)
                subtype_label_series.append(future_subtype)

        # 생성된 모든 window는 유효함 (window_mask=1) - 이후 collate_fn에서 배치 간 length를 맞춰줄 때 window_mask에 0을 할당함.
        window_mask = [1] * len(sequence_series)

        return {
            'stay_id': stay_id,
            'modality_series': sequence_series,
            'has_cxr': has_cxr_series,
            'has_text': has_text_series,
            'edema_label_series': edema_label_series,
            'subtype_label_series': subtype_label_series,
            'window_mask': window_mask,
            'valid_mask_series': valid_mask_series
        }
    
    def collate_fn(self, batch):
        """
        배치 내 고유 이미지/텍스트만 추출하여 메모리와 연산량 절약
        - 중복 제거: 같은 이미지/텍스트가 여러 타임스텝에 있어도 한 번만 처리
        - 배치에 동일한 image가 100번 등장해도, 1번만 인코딩하고 인덱스로 참조함.
        """
        args = self.args
        stay_ids = [item['stay_id'] for item in batch]
        modality_series_list = [item['modality_series'] for item in batch]
        edema_label_series_list = [item['edema_label_series'] for item in batch]
        subtype_label_series_list = [item['subtype_label_series'] for item in batch]
        window_mask_list = [item['window_mask'] for item in batch]
        valid_seq_mask_list = [item['valid_mask_series'] for item in batch]

        max_windows = max(len(x) for x in modality_series_list) # 배치 내 최대 윈도우 개수 (패딩 기준)

        # ==================== 배치 내 고유 이미지/텍스트/프롬프트 추출 ====================
        unique_img_paths = []  # 배치 전체에서 중복 제거된 이미지 경로
        unique_txt_keys = []   # 배치 전체에서 중복 제거된 텍스트 키 (stay_id, hour_slot)
        unique_prompt_keys = []  # 배치 전체에서 중복 제거된 프롬프트 키 (stay_id, prompt_id)

        img_path_to_idx = {}  # {hash_path: unique_list_idx}
        txt_key_to_idx = {}   # {(stay_id, hour_slot): unique_list_idx}
        prompt_key_to_idx = {}  # {(stay_id, prompt_id): unique_list_idx}

        # 배치 전체를 순회하며 고유 항목 수집
        for item in batch:
            stay_id = item['stay_id']
            for window in item['modality_series']:
                for step in window:
                    # 이미지 경로 추출 (hour_slot → 실제 경로)
                    img_hour = step['img_index']  # hour_slot or -1
                    if img_hour != -1:
                        img_path = self.image_map[stay_id][img_hour]
                        if img_path not in img_path_to_idx:
                            img_path_to_idx[img_path] = len(unique_img_paths)
                            unique_img_paths.append(img_path)

                    # 텍스트 키 추출 (stay_id, hour_slot) - radiology report
                    txt_hour = step['txt_index']  # hour_slot or -1
                    if txt_hour != -1:
                        txt_key = (stay_id, txt_hour)
                        if txt_key not in txt_key_to_idx:
                            txt_key_to_idx[txt_key] = len(unique_txt_keys)
                            unique_txt_keys.append(txt_key)

                    # Clinical prompt 키 추출 (stay_id, prompt_id)
                    # -1도 포함 ("No clinical information available." 등)
                    prompt_id = step['prompt_index']  # prompt_id (including -1)
                    prompt_key = (stay_id, prompt_id)
                    if prompt_key not in prompt_key_to_idx:
                        prompt_key_to_idx[prompt_key] = len(unique_prompt_keys)
                        unique_prompt_keys.append(prompt_key)

        # ==================== 텐서 초기화 ====================
        ts_feature_dim = batch[0]['modality_series'][0][0]['ts_features'].shape[-1]
        ts_tensor = torch.zeros(len(batch), max_windows, args.window_size, ts_feature_dim)

        # -1: 이미지/텍스트/프롬프트 없음, 0~N-1: unique_list의 인덱스
        img_index_tensor = torch.full((len(batch), max_windows, args.window_size), fill_value=-1, dtype=torch.long)
        text_index_tensor = torch.full((len(batch), max_windows, args.window_size), fill_value=-1, dtype=torch.long)
        prompt_index_tensor = torch.full((len(batch), max_windows, args.window_size), fill_value=-1, dtype=torch.long)

        # 절대 시간 정보 (hour_slot) - Time2Vec에서 사용
        time_steps_tensor = torch.zeros(len(batch), max_windows, args.window_size, dtype=torch.float32)

        window_mask_tensor = torch.zeros(len(batch), max_windows, dtype=torch.bool) # 패딩 윈도우는 False

        # Sequence mask는 window 내 유효한 time step 표시
        valid_seq_mask_tensor = torch.tensor(
            [m + [[0]*args.window_size]*(max_windows - len(m)) for m in valid_seq_mask_list],
            dtype=torch.bool
        )

        # 각 window의 라벨 (-1은 패딩이거나 유효하지 않은 경우)
        edema_label_tensor = torch.full((len(batch), max_windows), fill_value=-1, dtype=torch.long)
        subtype_label_tensor = torch.full((len(batch), max_windows), fill_value=-1, dtype=torch.long)

        # ==================== 실제 값 채우기 ====================
        for i, item in enumerate(batch):
            stay_id = item['stay_id']
            num_windows = len(item['modality_series'])

            for j, window in enumerate(item['modality_series']):
                for t, step in enumerate(window):
                    ts_tensor[i, j, t] = step['ts_features']

                    time_steps_tensor[i, j, t] = step['time_step']

                    img_hour = step['img_index']
                    if img_hour != -1:
                        img_path = self.image_map[stay_id][img_hour]
                        img_index_tensor[i, j, t] = img_path_to_idx[img_path]

                    txt_hour = step['txt_index']
                    if txt_hour != -1:
                        txt_key = (stay_id, txt_hour)
                        text_index_tensor[i, j, t] = txt_key_to_idx[txt_key]

                    # Clinical prompt index (always has value, including -1)
                    prompt_id = step['prompt_index']
                    prompt_key = (stay_id, prompt_id)
                    prompt_index_tensor[i, j, t] = prompt_key_to_idx[prompt_key]

            window_mask_tensor[i, :num_windows] = torch.tensor(window_mask_list[i][:num_windows], dtype=torch.bool)
            edema_label_tensor[i, :num_windows] = torch.tensor(edema_label_series_list[i], dtype=torch.long)
            subtype_label_tensor[i, :num_windows] = torch.tensor(subtype_label_series_list[i], dtype=torch.long)

        # ==================== Clinical Prompt Extraction ====================
        # Extract unique prompt texts (raw text for tokenization in model forward)
        unique_prompt_texts = []
        for stay_id, prompt_id in unique_prompt_keys:
            unique_prompt_texts.append(self.prompt_map[stay_id][prompt_id])

        return {
            'stay_ids': stay_ids,
            'ts_tensor': ts_tensor,                      # [B, W, T, D] - Features and observed_mask
            'time_steps': time_steps_tensor,             # [B, W, T] - hour_slot
            'img_index_tensor': img_index_tensor,        # [B, W, T] → 0~N-1 (unique_img_paths 인덱스)
            'text_index_tensor': text_index_tensor,      # [B, W, T] → 0~M-1 (unique_txt_keys 인덱스)
            'prompt_index_tensor': prompt_index_tensor,  # [B, W, T] → 0~P-1 (unique_prompt_keys 인덱스)
            'unique_img_paths': unique_img_paths,        # List[str] - 배치 내 고유 이미지 경로 (길이 N)
            'unique_txt_keys': unique_txt_keys,          # List[(stay_id, hour_slot)] - 배치 내 고유 텍스트 키 (길이 M)
            'unique_prompt_texts': unique_prompt_texts,  # List[str] - 배치 내 고유 clinical prompt 텍스트 (길이 P)
            'window_mask': window_mask_tensor,           # [B, W]
            'valid_seq_mask': valid_seq_mask_tensor,     # [B, W, T]
            'edema_labels': edema_label_tensor,          # [B, W] - Binary edema labels
            'subtype_labels': subtype_label_tensor,      # [B, W] - Subtype labels (0, 1, 2, or -1)
        }

    def _extract_label_metadata(self):
        """
        - Dataset 초기화 시 각 환자가 몇 개의 윈도우를 생성할 수 있는지 파악
        - ICU stay가 너무 짧은 환자를 필터링
        """
        label_meta = []

        for idx in range(len(self.stay_ids)):
            stay_id = self.stay_ids[idx]
            stay_data = self.stay_groups.get_group(stay_id).sort_values('hour_slot')
            edema_labels = stay_data['Edema'].to_numpy()
            subtype_labels = stay_data['subtype_label'].to_numpy()

            window_edema_labels = []
            window_subtype_labels = []
            L = len(stay_data)

            # 환자별 윈도우 슬라이싱
            if L >= self.window_size + self.prediction_horizon:

                max_start_idx = L - self.window_size - self.prediction_horizon

                for i in range(0, max_start_idx + 1, self.stride):

                    future_label_idx = i + self.window_size + self.prediction_horizon - 1

                    if future_label_idx < L:
                        edema_lab = edema_labels[future_label_idx]
                        subtype_lab = subtype_labels[future_label_idx]
                        if np.isnan(edema_lab):
                            edema_lab = -1
                        if np.isnan(subtype_lab):
                            subtype_lab = -1
                        window_edema_labels.append(edema_lab)
                        window_subtype_labels.append(subtype_lab)

            label_meta.append({
                'stay_id': stay_id,
                'edema_label_series': window_edema_labels,
                'subtype_label_series': window_subtype_labels
            })
        return label_meta
    
    def __len__(self): 
        return len(self.stay_ids)


#######################################################################
# 데이터셋 정의
#######################################################################
def get_dataloaders(ts_df, cxr_df, text_df, clinical_prompt_df, args, accelerator=None, num_workers=8):
    # 데이터 병합
    with timer("Dataset 병합"):
        merged_df = merged_dataframes(ts_df, cxr_df, text_df, clinical_prompt_df)

    # 데이터 분할
    train_df, val_df, test_df = split_dataset(
        merged_df=merged_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )

    with timer("Dataset 생성"):
        train_dataset = SCL_Multi_Dataset(args, train_df)
        val_dataset = SCL_Multi_Dataset(args, val_df)
        test_dataset = SCL_Multi_Dataset(args, test_df)

    with timer("윈도우 라벨 분포 계산"):
        train_dist = calculate_window_label_distribution(train_dataset, "Train")
        val_dist = calculate_window_label_distribution(val_dataset, "Validation")
        test_dist = calculate_window_label_distribution(test_dataset, "Test")

    with timer("윈도우별 모달리티 분포 계산"):
        train_modality_dist = calculate_modality_distribution(train_dataset, "Train")
        val_modality_dist = calculate_modality_distribution(val_dataset, "Validation")
        test_modality_dist = calculate_modality_distribution(test_dataset, "Test")

    with timer("샘플러 가동"):
        train_sampler = StratifiedPatientSampler(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            accelerator=accelerator,
            shuffle=True,
            drop_last=True,
            seed=args.random_seed,
            split="Train"
        )

        val_sampler = StratifiedPatientSampler(
            dataset=val_dataset,
            batch_size=args.val_batch_size,
            accelerator=accelerator,
            shuffle=False,
            drop_last=True,
            seed=args.random_seed,
            split="Validation"
        )

        test_sampler = StratifiedPatientSampler(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            accelerator=accelerator,
            shuffle=False,
            drop_last=True,
            seed=args.random_seed,
            split="Test"
        )

    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn
    test_collate_fn = test_dataset.collate_fn

    with timer("데이터로더 정의"):
        # Configure DataLoader settings based on num_workers
        dataloader_kwargs = {
            'pin_memory': True,
            'num_workers': num_workers,
        }

        # Only use prefetch_factor and persistent_workers if num_workers > 0
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = 2
            dataloader_kwargs['persistent_workers'] = True
            dataloader_kwargs['worker_init_fn'] = seed_worker

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            collate_fn=train_collate_fn,
            **dataloader_kwargs
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_sampler=val_sampler,
            collate_fn=val_collate_fn,
            **dataloader_kwargs
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_sampler=test_sampler,
            collate_fn=test_collate_fn,
            **dataloader_kwargs
        )

    # # 윈도우 레벨 라벨 비율 비교
    # print("\n" + "="*60)
    # print("Window-level Label Ratio Comparison")
    # print("="*60)
    # print(f"{'Dataset':<12} {'Cardio':>10} {'Non-cardio':>12} {'Negative':>10} {'Unlabeled':>10}")
    # print("-"*60)

    # for name, dist in [("Train", train_dist), ("Validation", val_dist), ("Test", test_dist)]:
    #     total = dist['total']
    #     cardio_pct = dist['cardio'] / total * 100
    #     noncardio_pct = dist['noncardio'] / total * 100
    #     negative_pct = dist['negative'] / total * 100
    #     unlabeled_pct = dist['unlabeled'] / total * 100
    #     print(f"{name:<12} {cardio_pct:>9.2f}% {noncardio_pct:>11.2f}% {negative_pct:>9.2f}% {unlabeled_pct:>9.2f}%")

    # print("="*60 + "\n")

    # # 윈도우 레벨 모달리티 조합 비율 비교
    # print("\n" + "="*70)
    # print("Window-level Modality Combination Ratio Comparison")
    # print("="*70)
    # print(f"{'Dataset':<12} {'TS only':>12} {'TS+Image':>12} {'TS+Text':>12} {'TS+Img+Text':>14}")
    # print("-"*70)

    # for name, dist in [("Train", train_modality_dist), ("Validation", val_modality_dist), ("Test", test_modality_dist)]:
    #     total = dist['total']
    #     ts_only_pct = dist['ts_only'] / total * 100
    #     ts_img_pct = dist['ts_img'] / total * 100
    #     ts_text_pct = dist['ts_text'] / total * 100
    #     ts_img_text_pct = dist['ts_img_text'] / total * 100
    #     print(f"{name:<12} {ts_only_pct:>11.2f}% {ts_img_pct:>11.2f}% {ts_text_pct:>11.2f}% {ts_img_text_pct:>13.2f}%")

    # print("="*70 + "\n")

    return train_dataloader, val_dataloader, test_dataloader, train_sampler


#######################################################################
# 데이터셋 정의를 위한 기타 함수
#######################################################################
# merging dataframes
def merged_dataframes(ts_df, img_df, text_df, clinical_prompt_df):
    merged_df = (ts_df
            .merge(img_df, on=['stay_id', 'hour_slot'], how='outer')
            .merge(text_df, on=['stay_id','hour_slot'], how='outer')
            .merge(clinical_prompt_df[['hadm_id', 'stay_id', 'hour_slot', 'clinical_prompt', 'prompt_id']],
                   on=['hadm_id', 'stay_id', 'hour_slot'], how='left')
    )

    ########################################################################
    # Analyze modality combinations at hour_slot level
    print(f"\n{'='*80}")
    print(f"📊 HOUR_SLOT LEVEL MODALITY COMBINATION ANALYSIS")
    print(f"{'='*80}")

    # Define modality flags
    has_ts = merged_df['hour_slot'].notna()  # TS always exists (from ts_df)
    has_img = merged_df['cxr_flag'].notna() & (merged_df['cxr_flag'] == 1)
    has_text = merged_df['text_flag'].notna() & (merged_df['text_flag'] == 1)

    # Count combinations
    ts_only = (has_ts & ~has_img & ~has_text).sum()
    ts_img = (has_ts & has_img & ~has_text).sum()
    ts_text = (has_ts & ~has_img & has_text).sum()
    ts_img_text = (has_ts & has_img & has_text).sum()

    total_hourslots = len(merged_df)

    print(f"\n[Merged DataFrame] Total hour_slots: {total_hourslots:,}")
    print(f"{'─'*80}")
    print(f"Modality Combinations:")
    print(f"  TS only:              {ts_only:>8,} ({ts_only/total_hourslots*100:>5.2f}%)")
    print(f"  TS + Image:           {ts_img:>8,} ({ts_img/total_hourslots*100:>5.2f}%)")
    print(f"  TS + Text:            {ts_text:>8,} ({ts_text/total_hourslots*100:>5.2f}%)")
    print(f"  TS + Image + Text:    {ts_img_text:>8,} ({ts_img_text/total_hourslots*100:>5.2f}%)")
    print(f"{'─'*80}")
    print(f"Multimodal Coverage:")
    print(f"  At least Image:       {(has_img).sum():>8,} ({(has_img).sum()/total_hourslots*100:>5.2f}%)")
    print(f"  At least Text:        {(has_text).sum():>8,} ({(has_text).sum()/total_hourslots*100:>5.2f}%)")
    print(f"  At least one extra:   {(has_img | has_text).sum():>8,} ({(has_img | has_text).sum()/total_hourslots*100:>5.2f}%)")
    print(f"{'='*80}\n")
    ########################################################################
    return merged_df


def stable_hash(s):
        return hashlib.md5(s.encode()).hexdigest()


def calculate_window_label_distribution(dataset, dataset_name="Dataset"):
    """
    - 데이터셋의 window-level 라벨 분포를 계산하고 출력
    - Multi-task: Edema와 Subtype 분포를 모두 출력
    """
    # Check if multi-task (edema_label_series exists)
    if len(dataset.label_metadata) > 0 and 'edema_label_series' in dataset.label_metadata[0]:
        # Multi-task distribution
        edema_0_windows = 0
        edema_1_windows = 0
        edema_unlabeled_windows = 0

        subtype_1_windows = 0  # non-cardiogenic
        subtype_2_windows = 0  # cardiogenic
        subtype_unlabeled_windows = 0

        for meta in dataset.label_metadata:
            edema_labels = meta['edema_label_series']
            subtype_labels = meta['subtype_label_series']

            for edema_label, subtype_label in zip(edema_labels, subtype_labels):
                # Count edema distribution
                if edema_label == 0:
                    edema_0_windows += 1
                elif edema_label == 1:
                    edema_1_windows += 1
                else:
                    edema_unlabeled_windows += 1

                # Count subtype distribution (only for edema=1)
                # Subtype labels: 1 (non-cardiogenic), 2 (cardiogenic)
                if edema_label == 1:
                    if subtype_label == 1:
                        subtype_1_windows += 1
                    elif subtype_label == 2:
                        subtype_2_windows += 1
                    else:
                        # -1 or NaN (unlabeled)
                        subtype_unlabeled_windows += 1

        total_windows = edema_0_windows + edema_1_windows + edema_unlabeled_windows

        print(f"\n{'='*60}")
        print(f"[{dataset_name}] Window-level Label Distribution (Multi-task)")
        print(f"{'='*60}")
        print(f"Edema Distribution:")
        print(f"  No edema (0):        {edema_0_windows:>6} ({edema_0_windows/total_windows*100:>5.2f}%)")
        print(f"  Has edema (1):       {edema_1_windows:>6} ({edema_1_windows/total_windows*100:>5.2f}%)")
        print(f"  Unlabeled:           {edema_unlabeled_windows:>6} ({edema_unlabeled_windows/total_windows*100:>5.2f}%)")
        print(f"{'─'*60}")
        print(f"Subtype Distribution (among edema=1):")
        edema_1_total = subtype_1_windows + subtype_2_windows + subtype_unlabeled_windows
        if edema_1_total > 0:
            print(f"  Non-cardiogenic (1): {subtype_1_windows:>6} ({subtype_1_windows/edema_1_total*100:>5.2f}%)")
            print(f"  Cardiogenic (2):     {subtype_2_windows:>6} ({subtype_2_windows/edema_1_total*100:>5.2f}%)")
            print(f"  Unlabeled:           {subtype_unlabeled_windows:>6} ({subtype_unlabeled_windows/edema_1_total*100:>5.2f}%)")
        print(f"{'─'*60}")
        print(f"Total windows:         {total_windows:>6}")
        print(f"{'='*60}\n")

        return {
            'edema_0': edema_0_windows,
            'edema_1': edema_1_windows,
            'edema_unlabeled': edema_unlabeled_windows,
            'subtype_1': subtype_1_windows,
            'subtype_2': subtype_2_windows,
            'subtype_unlabeled': subtype_unlabeled_windows,
            'total': total_windows
        }

    else:
        # Legacy distribution (3-class)
        cardio_windows = 0
        noncardio_windows = 0
        negative_windows = 0
        unlabeled_windows = 0

        for meta in dataset.label_metadata:
            for label in meta['label_series']:
                if label == 2:
                    cardio_windows += 1
                elif label == 1:
                    noncardio_windows += 1
                elif label == 0:
                    negative_windows += 1
                else:  # -1 or NaN
                    unlabeled_windows += 1

        total_windows = cardio_windows + noncardio_windows + negative_windows + unlabeled_windows

        print(f"\n{'='*60}")
        print(f"[{dataset_name}] Window-level Label Distribution")
        print(f"{'='*60}")
        print(f"Cardio windows:        {cardio_windows:>6} ({cardio_windows/total_windows*100:>5.2f}%)")
        print(f"Non-cardio windows:    {noncardio_windows:>6} ({noncardio_windows/total_windows*100:>5.2f}%)")
        print(f"Negative windows:      {negative_windows:>6} ({negative_windows/total_windows*100:>5.2f}%)")
        print(f"Unlabeled windows:     {unlabeled_windows:>6} ({unlabeled_windows/total_windows*100:>5.2f}%)")
        print(f"{'─'*60}")
        print(f"Total windows:         {total_windows:>6}")
        print(f"{'='*60}\n")

        return {
            'cardio': cardio_windows,
            'noncardio': noncardio_windows,
            'negative': negative_windows,
            'unlabeled': unlabeled_windows,
            'total': total_windows
        }


def calculate_modality_distribution(dataset, dataset_name="Dataset"):
    """
    - 데이터셋의 window-level 모달리티 조합 분포를 계산하고 출력
    - 각 window가 어떤 모달리티 조합을 가지고 있는지 분석

    카테고리:
    1. ts_only: 시계열만 (이미지 X, 텍스트 X)
    2. ts_img: 시계열 + 이미지 (텍스트 X)
    3. ts_text: 시계열 + 텍스트 (이미지 X)
    4. ts_img_text: 시계열 + 이미지 + 텍스트
    """
    ts_only = 0
    ts_img = 0
    ts_text = 0
    ts_img_text = 0

    # 모든 환자 데이터를 순회하며 window별 모달리티 조합 분석
    for stay_id in dataset.stay_ids:
        stay_data = dataset.stay_groups.get_group(stay_id).sort_values('hour_slot')

        hour_slots = stay_data['hour_slot'].to_numpy()
        L = len(stay_data)

        # 각 hour_slot에 이미지와 텍스트가 있는지 확인
        img_index_series = [t if t in dataset.image_map[stay_id] else -1 for t in hour_slots]
        text_index_series = [t if t in dataset.text_map[stay_id] else -1 for t in hour_slots]

        # Sliding window 생성 (dataset의 __getitem__과 동일한 로직)
        if L >= dataset.window_size + dataset.prediction_horizon:
            max_start_idx = L - dataset.window_size - dataset.prediction_horizon

            for i in range(0, max_start_idx + 1, dataset.stride):
                window_img = img_index_series[i:i + dataset.window_size]
                window_text = text_index_series[i:i + dataset.window_size]

                # window 내에 이미지/텍스트가 하나라도 있는지 확인
                has_img = any(x != -1 for x in window_img)
                has_text = any(x != -1 for x in window_text)

                # 모달리티 조합에 따라 카운트
                if has_img and has_text:
                    ts_img_text += 1
                elif has_img:
                    ts_img += 1
                elif has_text:
                    ts_text += 1
                else:
                    ts_only += 1

    total_windows = ts_only + ts_img + ts_text + ts_img_text

    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Window-level Modality Distribution")
    print(f"{'='*60}")
    print(f"TS only:               {ts_only:>6} ({ts_only/total_windows*100:>5.2f}%)")
    print(f"TS + Image:            {ts_img:>6} ({ts_img/total_windows*100:>5.2f}%)")
    print(f"TS + Text:             {ts_text:>6} ({ts_text/total_windows*100:>5.2f}%)")
    print(f"TS + Image + Text:     {ts_img_text:>6} ({ts_img_text/total_windows*100:>5.2f}%)")
    print(f"{'─'*60}")
    print(f"Total windows:         {total_windows:>6}")
    print(f"{'='*60}\n")

    return {
        'ts_only': ts_only,
        'ts_img': ts_img,
        'ts_text': ts_text,
        'ts_img_text': ts_img_text,
        'total': total_windows
    }


def analyze_batch_label_distribution(dataloader, dataset_name="Dataset", num_batches=None):
    """
    - DataLoader의 각 배치별 라벨 분포를 분석하고 통계를 출력
    - num_batches: 분석할 배치 수 (None이면 전체)
    """
    batch_stats = []

    print(f"\n{'='*80}")
    print(f"[{dataset_name}] Batch-level Label Distribution Analysis")
    print(f"{'='*80}")

    for batch_idx, batch in enumerate(dataloader):
        if num_batches and batch_idx >= num_batches:
            break

        labels = batch['labels']  # [B, W]
        window_mask = batch['window_mask']  # [B, W]

        # 유효한 window만 필터링 (window_mask=True & label != -1)
        valid_labels = labels[window_mask & (labels != -1)]

        if len(valid_labels) == 0:
            continue

        # 각 라벨별 개수 계산
        cardio_count = (valid_labels == 2).sum().item()
        noncardio_count = (valid_labels == 1).sum().item()
        negative_count = (valid_labels == 0).sum().item()
        total_valid = len(valid_labels)

        batch_stats.append({
            'batch_idx': batch_idx,
            'cardio': cardio_count,
            'noncardio': noncardio_count,
            'negative': negative_count,
            'total': total_valid,
            'cardio_pct': cardio_count / total_valid * 100 if total_valid > 0 else 0,
            'noncardio_pct': noncardio_count / total_valid * 100 if total_valid > 0 else 0,
            'negative_pct': negative_count / total_valid * 100 if total_valid > 0 else 0
        })

    if not batch_stats:
        print(f"[Warning] No valid batches found in {dataset_name}")
        return

    # 전체 배치 통계
    total_cardio = sum(stat['cardio'] for stat in batch_stats)
    total_noncardio = sum(stat['noncardio'] for stat in batch_stats)
    total_negative = sum(stat['negative'] for stat in batch_stats)
    total_windows = sum(stat['total'] for stat in batch_stats)

    # 배치별 비율의 평균 및 표준편차
    cardio_pcts = [stat['cardio_pct'] for stat in batch_stats]
    noncardio_pcts = [stat['noncardio_pct'] for stat in batch_stats]
    negative_pcts = [stat['negative_pct'] for stat in batch_stats]

    cardio_mean = np.mean(cardio_pcts)
    cardio_std = np.std(cardio_pcts)
    noncardio_mean = np.mean(noncardio_pcts)
    noncardio_std = np.std(noncardio_pcts)
    negative_mean = np.mean(negative_pcts)
    negative_std = np.std(negative_pcts)

    print(f"\n전체 배치 통계 (분석 배치 수: {len(batch_stats)})")
    print(f"{'─'*80}")
    print(f"Total windows across all batches: {total_windows:,}")
    print(f"  Cardio:     {total_cardio:>6,} ({total_cardio/total_windows*100:>5.2f}%)")
    print(f"  Non-cardio: {total_noncardio:>6,} ({total_noncardio/total_windows*100:>5.2f}%)")
    print(f"  Negative:   {total_negative:>6,} ({total_negative/total_windows*100:>5.2f}%)")

    print(f"\n배치별 라벨 비율 (평균 ± 표준편차)")
    print(f"{'─'*80}")
    print(f"  Cardio:     {cardio_mean:>5.2f}% ± {cardio_std:>4.2f}%")
    print(f"  Non-cardio: {noncardio_mean:>5.2f}% ± {noncardio_std:>4.2f}%")
    print(f"  Negative:   {negative_mean:>5.2f}% ± {negative_std:>4.2f}%")

    # 배치 크기 통계
    batch_sizes = [stat['total'] for stat in batch_stats]
    print(f"\n배치당 유효 window 수")
    print(f"{'─'*80}")
    print(f"  평균: {np.mean(batch_sizes):.1f}")
    print(f"  최소: {np.min(batch_sizes)}")
    print(f"  최대: {np.max(batch_sizes)}")
    print(f"  표준편차: {np.std(batch_sizes):.1f}")

    print(f"{'='*80}\n")

    return batch_stats


def split_dataset(merged_df, train_ratio, val_ratio, random_seed=0):
    """
    환자 레벨에서 층화추출을 수행하여 train/val/test split
    Multi-task learning: Edema 라벨을 기준으로 층화추출
    (Main task: Edema detection, Sub task: Subtype classification)
    """
    stay_ids = merged_df['stay_id'].unique()

    # Check if Edema column exists (multi-task) or use legacy label column
    use_multitask = 'Edema' in merged_df.columns

    if use_multitask:
        print("[Dataset Split] Using Edema-based stratification for multi-task learning")
        # 각 환자의 대표 Edema 라벨 결정 (우선순위: edema=1 > edema=0 > unlabeled)
        stay_labels = []
        for stay_id in stay_ids:
            stay_data = merged_df[merged_df['stay_id'] == stay_id]
            edema_labels = stay_data['Edema'].to_numpy()

            if np.any(edema_labels == 1):  # Has edema
                stay_labels.append(1)
            elif np.any(edema_labels == 0):  # No edema
                stay_labels.append(0)
            else:  # Unlabeled (all NaN or -1)
                stay_labels.append(-1)
    else:
        print("[Dataset Split] Using legacy label-based stratification")
        # 각 환자의 대표 라벨 결정 (우선순위: cardio > noncardio > negative > unlabeled)
        stay_labels = []
        for stay_id in stay_ids:
            stay_data = merged_df[merged_df['stay_id'] == stay_id]
            labels = stay_data['label'].to_numpy()

            if np.any(labels == 2):  # Cardio
                stay_labels.append(2)
            elif np.any(labels == 1):  # Non-cardio
                stay_labels.append(1)
            elif np.any(labels == 0):  # Negative
                stay_labels.append(0)
            else:  # Unlabeled (all NaN or -1)
                stay_labels.append(-1)

    # 층화추출로 train/temp split
    train_stay_ids, temp_stay_ids = train_test_split(
        stay_ids,
        test_size = (1 - train_ratio),
        random_state=random_seed,
        stratify=stay_labels
    )

    # temp의 라벨 추출
    temp_stay_labels = []
    for stay_id in temp_stay_ids:
        stay_data = merged_df[merged_df['stay_id'] == stay_id]

        if use_multitask:
            edema_labels = stay_data['Edema'].to_numpy()
            if np.any(edema_labels == 1):
                temp_stay_labels.append(1)
            elif np.any(edema_labels == 0):
                temp_stay_labels.append(0)
            else:
                temp_stay_labels.append(-1)
        else:
            labels = stay_data['label'].to_numpy()
            if np.any(labels == 2):
                temp_stay_labels.append(2)
            elif np.any(labels == 1):
                temp_stay_labels.append(1)
            elif np.any(labels == 0):
                temp_stay_labels.append(0)
            else:
                temp_stay_labels.append(-1)

    val_size = val_ratio / (1 - train_ratio)

    # 층화추출로 val/test split
    val_stay_ids, test_stay_ids = train_test_split(
        temp_stay_ids,
        test_size = (1 - val_size),
        random_state=random_seed,
        stratify=temp_stay_labels
    )

    train_df = merged_df[merged_df['stay_id'].isin(train_stay_ids)]
    val_df = merged_df[merged_df['stay_id'].isin(val_stay_ids)]
    test_df = merged_df[merged_df['stay_id'].isin(test_stay_ids)]

    print("\n[Dataset Split] Stratified patient-level distribution:")
    print(f"Train patients: {len(train_stay_ids)}")
    print(f"Val patients:   {len(val_stay_ids)}")
    print(f"Test patients:  {len(test_stay_ids)}")

    # Print label distribution for multi-task
    if use_multitask:
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            edema_0 = (split_df['Edema'] == 0).sum()
            edema_1 = (split_df['Edema'] == 1).sum()
            edema_unlabeled = (split_df['Edema'] == -1).sum()

            # Among edema=1, count subtype distribution
            edema_1_df = split_df[split_df['Edema'] == 1]
            # subtype_0 = (edema_1_df['subtype_label'] == 0).sum() if 'subtype_label' in split_df.columns else 0
            subtype_1 = (edema_1_df['subtype_label'] == 1).sum() if 'subtype_label' in split_df.columns else 0
            subtype_2 = (edema_1_df['subtype_label'] == 2).sum() if 'subtype_label' in split_df.columns else 0
            # subtype_unlabeled = (edema_1_df['subtype_label'] == -1).sum() if 'subtype_label' in split_df.columns else 0 # 지금 NA 처리되어 있음.

            print(f"\n{'='*80}")
            print(f"{split_name} Set:")
            print(f"  Edema Negative={edema_0}, Edema Positive={edema_1}, Unlabeled+Uncertain={edema_unlabeled}")
            print(f"  Subtype (P(subtype|edema=1)): Non-cardio={subtype_1}, Cardio={subtype_2}")
            print(f"{'='*80}")

    return train_df, val_df, test_df


# dataset split with stratification
# def split_dataset(merged_df, train_ratio, val_ratio, random_seed=0):
#     """
#     환자 레벨에서 층화추출을 수행하여 train/val/test split
#     각 환자의 라벨을 고려하여 분할하여 각 split의 라벨 비율을 균등하게 유지
#     """
#     stay_ids = merged_df['stay_id'].unique()

#     # 각 환자의 대표 라벨 결정 (우선순위: cardio > noncardio > negative > unlabeled)
#     stay_labels = []
#     for stay_id in stay_ids:
#         stay_data = merged_df[merged_df['stay_id'] == stay_id]
#         labels = stay_data['label'].to_numpy()

#         if np.any(labels == 2):  # Cardio
#             stay_labels.append(2)
#         elif np.any(labels == 1):  # Non-cardio
#             stay_labels.append(1)
#         elif np.any(labels == 0):  # Negative
#             stay_labels.append(0)
#         else:  # Unlabeled (all NaN or -1)
#             stay_labels.append(-1)

#     # 층화추출로 train/temp split
#     train_stay_ids, temp_stay_ids = train_test_split(
#         stay_ids,
#         test_size = (1 - train_ratio),
#         random_state=random_seed,
#         stratify=stay_labels
#     )

#     # temp의 라벨 추출
#     temp_stay_labels = []
#     for stay_id in temp_stay_ids:
#         stay_data = merged_df[merged_df['stay_id'] == stay_id]
#         labels = stay_data['label'].to_numpy()

#         if np.any(labels == 2):
#             temp_stay_labels.append(2)
#         elif np.any(labels == 1):
#             temp_stay_labels.append(1)
#         elif np.any(labels == 0):
#             temp_stay_labels.append(0)
#         else:
#             temp_stay_labels.append(-1)

#     val_size = val_ratio / (1 - train_ratio)

#     # 층화추출로 val/test split
#     val_stay_ids, test_stay_ids = train_test_split(
#         temp_stay_ids,
#         test_size = (1 - val_size),
#         random_state=random_seed,
#         stratify=temp_stay_labels
#     )

#     train_df = merged_df[merged_df['stay_id'].isin(train_stay_ids)]
#     val_df = merged_df[merged_df['stay_id'].isin(val_stay_ids)]
#     test_df = merged_df[merged_df['stay_id'].isin(test_stay_ids)]

#     print("\n[Dataset Split] Stratified patient-level distribution:")
#     print(f"Train patients: {len(train_stay_ids)}")
#     print(f"Val patients:   {len(val_stay_ids)}")
#     print(f"Test patients:  {len(test_stay_ids)}")

#     return train_df, val_df, test_df


class StratifiedPatientSampler(Sampler):
    """
    층화추출 기반 Patient-level 배치 샘플러 (오버샘플링 없음)
    - 전체 환자를 윈도우 라벨 비율에 맞게 섞어서 배치 구성
    - B=batch_size (한 배치에 여러 환자의 윈도우)
    - 윈도우 라벨 비율을 자연스럽게 유지
    """
    def __init__(self, dataset, batch_size=32, accelerator=None, shuffle=True, drop_last=True, seed=42, split=None):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = seed
        self.split = split

        # 전체 환자 인덱스
        self.patient_indices = list(range(len(dataset)))

        # DDP 설정 (현재 미사용)
        if self.accelerator is not None:
            self.world_size = self.accelerator.num_processes
            self.rank = self.accelerator.process_index
        else:
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
            self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Epoch마다 환자 순서를 섞고 배치 생성"""
        random.seed(self.base_seed + epoch)
        patients = self.patient_indices.copy()

        if self.shuffle:
            random.shuffle(patients)

        # batch_size명의 환자를 하나의 미니 배치로 생성함.
        self.batches = []
        for i in range(0, len(patients), self.batch_size):
            batch = patients[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                self.batches.append(batch)

        # DDP alignment
        remainder = len(self.batches) % self.world_size

        if self.drop_last and remainder != 0:
            self.batches = self.batches[:len(self.batches) - remainder]
            if self.accelerator is None or self.accelerator.is_main_process:
                print(f"[StratifiedPatientSampler] drop_last=True: {remainder} batches removed for DDP alignment")
        elif not self.drop_last and remainder != 0:
            pad_need = self.world_size - remainder
            for _ in range(pad_need):
                self.batches.append(random.choice(self.batches))
            if self.accelerator is None or self.accelerator.is_main_process:
                print(f"[StratifiedPatientSampler] drop_last=False: {pad_need} batches padded for DDP alignment")

        if self.accelerator is None or self.accelerator.is_main_process:
            split_tag = f"[{self.split.upper()}]" if self.split else ""
            print(f"\n[StratifiedPatientSampler]{split_tag} Epoch {epoch} initialized")
            print(f"Batch size (patients per batch): {self.batch_size}")
            print(f"Total batches: {len(self.batches)}")
            print(f"Batches per GPU: {len(self.batches) // self.world_size}")
            print(f"Total patients: {len(patients)}")

    def __iter__(self):
        """각 GPU에 배치를 균등하게 분배"""
        my_batches = [self.batches[i] for i in range(self.rank, len(self.batches), self.world_size)]

        if self.accelerator is None or self.accelerator.is_main_process:
            print(f"[StratifiedPatientSampler][Rank {self.rank}] Yielding {len(my_batches)} batches")

        for batch in my_batches:
            yield batch

    def __len__(self):
        """각 GPU가 처리할 배치 수"""
        return len(self.batches) // self.world_size

    def get_actual_class_distribution(self):
        """
        - 실제 배치 구성에서의 윈도우 레벨 클래스 분포 반환
        - 오버샘플링이 없으므로 데이터셋 원본 분포와 동일
        - Unlabeled 윈도우(label=-1)는 Loss에서 ignore되므로 제외하고 계산
        """
        # Check if multi-task
        is_multitask = len(self.dataset.label_metadata) > 0 and 'edema_label_series' in self.dataset.label_metadata[0]

        if is_multitask:
            # Multi-task: count edema distribution
            edema_0_windows = 0
            edema_1_windows = 0
            unlabeled_windows = 0

            for batch in self.batches:
                for patient_idx in batch:
                    meta = self.dataset.label_metadata[patient_idx]
                    for label in meta['edema_label_series']:
                        if label == 0:
                            edema_0_windows += 1
                        elif label == 1:
                            edema_1_windows += 1
                        else:
                            unlabeled_windows += 1

            valid_total = edema_0_windows + edema_1_windows

            if valid_total == 0:
                return {'edema_0': 0.5, 'edema_1': 0.5, 'edema_0_count': 0, 'edema_1_count': 0}

            distribution = {
                'edema_0': edema_0_windows / valid_total,
                'edema_1': edema_1_windows / valid_total,
                'edema_0_count': edema_0_windows,
                'edema_1_count': edema_1_windows,
            }

            if self.accelerator is None or self.accelerator.is_main_process:
                print(f"\n[StratifiedPatientSampler] Window-level edema distribution (유효 라벨만):")
                print(f"  No edema (0): {edema_0_windows:,} windows ({distribution['edema_0']:.2%})")
                print(f"  Has edema (1): {edema_1_windows:,} windows ({distribution['edema_1']:.2%})")

        else:
            # Legacy: count 3-class distribution
            cardio_windows = 0
            noncardio_windows = 0
            negative_windows = 0
            unlabeled_windows = 0

            for batch in self.batches:
                for patient_idx in batch:
                    meta = self.dataset.label_metadata[patient_idx]
                    for label in meta['label_series']:
                        if label == 2:
                            cardio_windows += 1
                        elif label == 1:
                            noncardio_windows += 1
                        elif label == 0:
                            negative_windows += 1
                        else:
                            unlabeled_windows += 1

            valid_total = cardio_windows + noncardio_windows + negative_windows

            if valid_total == 0:
                return {'cardio': 0.33, 'noncardio': 0.33, 'negative': 0.34, 'cardio_count': 0, 'noncardio_count': 0, 'negative_count': 0}

            distribution = {
                'cardio': cardio_windows / valid_total,
                'noncardio': noncardio_windows / valid_total,
                'negative': negative_windows / valid_total,
                'negative_count': negative_windows,
                'noncardio_count': noncardio_windows,
                'cardio_count': cardio_windows
            }

            if self.accelerator is None or self.accelerator.is_main_process:
                print(f"\n[StratifiedPatientSampler] Window-level class distribution (유효 라벨만):")
                print(f"  Cardio: {cardio_windows:,} windows ({distribution['cardio']:.2%})")
                print(f"  Non-cardio: {noncardio_windows:,} windows ({distribution['noncardio']:.2%})")
                print(f"  Negative: {negative_windows:,} windows ({distribution['negative']:.2%})")
            print(f"  Unlabeled (CE에서 제외됨): {unlabeled_windows:,} windows")
            print(f"  Total valid: {valid_total:,} windows")

        return distribution