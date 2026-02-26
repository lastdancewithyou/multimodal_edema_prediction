import os
import numpy as np
from sklearn.model_selection import train_test_split
import random
import hashlib
from tqdm.auto import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler

from utils import timer, seed_worker


CACHED_IMAGE_DIR = "/home/DAHS1/gangmin/my_research/CXR/pt_20260128/"


class SCL_Multi_Dataset(Dataset):
    def __init__(self, args, merged_df, demo_df):
        self.args = args
        self.window_size = args.window_size
        self.stride = args.stride
        self.prediction_horizon = args.prediction_horizon

        self.merged_df = merged_df
        self.stay_groups = self.merged_df.groupby('stay_id') # stay_id 식별자를 기준으로 grouping
        self.stay_ids = list(self.stay_groups.groups.keys())

        exclude_cols = ['hadm_id', 'stay_id', 'hour_slot', 'Edema', 'subtype_label', 'cxr_flag', 'hash_path', 'text_flag', 'tokenized_text']
        all_feature_cols = [col for col in self.merged_df.columns if col not in exclude_cols]

        self.ts_features = all_feature_cols  # Includes both features and observed_mask
        print(f"[Dataset] Total features (including observed_mask): {len(self.ts_features)}")

        self.demo_df = demo_df

        # ========== Demo 사용 여부 제어 ==========
        if self.demo_df is not None:
            exclude_demo_cols = ['hadm_id']
            self.demo_cols = [col for col in self.demo_df.columns if col not in exclude_demo_cols]
            self.num_demo_features = len(self.demo_cols)

            self.hadm_to_demo = {}
            for _, row in self.demo_df.iterrows():
                hadm_id = row['hadm_id']
                demo_values = [row[col] for col in self.demo_cols]
                self.hadm_to_demo[hadm_id] = demo_values

        else:
            self.demo_cols = []
            self.num_demo_features = 0
            self.hadm_to_demo = {}

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

        # ========== image / text mapping 사전 구축 ==========
        # collate_fn에서 배치를 구성할 때, 중복 이미지/텍스트를 제거하고 unique한 것만 인코딩하기 위함
        self.image_map = {}
        self.text_map = {}

        for stay_id in self.stay_ids:
            stay_data = self.stay_groups.get_group(stay_id).sort_values('hour_slot')

            # cxr_flag == 1인 hour_slot만 매핑에 추가함.
            img_dict = {t: path for t, path, flag in zip(
                stay_data['hour_slot'], 
                stay_data['hash_path'],
                stay_data['cxr_flag']
            ) if flag == 1}

            # text_flag == 1인 hour_slot만 매핑에 추가함.
            text_dict = {t: token for t, token, flag in zip(
                stay_data['hour_slot'],
                stay_data['tokenized_text'],
                stay_data['text_flag']
            ) if flag == 1}

            self.image_map[stay_id] = img_dict
            self.text_map[stay_id] = text_dict

        # ========== Image caching ==========
        # 디스크 I/O 최소화를 위해 로드된 이미지를 메모리에 캐싱함. (안할 경우 데이터로더 로드 멈추는 우려 있음.)
        self.image_cache = {}

        self.to_3ch = args.img_to_3ch # DenseNet: 1채널 grayscale, ResNet: 3채널 RGB (channel 맞춰줘야 함.)
        print(f"[Dataset] Image preprocessing config: to_3ch={self.to_3ch}")

    def load_image_cached(self, hash_filename):
        """
        - 사전에 .pt 형식으로 전처리된 이미지를 불러오고 캐싱하는 함수
        1. 캐시에 있으면 즉시 반환
        2. 캐시에 없으면 디스크에서 로드
        3. to_3ch 작동
        4. 캐시에 저장 후 반환
        """
        if not hash_filename or not isinstance(hash_filename, str) or hash_filename.strip() == "":
            raise RuntimeError(
                f"[ERROR] Invalid image filename: {hash_filename}\n"
            )

        # Check cache
        if hash_filename in self.image_cache:
            return self.image_cache[hash_filename]

        # Load from disk
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

        # 각 hour_slot에 이미지와 텍스트가 있으면 hour_slot 값, 없으면 -1
        img_index_series = [t if t in self.image_map[stay_id] else -1 for t in hour_slots]
        text_index_series = [t if t in self.text_map[stay_id] else -1 for t in hour_slots]

        sequence_series, edema_label_series, subtype_label_series, has_cxr_series, has_text_series, valid_mask_series = [], [], [], [], [], []
        L = len(stay_data)

        # ========== Sliding Window 생성 ==========
        # ICU stay 길이가 window_size + prediction_horizon 이상인 경우
        if L >= self.window_size + self.prediction_horizon:

            max_start_idx = L - self.window_size - self.prediction_horizon  # Window의 시점 후 prediction_horizon 이후에도 라벨이 존재해야 함

            # stride에 따른 window 생성
            for i in range(0, max_start_idx + 1, self.stride):
                window_hours = hour_slots[i:i + self.window_size]
                window_ts = ts_feature_values[i:i + self.window_size]
                window_img = img_index_series[i:i + self.window_size]
                window_text = text_index_series[i:i + self.window_size]

                window_sequence = [
                    {
                        'time_step': int(window_hours[j]),
                        'ts_features': window_ts[j],  # Includes observed_mask_* as regular features
                        'img_index': window_img[j],
                        'txt_index': window_text[j]
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

        # 인구통계학적 정보 (hadm_id를 기준으로 함)
        demo_features = None
        if self.demo_df is not None and len(self.demo_cols) > 0:
            hadm_id = stay_data['hadm_id'].iloc[0]

            if hadm_id in self.hadm_to_demo:
                demo_values = self.hadm_to_demo[hadm_id]
                demo_features = torch.tensor(demo_values, dtype=torch.float32)
        
        return {
            'stay_id': stay_id,
            'modality_series': sequence_series,
            'has_cxr': has_cxr_series,
            'has_text': has_text_series,
            'edema_label_series': edema_label_series,
            'subtype_label_series': subtype_label_series,
            'window_mask': window_mask,
            'valid_mask_series' : valid_mask_series,
            'demo_features': demo_features
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

        # ==================== 배치 내 고유 이미지/텍스트 추출 ====================
        unique_img_paths = []  # 배치 전체에서 중복 제거된 이미지 경로
        unique_txt_keys = []   # 배치 전체에서 중복 제거된 텍스트 키 (stay_id, hour_slot)

        img_path_to_idx = {}  # {hash_path: unique_list_idx}
        txt_key_to_idx = {}   # {(stay_id, hour_slot): unique_list_idx}

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

                    # 텍스트 키 추출 (stay_id, hour_slot)
                    txt_hour = step['txt_index']  # hour_slot or -1
                    if txt_hour != -1:
                        txt_key = (stay_id, txt_hour)
                        if txt_key not in txt_key_to_idx:
                            txt_key_to_idx[txt_key] = len(unique_txt_keys)
                            unique_txt_keys.append(txt_key)

        # ==================== 텐서 초기화 ====================
        ts_feature_dim = batch[0]['modality_series'][0][0]['ts_features'].shape[-1]
        ts_tensor = torch.zeros(len(batch), max_windows, args.window_size, ts_feature_dim)

        # -1: 이미지/텍스트 없음, 0~N-1: unique_list의 인덱스
        img_index_tensor = torch.full((len(batch), max_windows, args.window_size), fill_value=-1, dtype=torch.long)
        text_index_tensor = torch.full((len(batch), max_windows, args.window_size), fill_value=-1, dtype=torch.long)

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

            window_mask_tensor[i, :num_windows] = torch.tensor(window_mask_list[i][:num_windows], dtype=torch.bool)
            edema_label_tensor[i, :num_windows] = torch.tensor(edema_label_series_list[i], dtype=torch.long)
            subtype_label_tensor[i, :num_windows] = torch.tensor(subtype_label_series_list[i], dtype=torch.long)

        # ==================== Demographic features ====================
        demo_features_list = [item['demo_features'] for item in batch]
        demographic_tensor = None
        if demo_features_list[0] is not None:
            demographic_tensor = torch.stack(demo_features_list, dim=0)

        return {
            'stay_ids': stay_ids,
            'ts_tensor': ts_tensor,                      # [B, W, T, D] - Features and observed_mask
            'time_steps': time_steps_tensor,             # [B, W, T] - hour_slot
            'img_index_tensor': img_index_tensor,        # [B, W, T] → 0~N-1 (unique_img_paths 인덱스)
            'text_index_tensor': text_index_tensor,      # [B, W, T] → 0~M-1 (unique_txt_keys 인덱스)
            'unique_img_paths': unique_img_paths,        # List[str] - 배치 내 고유 이미지 경로 (길이 N)
            'unique_txt_keys': unique_txt_keys,          # List[(stay_id, hour_slot)] - 배치 내 고유 텍스트 키 (길이 M)
            'window_mask': window_mask_tensor,           # [B, W]
            'valid_seq_mask': valid_seq_mask_tensor,     # [B, W, T]
            'edema_labels': edema_label_tensor,          # [B, W] - Binary edema labels
            'subtype_labels': subtype_label_tensor,      # [B, W] - Subtype labels (0, 1, 2, or -1)
            'demo_features': demographic_tensor          # [B, D_demo] or None
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
def get_dataloaders(ts_df, cxr_df, text_df, demo_df, args, accelerator=None):
    if not args.use_demographic:
        demo_df = None
    
    # 데이터 병합
    with timer("Dataset 병합"):
        merged_df = merged_dataframes(ts_df, cxr_df, text_df)

    # 데이터 분할
    train_df, val_df, test_df = split_dataset(
        merged_df=merged_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )

    with timer("Dataset 생성"):
        train_dataset = SCL_Multi_Dataset(args, train_df, demo_df)
        val_dataset = SCL_Multi_Dataset(args, val_df, demo_df)
        test_dataset = SCL_Multi_Dataset(args, test_df, demo_df)
        args.num_demo_features = train_dataset.num_demo_features

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
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            collate_fn=train_collate_fn,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=seed_worker
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_sampler=val_sampler,
            collate_fn=val_collate_fn,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=seed_worker
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_sampler=test_sampler,
            collate_fn=test_collate_fn,
            pin_memory=True,
            num_workers=8,
            worker_init_fn=seed_worker
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
def merged_dataframes(ts_df, img_df, text_df): 
    merged_df = (ts_df
            .merge(img_df, on=['stay_id', 'hour_slot'], how='outer')
            .merge(text_df, on=['stay_id','hour_slot'], how='outer')
    )
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


########################################################################################################
########################################################################################################
# Grave of codes
# class LabelBasedSampler(Sampler):
#     def __init__(self, dataset, cardio, noncardio, negative, unlabeled, index_map, batch_size, label_ratio=[3,2,5], accelerator=None, drop_last=True):
#         super().__init__()
#         self.dataset = dataset
#         self.accelerator = accelerator

#         # Stay IDs → dataset index 변환 준비
#         self.index_map = index_map

#         valid_stay_ids = {
#             dataset.stay_ids[i] 
#             for i, meta in enumerate(dataset.label_metadata)
#             if len(meta["label_series"]) > 0    # 유효 window가 1개 이상일 때만s
#         }

#         # 전수 환자 리스트 (중복 없음)
#         self.cardio_patients = [sid for sid in set(cardio) if sid in valid_stay_ids]
#         self.noncardio_patients = [sid for sid in set(noncardio) if sid in valid_stay_ids]
#         self.neg_patients = [sid for sid in set(negative + unlabeled) if sid in valid_stay_ids]
#         print(f"[Sampler] Filtered valid patients: Cardio={len(self.cardio_patients)}, "
#             f"Non-cardio={len(self.noncardio_patients)}, Negative + Unlabeled={len(self.neg_patients)}")

#         # Ratio
#         total_ratio = sum(label_ratio)
#         self.cardio_ratio = label_ratio[0] / total_ratio
#         self.noncardio_ratio = label_ratio[1] / total_ratio
#         self.neg_ratio = label_ratio[2] / total_ratio

#         self.batch_size = batch_size
#         self.drop_last = drop_last

#         # DDP
#         if self.accelerator is not None:
#             self.world_size = self.accelerator.num_processes
#             self.rank = self.accelerator.process_index
#         else:
#             self.world_size = dist.get_world_size() if dist.is_initialized() else 1
#             self.rank = dist.get_rank() if dist.is_initialized() else 0

#         self.set_epoch(0)

#     # ---------------------------------------------------------
#     #  Positive 전수 사용 + 부족하면 oversampling
#     # ---------------------------------------------------------
#     def _oversample_if_needed(self, pool, target_count):
#         if len(pool) >= target_count:
#             return pool.copy()
#         need = target_count - len(pool)
#         return pool + random.choices(pool, k=need)

#     # ---------------------------------------------------------
#     #  메인 배치 생성 (rank 고려 X)
#     # ---------------------------------------------------------
#     def _generate_batches(self):
#         # batch 내 각 레이블 수
#         C = max(1, int(self.batch_size * self.cardio_ratio))
#         N = max(1, int(self.batch_size * self.noncardio_ratio))
#         R = self.batch_size - C - N  # negative/unlabeled

#         if self.accelerator is None or self.accelerator.is_main_process:
#             print(f"\n[Sampler] Generating batches...")
#             print(f"[Sampler] Batch label counts → cardio={C}, noncardio={N}, Neg+Unlabeled={R}")
#             print(f"[Sampler] Pools BEFORE oversampling: cardio={len(self.cardio_patients)}, "
#                 f"noncardio={len(self.noncardio_patients)}, Neg+Unlabeled={len(self.neg_patients)}")

#         # 1) cardio/noncardio는 "전수 순회용 리스트"를 먼저 만든 뒤 → 그 다음에야 oversampling
#         cardio_order = self.cardio_patients.copy()
#         noncardio_order = self.noncardio_patients.copy()
#         random.shuffle(cardio_order)
#         random.shuffle(noncardio_order)

#         c_idx = 0  # cardio_order에서 현재 어디까지 썼는지
#         n_idx = 0  # noncardio_order에서 현재 어디까지 썼는지

#         # 2) negative/unlabeled는 전수 사용 (중복 없음)
#         neg_pool = self.neg_patients.copy()
#         random.shuffle(neg_pool)

#         batches = []

#         # === used patient tracking (unused 계산용) ===
#         used_cardio = set()
#         used_noncardio = set()
#         used_neg = set()

#         # 여러 epoch 동안 간헐적으로 남는 neg_pool 때문에 불균형이 생길 수 있음
#         # neg_pool이 R개 미만으로 남으면 더 이상 full batch 생성 불가
#         while len(neg_pool) >= R:
#             batch_ids = []

#             # ---------- cardio 채우기 (전수 순회 우선) ----------
#             need_c = C
#             # 1) 아직 안 쓴 cardio_order에서 최대한 가져오기
#             remain_c = len(cardio_order) - c_idx
#             take_c = min(need_c, remain_c)
#             if take_c > 0:
#                 batch_ids += cardio_order[c_idx:c_idx+take_c]
#                 used_cardio.update(cardio_order[c_idx:c_idx+take_c])
#                 c_idx += take_c
#                 need_c -= take_c

#             # 2) 그래도 부족하면 이제서야 oversampling
#             if need_c > 0 and len(self.cardio_patients) > 0:
#                 extra_c = random.choices(self.cardio_patients, k=need_c)
#                 batch_ids += extra_c
#                 used_cardio.update(extra_c)

#             # ---------- noncardio 채우기 (전수 순회 우선) ----------
#             need_n = N
#             remain_n = len(noncardio_order) - n_idx
#             take_n = min(need_n, remain_n)
#             if take_n > 0:
#                 batch_ids += noncardio_order[n_idx:n_idx+take_n]
#                 used_noncardio.update(noncardio_order[n_idx:n_idx+take_n])
#                 n_idx += take_n
#                 need_n -= take_n

#             # 2) 그래도 부족하면 oversampling
#             if need_n > 0 and len(self.noncardio_patients) > 0:
#                 extra_n = random.choices(self.noncardio_patients, k=need_n)
#                 batch_ids += extra_n
#                 used_noncardio.update(extra_n)

#             # ---------- negative/unlabeled 채우기 (전수 사용, 중복 없음) ----------
#             batch_ids += neg_pool[:R]
#             used_neg.update(neg_pool[:R])
#             neg_pool = neg_pool[R:]

#             # safety: positive 한 명은 반드시 들어가야 함
#             if not any(sid in self.cardio_patients or sid in self.noncardio_patients for sid in batch_ids):
#                 if len(self.cardio_patients + self.noncardio_patients) > 0:
#                     batch_ids[0] = random.choice(self.cardio_patients + self.noncardio_patients)

#             random.shuffle(batch_ids)
#             batches.append([self.index_map[sid] for sid in batch_ids])

#         # ---------- 마지막 batch 처리 (negative 부족) ----------
#         # 여기서는 "전수 순회 우선 + 부족시 oversampling" 로직을 그대로 사용하되,
#         # neg_pool은 남은 것만 (있다면) 사용
#         remain_c = len(cardio_order) - c_idx
#         remain_n = len(noncardio_order) - n_idx

#         if remain_c + remain_n > 0:
#             fb = []

#             # cardio fill
#             need_c = C
#             take_c = min(need_c, remain_c)
#             if take_c > 0:
#                 fb += cardio_order[c_idx:c_idx+take_c]
#                 used_cardio.update(cardio_order[c_idx:c_idx+take_c])
#                 c_idx += take_c
#                 need_c -= take_c
#             if need_c > 0 and len(self.cardio_patients) > 0:
#                 extra_c = random.choices(self.cardio_patients, k=need_c)
#                 fb += extra_c
#                 used_cardio.update(extra_c)

#             # noncardio fill
#             need_n = N
#             take_n = min(need_n, remain_n)
#             if take_n > 0:
#                 fb += noncardio_order[n_idx:n_idx+take_n]
#                 used_noncardio.update(noncardio_order[n_idx:n_idx+take_n])
#                 n_idx += take_n
#                 need_n -= take_n
#             if need_n > 0 and len(self.noncardio_patients) > 0:
#                 extra_n = random.choices(self.noncardio_patients, k=need_n)
#                 fb += extra_n
#                 used_noncardio.update(extra_n)

#             # 남은 neg들 전부 추가 (oversample X)
#             if len(neg_pool) > 0:
#                 fb += neg_pool
#                 used_neg.update(neg_pool)
#                 neg_pool = []

#             if len(fb) >= (C + N):  # 최소 positive 수는 맞추고
#                 random.shuffle(fb)
#                 batches.append([self.index_map[sid] for sid in fb])
#                 if self.accelerator is None or self.accelerator.is_main_process:
#                     print(f"[Sampler] Final batch created: {len(fb)} samples "
#                         f"(cardio={C}, noncardio={N}, neg={len(fb)-C-N})")

#         # ---------- 남은 환자/unused positive 통계 ----------
#         if self.accelerator is None or self.accelerator.is_main_process:
#             unused_cardio = len(self.cardio_patients) - len(used_cardio)
#             unused_noncardio = len(self.noncardio_patients) - len(used_noncardio)
#             unused_neg = len(self.neg_patients) - len(used_neg)

#             if unused_neg > 0:
#                 print(f"[Sampler] Warning: {unused_neg} negative/unlabeled patients unused "
#                     f"(not enough positive slots for full batches)")

#             if unused_cardio > 0 or unused_noncardio > 0:
#                 print(f"[Sampler] Unused positive patients: cardio={unused_cardio}, noncardio={unused_noncardio}")

#         # ---------- DDP-safe 패딩/드롭 ----------
#         remainder = len(batches) % self.world_size

#         if self.drop_last:
#             if remainder != 0:
#                 batches = batches[:len(batches) - remainder]
#                 if self.accelerator is None or self.accelerator.is_main_process:
#                     print(f"[Sampler] drop_last=True: {remainder} batches removed for DDP alignment")
#         else:
#             if remainder != 0:
#                 pad_need = self.world_size - remainder
#                 for _ in range(pad_need):
#                     batches.append(random.choice(batches))
#                 if self.accelerator is None or self.accelerator.is_main_process:
#                     print(f"[Sampler] drop_last=False: {pad_need} batches padded for DDP alignment")

#         # === 최종 배치 요약 ===
#         total_c = total_nc = total_n = 0
#         for batch in batches:
#             for idx in batch:
#                 sid = self.dataset.stay_ids[idx]
#                 if sid in self.cardio_patients:
#                     total_c += 1
#                 elif sid in self.noncardio_patients:
#                     total_nc += 1
#                 else:
#                     total_n += 1

#         tot = total_c + total_nc + total_n
#         if self.accelerator is None or self.accelerator.is_main_process:
#             print(f"\n[Sampler] FINAL BATCH SUMMARY:")
#             print(f"  Total batches: {len(batches)} (after drop_last={self.drop_last})")
#             print(f"  cardio={total_c} ({total_c/tot:.2%}), "
#                 f"noncardio={total_nc} ({total_nc/tot:.2%}), "
#                 f"Negative+UnLabeled={total_n} ({total_n/tot:.2%})")
#         return batches

#     def _precompute_class_stats(self):
#         """
#         배치별 클래스 정보를 사전 계산하여 저장
#         __iter__()와 get_actual_class_distribution()의 반복 계산 제거 (성능 최적화)
#         """
#         self.batch_class_counts = []  # [(cardio_count, noncardio_count, neg_count), ...]

#         for batch in self.batches:
#             c = nc = n = 0
#             for idx in batch:
#                 sid = self.dataset.stay_ids[idx]
#                 if sid in self.cardio_patients:
#                     c += 1
#                 elif sid in self.noncardio_patients:
#                     nc += 1
#                 else:
#                     n += 1
#             self.batch_class_counts.append((c, nc, n))

#     # ---------------------------------------------------------
#     def set_epoch(self, epoch):
#         random.seed(epoch)
#         self.batches = self._generate_batches()
#         self._precompute_class_stats()
#         print(f"[Sampler] Epoch {epoch} initialized — total batches={len(self.batches)}")

#     # ---------------------------------------------------------
#     def __iter__(self):
#         # === GPU 할당 이전: 전체 배치 통계 (사전 계산된 데이터 사용) ===
#         c_all = sum(c for c, _, _ in self.batch_class_counts)
#         nc_all = sum(nc for _, nc, _ in self.batch_class_counts)
#         n_all = sum(n for _, _, n in self.batch_class_counts)

#         t_all = c_all + nc_all + n_all
#         print(f"\n[Sampler] ALL GPUs TOTAL: batches={len(self.batches)} | "
#             f"cardio={c_all}({c_all/t_all:.2%}), noncardio={nc_all}({nc_all/t_all:.2%}), Neg+Unlabeled={n_all}({n_all/t_all:.2%})")

#         # === GPU 할당 이후: rank-based slicing ===
#         my_batch_indices = list(range(self.rank, len(self.batches), self.world_size))
#         my_batches = [self.batches[i] for i in my_batch_indices]

#         # === 각 Rank별 통계 (사전 계산된 데이터 사용) ===
#         c = sum(self.batch_class_counts[i][0] for i in my_batch_indices)
#         nc = sum(self.batch_class_counts[i][1] for i in my_batch_indices)
#         n = sum(self.batch_class_counts[i][2] for i in my_batch_indices)

#         t = c + nc + n
#         print(f"[Sampler][Rank {self.rank}] GPU 할당 후: batches={len(my_batches)} | "
#             f"cardio={c}({c/t:.2%}), noncardio={nc}({nc/t:.2%}), Neg+Unlabeled={n}({n/t:.2%})")

#         for batch in my_batches:
#             yield batch

#     def __len__(self):
#         local_batches = len(self.batches) // self.world_size
#         # 첫 호출시만 로깅 (rank 0에서)
#         if self.rank == 0 and not hasattr(self, '_logged'):
#             print(f"\n[Sampler] Total batches across all GPUs: {len(self.batches)}")
#             print(f"[Sampler] World size (number of GPUs): {self.world_size}")
#             print(f"[Sampler] Batches per GPU: {local_batches}")
#             print(f"[Sampler] Calculation: {len(self.batches)} / {self.world_size} = {local_batches}\n")
#             self._logged = True
#         return local_batches

#     def get_actual_class_distribution(self):
#         """
#         실제 배치 구성에서의 클래스 분포 반환
#         사전 계산된 통계 정보 사용 (성능 최적화)
#         동적 클래스 가중치 계산에 사용

#         Returns:
#             dict: {
#                 'cardio': float (0.0 ~ 1.0),
#                 'noncardio': float (0.0 ~ 1.0),
#                 'negative': float (0.0 ~ 1.0)
#             }
#         """
#         cardio_count = sum(c for c, _, _ in self.batch_class_counts)
#         noncardio_count = sum(nc for _, nc, _ in self.batch_class_counts)
#         neg_unlabeled_count = sum(n for _, _, n in self.batch_class_counts)

#         total = cardio_count + noncardio_count + neg_unlabeled_count

#         if total == 0:
#             return {
#                 'cardio': 0.33,
#                 'noncardio': 0.33,
#                 'negative': 0.34
#             }

#         distribution = {
#             'cardio': cardio_count / total,
#             'noncardio': noncardio_count / total,
#             'negative': neg_unlabeled_count / total
#         }

#         print(f"\n[LabelBasedSampler] Actual class distribution (for dynamic weights):")
#         print(f"  Cardio: {cardio_count:,} samples ({distribution['cardio']:.2%})")
#         print(f"  Non-cardio: {noncardio_count:,} samples ({distribution['noncardio']:.2%})")
#         print(f"  Negative + Unlabeled: {neg_unlabeled_count:,} samples ({distribution['negative']:.2%})")
#         return distribution


# class StratifiedMemoryAwareSampler(Sampler):
#     def __init__(
#         self,
#         dataset,
#         cardio,
#         noncardio, 
#         negative,
#         unlabeled,
#         patient_to_index,
#         index_to_stay_id,
#         base_batch_size=16,
#         max_windows=2000,
#         max_img_per_batch=20000,
#         max_txt_per_batch=20000,
#         shuffle=True,
#         drop_last=True,
#         seed=42
#     ):
#         self.dataset = dataset
#         self.cardio = cardio
#         self.noncardio = noncardio
#         self.negative = negative
#         self.unlabeled = unlabeled
#         self.index_map = patient_to_index
#         self.stay_id_map = index_to_stay_id

#         self.base_batch_size = base_batch_size
#         self.max_windows = max_windows
#         self.max_img_per_batch = max_img_per_batch
#         self.max_txt_per_batch = max_txt_per_batch

#         self.shuffle = shuffle
#         self.drop_last = drop_last
#         self.base_seed = seed
#         self.seed = seed
#         self.rng = random.Random(self.seed)

#         # 데이터셋 전체 인덱스 준비
#         self.lengths = [len(dataset.label_metadata[idx]['label_series']) for idx in range(len(dataset))]
        
#         # 라벨별 인덱스 분류
#         self.pos_cardio_indices = [self.index_map[pid] for pid in cardio if pid in self.index_map]
#         self.pos_noncardio_indices = [self.index_map[pid] for pid in noncardio if pid in self.index_map]
        
#         combined_neg = list(set(negative + unlabeled))
#         self.neg_unlabeled_indices = [self.index_map[pid] for pid in combined_neg if pid in self.index_map]

#         self.set_epoch(0)

#         print(f"Total patients(stay ids): {len(dataset)}")
#         print(f"Cardio: {len(self.pos_cardio_indices)}")
#         print(f"Non-cardio: {len(self.pos_noncardio_indices)}")
#         print(f"Negative + Unlabeled: {len(self.neg_unlabeled_indices)}")

#     def _can_add_to_batch(self, candidate_idx, current_windows, current_img, current_txt):
#         """배치에 샘플을 추가할 수 있는지 메모리 제약 확인"""
#         L = self.lengths[candidate_idx]
#         img = self.dataset.img_token_counts[candidate_idx]
#         txt = self.dataset.txt_token_counts[candidate_idx]

#         if self.max_windows is not None and current_windows + L > self.max_windows:
#             return False
#         if current_img + img > self.max_img_per_batch:
#             return False
#         if current_txt + txt > self.max_txt_per_batch:
#             return False
#         return True
    
#     def _generate_batches(self):
#         cardio_pool = self.pos_cardio_indices.copy()
#         noncardio_pool = self.pos_noncardio_indices.copy()
#         neg_pool = self.neg_unlabeled_indices.copy()

#         print(f"\n[_generate_batches] BEFORE oversampling:")
#         print(f"  Cardio pool size: {len(cardio_pool)}")
#         print(f"  Noncardio pool size: {len(noncardio_pool)}")
#         print(f"  Negative + unlabeled pool size: {len(neg_pool)}")

#         # 전수 학습 보장: negative + unlabeled는 1 epoch에 한 번씩만 사용
#         if self.shuffle:
#             self.rng.shuffle(cardio_pool)
#             self.rng.shuffle(noncardio_pool)
#             self.rng.shuffle(neg_pool)

#         # --- 2. 오버샘플링 설정 ---
#         # Non-cardio가 sparse하므로 더 높은 오버샘플링 비율 적용
#         oversample_ratio_cardio = 1.5  # cardio: 1.5배 (충분함)
#         oversample_ratio_noncardio = 3.0  # noncardio: 2.5배 → 4.0배 (sparse 모달리티 대비)

#         num_cardio_oversample = int(len(cardio_pool) * (oversample_ratio_cardio - 1))
#         num_noncardio_oversample = int(len(noncardio_pool) * (oversample_ratio_noncardio - 1))

#         print(f"\n[_generate_batches] Oversampling amounts:")
#         print(f"  Cardio samples to add: {num_cardio_oversample} (ratio: {oversample_ratio_cardio})")
#         print(f"  Noncardio samples to add: {num_noncardio_oversample} (ratio: {oversample_ratio_noncardio})")

#         cardio_pool += self.rng.choices(cardio_pool, k=num_cardio_oversample)
#         noncardio_pool += self.rng.choices(noncardio_pool, k=num_noncardio_oversample)

#         print(f"\n[_generate_batches] AFTER oversampling:")
#         print(f"  Cardio pool size: {len(cardio_pool)}")
#         print(f"  Noncardio pool size: {len(noncardio_pool)}")
#         print(f"  Negative pool size: {len(neg_pool)}")

#         # --- 3. 전체 pool 섞기 ---
#         all_candidates = cardio_pool + noncardio_pool + neg_pool
#         self.rng.shuffle(all_candidates)

#         # --- 4. 메모리 제약을 고려한 배치 구성 ---
#         batches = []
#         i = 0
#         while i < len(all_candidates):
#             batch_indices = []
#             current_windows = current_img = current_txt = 0

#             while i < len(all_candidates) and len(batch_indices) < self.base_batch_size:
#                 idx = all_candidates[i]
#                 L = self.lengths[idx]
#                 img = self.dataset.img_token_counts[idx]
#                 txt = self.dataset.txt_token_counts[idx]

#                 if (self.max_windows is not None and current_windows + L > self.max_windows) or \
#                 (current_img + img > self.max_img_per_batch) or \
#                 (current_txt + txt > self.max_txt_per_batch):
#                     break

#                 batch_indices.append(idx)
#                 current_windows += L
#                 current_img += img
#                 current_txt += txt
#                 i += 1

#             if batch_indices:
#                 batches.append(batch_indices)
#             else:
#                 # 어떤 것도 추가할 수 없는 상황 (단일 샘플도 메모리 초과) → 강제로 하나라도 넣음
#                 batches.append([all_candidates[i]])
#                 i += 1

#         # 최종 샘플 수 계산
#         total_samples_in_batches = sum(len(batch) for batch in batches)
#         print(f"\n[_generate_batches] FINAL BATCHES:")
#         print(f"  Total batches: {len(batches)}")
#         print(f"  Total samples across all batches: {total_samples_in_batches}")
#         print(f"  Expected with oversampling: {len(cardio_pool) + len(noncardio_pool) + len(neg_pool)}")

#         # --- 5. GPU 분배 (이전과 동일) ---
#         # world_size, stratify, drop_last 처리 등은 self.__iter__에서 분할됨
#         return batches

#     def __iter__(self):
#         """Stratified sampling으로 각 GPU에 유사한 라벨 분포 할당"""
#         self.rng.seed(self.seed)
#         batches = self.batches.copy()
        
#         if self.shuffle:
#             self.rng.shuffle(batches)

#         # 라벨 집합 준비 (stratified split용)
#         cardio_set = set(self.cardio)
#         noncardio_set = set(self.noncardio)

#         if dist.is_available() and dist.is_initialized():
#             rank = dist.get_rank()
#             world_size = dist.get_world_size()

#             print(f"[StratifiedSampler Rank {rank}] 전체 배치 수: {len(batches)}")

#             batch_class_info = []
#             for batch_idx, batch in enumerate(batches):
#                 stay_ids = [self.stay_id_map[idx] for idx in batch]
#                 num_cardio = sum(1 for sid in stay_ids if sid in cardio_set)
#                 num_noncardio = sum(1 for sid in stay_ids if sid in noncardio_set)
#                 num_negative = len(stay_ids) - num_cardio - num_noncardio
#                 batch_class_info.append((batch_idx, batch, num_cardio, num_noncardio, num_negative))

#             # Window count 기반으로 stratified GPU 배치 분배
#             gpu_batches = [[] for _ in range(world_size)]
#             gpu_stats = [[0, 0, 0] for _ in range(world_size)]  # [cardio, noncardio, negative]
#             gpu_window_counts = [0 for _ in range(world_size)]  # 각 GPU의 총 window 수

#             for i, (batch_idx, batch, c, nc, neg_unlabeled) in enumerate(batch_class_info):
#                 batch_window_count = sum(self.lengths[idx] for idx in batch)
#                 target_gpu = min(range(world_size), key=lambda x: gpu_window_counts[x])
                
#                 gpu_batches[target_gpu].append(batch)
#                 gpu_stats[target_gpu][0] += c
#                 gpu_stats[target_gpu][1] += nc
#                 gpu_stats[target_gpu][2] += neg_unlabeled
#                 gpu_window_counts[target_gpu] += batch_window_count

#             if self.drop_last:
#                 min_batches = min(len(gpu_batch) for gpu_batch in gpu_batches)
#                 print(f"[StratifiedSampler Rank {rank}] Min batches across GPUs: {min_batches}")

#                 for gpu_id in range(world_size):
#                     original_count = len(gpu_batches[gpu_id])
#                     gpu_batches[gpu_id] = gpu_batches[gpu_id][:min_batches]
#                     if original_count > min_batches:
#                         print(f"[StratifiedSampler] GPU {gpu_id}: Dropped {original_count - min_batches} batches")

#             for gpu_id in range(world_size):
#                 c, nc, neg_unlabeled = gpu_stats[gpu_id]
#                 total = c + nc + neg_unlabeled
#                 if total > 0:
#                     print(f"[StratifiedSampler] GPU {gpu_id}: {len(gpu_batches[gpu_id])} batches, "
#                           f"Windows={gpu_window_counts[gpu_id]}, "
#                           f"Cardio={c}({c/total:.1%}), Non-cardio={nc}({nc/total:.1%}), Negative + Unlabeled={neg_unlabeled}({neg_unlabeled/total:.1%})")
                    
#             batches = gpu_batches[rank]

#         for batch in batches:
#             yield batch

#     def __len__(self):
#         if dist.is_available() and dist.is_initialized():
#             world_size = dist.get_world_size()
#             return (len(self.batches) // world_size) - 1  # drop_last 보장
#         else:
#             return len(self.batches)

#     def set_epoch(self, epoch):
#         self.seed = self.base_seed + epoch
#         self.rng.seed(self.seed)
#         self.batches = self._generate_batches()

#         if dist.is_available() and dist.is_initialized():
#             dist.barrier()

#     def get_actual_class_distribution(self):
#         """
#         실제 배치 구성에서의 클래스 분포 반환
#         생성된 배치들을 순회하며 실제 오버샘플링된 개수 계산
#         Sampler의 오버샘플링을 정확히 반영한 동적 class_weights 계산에 사용

#         Returns:
#             dict: {
#                 'cardio': float (0.0 ~ 1.0),
#                 'noncardio': float (0.0 ~ 1.0),
#                 'negative': float (0.0 ~ 1.0)
#             }
#         """
#         cardio_count = 0
#         noncardio_count = 0
#         neg_unlabeled_count = 0

#         cardio_set = set(self.cardio)
#         noncardio_set = set(self.noncardio)

#         # 생성된 모든 배치에서 클래스 개수 계산 (오버샘플링 반영됨!)
#         for batch_indices in self.batches:
#             for idx in batch_indices:
#                 stay_id = self.stay_id_map[idx]

#                 if stay_id in cardio_set:
#                     cardio_count += 1
#                 elif stay_id in noncardio_set:
#                     noncardio_count += 1
#                 else:
#                     neg_unlabeled_count += 1

#         total = cardio_count + noncardio_count + neg_unlabeled_count

#         if total == 0:
#             return {
#                 'cardio': 0.33,
#                 'noncardio': 0.33,
#                 'negative': 0.34
#             }

#         distribution = {
#             'cardio': cardio_count / total,
#             'noncardio': noncardio_count / total,
#             'negative': neg_unlabeled_count / total
#         }

#         print(f"\n[StratifiedSampler] Actual class distribution (오버샘플링 반영됨):")
#         print(f"  Cardio: {cardio_count:,} samples ({distribution['cardio']:.2%})")
#         print(f"  Non-cardio: {noncardio_count:,} samples ({distribution['noncardio']:.2%})")
#         print(f"  Negative + Unlabeled: {neg_unlabeled_count:,} samples ({distribution['negative']:.2%})")

#         return distribution

# def group_patients_by_label(dataset):
#     cardio_patients = set()
#     noncardio_patients = set()
#     negative_patients = set()
#     unlabeled_patients = set()

#     patient_to_index = {}
#     index_to_stay_id = {}

#     for idx, stay_id in tqdm(enumerate(dataset.stay_ids), total=len(dataset.stay_ids), desc="🗂️ Grouping"):
#         stay_df = dataset.stay_groups.get_group(stay_id).sort_values("hour_slot")
#         raw_labels = stay_df["label"].to_numpy()

#         if np.all(np.isnan(raw_labels)):
#             unlabeled_patients.add(stay_id)

#         elif np.any(raw_labels == 2):
#             cardio_patients.add(stay_id)

#         elif np.any(raw_labels == 1):
#             noncardio_patients.add(stay_id)

#         elif np.any(raw_labels == 0):
#             negative_patients.add(stay_id)

#         else:
#             unlabeled_patients.add(stay_id)

#         patient_to_index[stay_id] = idx
#         index_to_stay_id[idx] = stay_id

#     return list(cardio_patients), list(noncardio_patients), list(negative_patients), list(unlabeled_patients), patient_to_index, index_to_stay_id