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


from utils.utils import timer, seed_worker


class SCL_Multi_Dataset(Dataset):
    def __init__(self, args, merged_df, include_nan_labels=False, stride=None):
        self.args = args
        self.window_size = args.window_size
        self.stride = stride
        self.prediction_horizon = args.prediction_horizon
        self.include_nan_labels = include_nan_labels

        self.merged_df = merged_df
        self.stay_groups = self.merged_df.groupby('stay_id') # stay_id 식별자를 기준으로 grouping
        self.stay_ids = list(self.stay_groups.groups.keys())

        exclude_cols = [
            'subject_id', 'hadm_id', 'stay_id', 'hour_slot',
            # New Label
            'Cardiomegaly', 'Consolidation', 'Edema', 'Pneumonia', 'Edema_soft',
            # Img
            'cxr_flag', 'raddino_emb_path', 'hybrid_emb_path',
            # Text
            'text_flag', 'text_embed_path'
        ]

        all_feature_cols = [col for col in self.merged_df.columns if col not in exclude_cols]

        self.ts_features = all_feature_cols
        print(f"[Dataset] Total features (including variable flags): {len(self.ts_features)}")

        ts_features_set = set(self.ts_features)
        value_cols = [
            c for c in self.ts_features
            if not c.endswith('_flag') and f"{c}_flag" in ts_features_set
        ]
        flag_cols = [f"{c}_flag" for c in value_cols]
        flagless_cols = [
            c for c in self.ts_features
            if not c.endswith('_flag') and f"{c}_flag" not in ts_features_set
        ]

        col_to_idx = {c: i for i, c in enumerate(self.ts_features)}
        self.value_col_idx = torch.tensor([col_to_idx[c] for c in value_cols], dtype=torch.long)
        self.flag_col_idx = torch.tensor([col_to_idx[c] for c in flag_cols], dtype=torch.long)
        print(f"[Dataset] Maskable value/flag pairs: {len(value_cols)}")
        if flagless_cols:
            print(f"[Dataset] Flag-less cols (excluded from masking, {len(flagless_cols)}): {flagless_cols}")

        # ========== image / text / clinical_prompt mapping 사전 구축 ==========
        # collate_fn에서 배치를 구성할 때, 중복 이미지/텍스트/프롬프트를 제거하고 unique한 것만 인코딩하기 위함
        self.image_map = {}
        self.segment_map = {}
        self.text_map = {}

        for stay_id in self.stay_ids:
            stay_data = self.stay_groups.get_group(stay_id)

            # cxr_flag == 1인 hour_slot만 매핑에 추가함.
            img_dict = {t: p for t, p, flag in zip(stay_data['hour_slot'], stay_data['raddino_emb_path'], stay_data['cxr_flag']) if flag == 1}
            segment_dict = {t: p for t, p, flag in zip(stay_data['hour_slot'], stay_data['hybrid_emb_path'], stay_data['cxr_flag']) if flag == 1}
            # 향후 lesion dict 추가

            # text_flag == 1인 hour_slot만 매핑에 추가함
            text_dict = {t: p for t, p, flag in zip(stay_data['hour_slot'], stay_data['text_embed_path'], stay_data['text_flag']) if flag == 1}

            self.image_map[stay_id] = img_dict
            self.segment_map[stay_id] = segment_dict
            self.text_map[stay_id] = text_dict

        self.image_emb_cache = {}
        self.segment_emb_cache = {}
        self.text_emb_cache = {}

        # ========== Window-level Indexing ==========
        self._build_window_index()
        print(f"[Dataset] Window-level indexing complete: {len(self.window_index):,} total windows")

    ################################################################################################
    def _load_emb(self, path, cache):
        if not isinstance(path, str) or path.strip() == "":
            raise RuntimeError(f"[ERROR] Invalid embedding path: {path!r}")
        if path in cache:
            return cache[path]
        if not os.path.exists(path):
            raise RuntimeError(f"[ERROR] Embedding file does not exist: {path}")
        try:
            emb = torch.load(path, map_location='cpu')
        except Exception:
            raise RuntimeError(f"[ERROR] Failed to load embedding: {path}")
        cache[path] = emb
        return emb

    def load_raddino_emb(self, path):
        payload = self._load_emb(path, self.image_emb_cache)
        return payload['global_emb']

    def load_segment_emb(self, path):
        payload = self._load_emb(path, self.segment_emb_cache)
        seg_emb = torch.stack([payload['lung'], payload['heart']], dim=0) # 폐와 심장 임베딩 분리 (shape: [2, 32])
        return seg_emb

    def load_ctr(self, path):
        # payload['geometry'] = [CTR, heart_width, thoracic_width] — CTR(Cardiothoracic Ratio)만 사용
        payload = self._load_emb(path, self.segment_emb_cache)
        return payload['geometry'][0].float()  # scalar

    def load_text_emb(self, path):
        return self._load_emb(path, self.text_emb_cache)
    ################################################################################################


    def _build_window_index(self):
        """
        전체 windows를 flat list로 구성, 각 window를 독립적인 샘플로 취급하여 window-level batching 지원
        """
        self.window_index = []
        self.window_labels = {}

        # Valid patient filtering
        valid_patients = []
        skipped_patients = 0

        for stay_id in self.stay_ids:
            stay_data = self.stay_groups.get_group(stay_id)
            L = len(stay_data)

            if L >= self.window_size + self.prediction_horizon:
                valid_patients.append(stay_id)
                max_start = L - self.window_size - self.prediction_horizon

                edema_soft_labels = stay_data['Edema_soft'].to_numpy()
                edema_hard_labels = stay_data['Edema'].to_numpy()      # Track A
                cxr_flags = stay_data['cxr_flag'].to_numpy()           # Track A anchor
                cardio_labels = stay_data['Cardiomegaly'].to_numpy()   # Sub-task
                pneumo_labels = stay_data['Pneumonia'].to_numpy()      # Sub-task

                for i in range(0, max_start + 1, self.stride):
                    label_idx = i + self.window_size + self.prediction_horizon - 1
                    edema_soft_label = edema_soft_labels[label_idx]

                    # NaN 윈도우 처리: include_nan_labels=False면 스킵, True면 contrastive 학습용으로 포함
                    if pd.isna(edema_soft_label) and not self.include_nan_labels:
                        continue

                    edema_hard_label = edema_hard_labels[label_idx]
                    
                    # cxr_anchor = int(cxr_flags[label_idx])
                    cxr_flag_val = cxr_flags[label_idx]
                    cxr_anchor = int(cxr_flag_val) if pd.notna(cxr_flag_val) else 0

                    cardio_label = cardio_labels[label_idx]
                    pneumo_label = pneumo_labels[label_idx]

                    window_id = len(self.window_index)
                    self.window_index.append((stay_id, i))

                    self.window_labels[window_id] = {
                        'edema_soft': float(edema_soft_label),
                        'edema_hard': int(edema_hard_label) if pd.notna(edema_hard_label) else -1,
                        'cxr_anchor': cxr_anchor,
                        # NaN은 float('nan') 그대로 보존 → loss에서 마스킹
                        'cardiomegaly': float(cardio_label) if pd.notna(cardio_label) else float('nan'),
                        'pneumonia': float(pneumo_label) if pd.notna(pneumo_label) else float('nan'),
                        # 'subtype': int(subtype_label) if pd.notna(subtype_label) else -1,
                    }
            else:
                skipped_patients += 1

        self.stay_ids = valid_patients # Update stay_ids to only include valid patients

        if skipped_patients > 0:
            print(f"[Dataset] Skipped {skipped_patients} patients with insufficient data (< {self.window_size + self.prediction_horizon}h)")
        print(f"[Dataset] Built window index: {len(self.window_index):,} windows from {len(self.stay_ids):,} patients")


    def __getitem__(self, idx):
        """
        단일 window 반환 (24h 고정, window-level batching)
        """
        # Window index에서 (stay_id, start_idx) 조회
        stay_id, start_idx = self.window_index[idx]
        stay_data = self.stay_groups.get_group(stay_id)

        # Window 데이터 추출 (start_idx부터 window_size만큼)
        window_data = stay_data.iloc[start_idx:start_idx + self.window_size]

        # Hour slots 및 시계열 features
        hour_slots = window_data['hour_slot'].to_numpy()
        ts_features = torch.tensor(window_data[self.ts_features].astype(np.float32).to_numpy(), dtype=torch.float32)  # [24, D]
        img_indices = [t if t in self.image_map[stay_id] else -1 for t in hour_slots]
        text_indices = [t if t in self.text_map[stay_id] else -1 for t in hour_slots]

        # Window sequence 생성
        window_sequence = [
            {
                'time_step': int(hour_slots[j]),
                'ts_features': ts_features[j],
                'img_index': img_indices[j],
                'txt_index': text_indices[j],
            }
            for j in range(self.window_size)
        ]

        labels = self.window_labels[idx]

        return {
            'stay_id': stay_id,
            'window_idx': idx,
            'time_steps': hour_slots,
            'ts_features': ts_features,
            'window_sequence': window_sequence,
            'img_indices': img_indices,
            'text_indices': text_indices,
            'has_cxr': [int(x != -1) for x in img_indices],
            'has_text': [int(x != -1) for x in text_indices],
            'edema_soft': labels['edema_soft'],
            'edema_hard': labels['edema_hard'],
            'cxr_anchor': labels['cxr_anchor'],
            'cardiomegaly': labels['cardiomegaly'],
            'pneumonia': labels['pneumonia'],
            # 'subtype_label': labels['subtype'],  # scalar
        }
    
    def collate_fn(self, batch):
        """
        Window-level batching용 collate function (고정 길이, 패딩 불필요)
        - 각 window는 24h 고정 길이
        - 배치 내 고유 이미지/텍스트만 추출하여 메모리와 연산량 절약
        """
        args = self.args
        B = len(batch)

        # ==================== 배치 내 고유 항목 추출 + 인덱스 매핑 ====================
        unique_img_paths = []
        unique_segment_paths = []
        unique_text_paths = []

        img_path_to_idx = {}
        segment_path_to_idx = {}
        text_path_to_idx = {}

        img_index_tensor = torch.full((B, args.window_size), fill_value=-1, dtype=torch.long)
        segment_index_tensor = torch.full((B, args.window_size), fill_value=-1, dtype=torch.long)
        text_index_tensor = torch.full((B, args.window_size), fill_value=-1, dtype=torch.long)

        # 고유 항목 수집 + 인덱스 매핑 동시 수행
        for i, item in enumerate(batch):
            stay_id = item['stay_id']
            for t, step in enumerate(item['window_sequence']):
                # 이미지 처리
                img_hour = step['img_index']
                if img_hour != -1:
                    img_path = self.image_map[stay_id][img_hour]
                    if img_path not in img_path_to_idx:
                        img_path_to_idx[img_path] = len(unique_img_paths)
                        unique_img_paths.append(img_path)
                    img_index_tensor[i, t] = img_path_to_idx[img_path]

                    # Lung, heart segment embed
                    segment_path = self.segment_map[stay_id][img_hour]
                    if segment_path not in segment_path_to_idx:
                        segment_path_to_idx[segment_path] = len(unique_segment_paths)
                        unique_segment_paths.append(segment_path)
                    segment_index_tensor[i, t] = segment_path_to_idx[segment_path]

                # 텍스트 처리
                txt_hour = step['txt_index']
                if txt_hour != -1:
                    txt_path = self.text_map[stay_id][txt_hour]
                    if txt_path not in text_path_to_idx:
                        text_path_to_idx[txt_path] = len(unique_text_paths)
                        unique_text_paths.append(txt_path)
                    text_index_tensor[i, t] = text_path_to_idx[txt_path]

        unique_img_embs = (
            torch.stack([self.load_raddino_emb(p) for p in unique_img_paths])
            if unique_img_paths else torch.empty(0)
        )
        unique_segment_embs = (
            torch.stack([self.load_segment_emb(p) for p in unique_segment_paths])
            if unique_segment_paths else torch.empty(0)
        )
        unique_ctr = (
            torch.stack([self.load_ctr(p) for p in unique_segment_paths])
            if unique_segment_paths else torch.empty(0)
        )  # [N_seg], CTR scalar
        unique_text_embs = (
            torch.stack([self.load_text_emb(p) for p in unique_text_paths])
            if unique_text_paths else torch.empty(0)
        )

        # ==================== 라벨 텐서 ====================
        edema_soft_labels = torch.tensor([item['edema_soft'] for item in batch], dtype=torch.float32)
        edema_hard_labels = torch.tensor([item['edema_hard'] for item in batch], dtype=torch.long)
        cxr_anchor_mask = torch.tensor([item['cxr_anchor'] for item in batch], dtype=torch.long)
        cardiomegaly_labels = torch.tensor([item['cardiomegaly'] for item in batch], dtype=torch.float32)
        pneumonia_labels = torch.tensor([item['pneumonia'] for item in batch], dtype=torch.float32)
        # subtype_labels = torch.tensor([item['subtype_label'] for item in batch], dtype=torch.long)

        # Window-level batching: 각 배치 아이템이 하나의 window
        ts_tensor = torch.stack([item['ts_features'] for item in batch])  # [B, 24, D]
        time_steps_tensor = torch.stack([torch.tensor(item['time_steps'], dtype=torch.float32) for item in batch])  # [B, 24]

        return {
            'stay_ids': [item['stay_id'] for item in batch],
            'ts_tensor': ts_tensor,
            'time_steps': time_steps_tensor,
            'img_index_tensor': img_index_tensor,
            'segment_index_tensor': segment_index_tensor,
            'text_index_tensor': text_index_tensor,
            'unique_img_embs': unique_img_embs,
            'unique_segment_embs': unique_segment_embs,
            'unique_ctr': unique_ctr,
            'unique_text_embs': unique_text_embs,
            'edema_soft_labels': edema_soft_labels,
            'edema_hard_labels': edema_hard_labels,
            'cxr_anchor_mask': cxr_anchor_mask,
            'cardiomegaly_labels': cardiomegaly_labels,
            'pneumonia_labels': pneumonia_labels,
            'value_col_idx': self.value_col_idx,
            'flag_col_idx': self.flag_col_idx,
            # 'subtype_labels': subtype_labels,
        }
    
    def __len__(self):
        """전체 window 개수 반환"""
        return len(self.window_index)


#######################################################################
# Window-level Sampler
#######################################################################
class DDPWindowSampler(Sampler):
    """
    Window-level stratified sampling
    - 각 window를 독립적인 샘플로 취급
    - Oversampling이 없는 Edema 라벨 기준 층화추출
    - DDP 지원
    """
    def __init__(self, dataset, batch_size, accelerator=None, shuffle=True, drop_last=True, seed=42, split=None):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = seed
        self.split = split

        # Window indices - 이미 dataset에서 _build_window_index()로 생성됨
        self.window_indices = list(range(len(dataset)))

        # DDP 설정
        if accelerator:
            self.world_size = accelerator.num_processes
            self.rank = accelerator.process_index
        else:
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
            self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Epoch마다 windows 섞기"""
        random.seed(self.base_seed + epoch)
        indices = self.window_indices.copy()

        if self.shuffle:
            random.shuffle(indices)

        # Batch 구성
        self.batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                self.batches.append(batch)

        # DDP alignment
        remainder = len(self.batches) % self.world_size
        if self.drop_last and remainder != 0:
            self.batches = self.batches[:len(self.batches) - remainder]
        elif not self.drop_last and remainder != 0:
            # Pad with random batches
            pad_need = self.world_size - remainder
            for _ in range(pad_need):
                self.batches.append(random.choice(self.batches))

        if self.rank == 0:
            split_tag = f"[{self.split.upper()}]" if self.split else ""
            print(f"\n[DDPWindowSampler]{split_tag} Epoch {epoch}")
            print(f"Total windows: {len(indices):,}")
            print(f"Batch size: {self.batch_size}")
            print(f"Total batches: {len(self.batches)}")
            print(f"Batches per GPU: {len(self.batches) // self.world_size}")

    def __iter__(self):
        """각 GPU에 배치를 균등하게 분배"""
        my_batches = [self.batches[i] for i in range(self.rank, len(self.batches), self.world_size)]
        for batch in my_batches:
            yield batch

    def __len__(self):
        """각 GPU가 처리할 배치 수"""
        return len(self.batches) // self.world_size


#######################################################################
# 데이터셋 정의
#######################################################################
def get_dataloaders(train_df, val_df, test_df, args, accelerator=None, num_workers=4):
    """
    Window-level batching을 위한 DataLoader 생성
    - train_df, val_df, test_df: time_series_cxr_preprocess.ipynb에서 이미 split & scaling & join 완료된 데이터
    """
    # Dataset 생성 (이미 split & scaling 완료된 데이터 사용)
    with timer("Dataset 생성"):
        train_stride = args.train_stride
        eval_stride = args.eval_stride
        
        train_dataset = SCL_Multi_Dataset(args, train_df, include_nan_labels=(args.unsupervised_weight > 0), stride=train_stride) # 일단 이렇게 결측 라벨을 제어함.
        val_dataset = SCL_Multi_Dataset(args, val_df, include_nan_labels=False, stride=eval_stride)
        test_dataset = SCL_Multi_Dataset(args, test_df, include_nan_labels=False, stride=eval_stride)

    # Window-level sampler 생성
    with timer("샘플러 가동"):
        train_sampler = DDPWindowSampler(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            accelerator=accelerator,
            shuffle=True,
            drop_last=True,
            seed=args.random_seed,
            split="Train"
        )

        val_sampler = DDPWindowSampler(
            dataset=val_dataset,
            batch_size=args.val_batch_size,
            accelerator=accelerator,
            shuffle=False,
            drop_last=False,
            seed=args.random_seed,
            split="Validation"
        )

        test_sampler = DDPWindowSampler(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            accelerator=accelerator,
            shuffle=False,
            drop_last=False,
            seed=args.random_seed,
            split="Test"
        )

    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn
    test_collate_fn = test_dataset.collate_fn

    with timer("데이터로더 정의"):
        dataloader_kwargs = {
            'pin_memory': True,
            'num_workers': num_workers,
        }

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

    return train_dataloader, val_dataloader, test_dataloader, train_sampler


# #######################################################################
# # 데이터셋 정의를 위한 기타 함수
# #######################################################################
# # merging dataframes
# def merged_dataframes(ts_df, img_df, text_df, clinical_prompt_df):
#     merged_df = (ts_df
#             .merge(img_df, on=['stay_id', 'hour_slot'], how='outer')
#             .merge(text_df, on=['stay_id','hour_slot'], how='outer')
#             .merge(clinical_prompt_df[['hadm_id', 'stay_id', 'hour_slot', 'clinical_prompt', 'prompt_id']], on=['hadm_id', 'stay_id', 'hour_slot'], how='left')
#     )

#     ########################################################################
#     # Analyze modality combinations at hour_slot level
#     print(f"\n{'='*80}")
#     print(f"📊 HOUR_SLOT LEVEL MODALITY COMBINATION ANALYSIS")
#     print(f"{'='*80}")

#     # Define modality flags
#     has_ts = merged_df['hour_slot'].notna()  # TS always exists (from ts_df)
#     has_img = merged_df['cxr_flag'].notna() & (merged_df['cxr_flag'] == 1)
#     has_text = merged_df['text_flag'].notna() & (merged_df['text_flag'] == 1)

#     # Count combinations
#     ts_only = (has_ts & ~has_img & ~has_text).sum()
#     ts_img = (has_ts & has_img & ~has_text).sum()
#     ts_text = (has_ts & ~has_img & has_text).sum()
#     ts_img_text = (has_ts & has_img & has_text).sum()

#     total_hourslots = len(merged_df)

#     print(f"\n[Merged DataFrame] Total hour_slots: {total_hourslots:,}")
#     print(f"{'─'*80}")
#     print(f"Modality Combinations:")
#     print(f"  TS only:              {ts_only:>8,} ({ts_only/total_hourslots*100:>5.2f}%)")
#     print(f"  TS + Image:           {ts_img:>8,} ({ts_img/total_hourslots*100:>5.2f}%)")
#     print(f"  TS + Text:            {ts_text:>8,} ({ts_text/total_hourslots*100:>5.2f}%)")
#     print(f"  TS + Image + Text:    {ts_img_text:>8,} ({ts_img_text/total_hourslots*100:>5.2f}%)")
#     print(f"{'─'*80}")
#     print(f"Multimodal Coverage:")
#     print(f"  At least Image:       {(has_img).sum():>8,} ({(has_img).sum()/total_hourslots*100:>5.2f}%)")
#     print(f"  At least Text:        {(has_text).sum():>8,} ({(has_text).sum()/total_hourslots*100:>5.2f}%)")
#     print(f"  At least one extra:   {(has_img | has_text).sum():>8,} ({(has_img | has_text).sum()/total_hourslots*100:>5.2f}%)")
#     print(f"{'='*80}\n")
#     ########################################################################
#     return merged_df


# def calculate_window_label_distribution(dataset, dataset_name="Dataset"):
#     """
#     - 데이터셋의 window-level 라벨 분포를 계산하고 출력
#     - Multi-task: Edema와 Subtype 분포를 모두 출력
#     """
#     # Check if multi-task (edema_label_series exists)
#     if len(dataset.label_metadata) > 0 and 'edema_label_series' in dataset.label_metadata[0]:
#         # Multi-task distribution
#         edema_0_windows = 0
#         edema_1_windows = 0
#         edema_unlabeled_windows = 0

#         subtype_1_windows = 0  # non-cardiogenic
#         subtype_2_windows = 0  # cardiogenic
#         subtype_unlabeled_windows = 0

#         for meta in dataset.label_metadata:
#             edema_labels = meta['edema_label_series']
#             subtype_labels = meta['subtype_label_series']

#             for edema_label, subtype_label in zip(edema_labels, subtype_labels):
#                 # Count edema distribution
#                 if edema_label == 0:
#                     edema_0_windows += 1
#                 elif edema_label == 1:
#                     edema_1_windows += 1
#                 else:
#                     edema_unlabeled_windows += 1

#                 # Count subtype distribution (only for edema=1)
#                 # Subtype labels: 1 (non-cardiogenic), 2 (cardiogenic)
#                 if edema_label == 1:
#                     if subtype_label == 1:
#                         subtype_1_windows += 1
#                     elif subtype_label == 2:
#                         subtype_2_windows += 1
#                     else:
#                         # -1 or NaN (unlabeled)
#                         subtype_unlabeled_windows += 1

#         total_windows = edema_0_windows + edema_1_windows + edema_unlabeled_windows

#         print(f"\n{'='*60}")
#         print(f"[{dataset_name}] Window-level Label Distribution (Multi-task)")
#         print(f"{'='*60}")
#         print(f"Edema Distribution:")
#         print(f"  No edema (0):        {edema_0_windows:>6} ({edema_0_windows/total_windows*100:>5.2f}%)")
#         print(f"  Has edema (1):       {edema_1_windows:>6} ({edema_1_windows/total_windows*100:>5.2f}%)")
#         print(f"  Unlabeled:           {edema_unlabeled_windows:>6} ({edema_unlabeled_windows/total_windows*100:>5.2f}%)")
#         print(f"{'─'*60}")
#         print(f"Subtype Distribution (among edema=1):")
#         edema_1_total = subtype_1_windows + subtype_2_windows + subtype_unlabeled_windows
#         if edema_1_total > 0:
#             print(f"  Non-cardiogenic (1): {subtype_1_windows:>6} ({subtype_1_windows/edema_1_total*100:>5.2f}%)")
#             print(f"  Cardiogenic (2):     {subtype_2_windows:>6} ({subtype_2_windows/edema_1_total*100:>5.2f}%)")
#             print(f"  Unlabeled:           {subtype_unlabeled_windows:>6} ({subtype_unlabeled_windows/edema_1_total*100:>5.2f}%)")
#         print(f"{'─'*60}")
#         print(f"Total windows:         {total_windows:>6}")
#         print(f"{'='*60}\n")

#         return {
#             'edema_0': edema_0_windows,
#             'edema_1': edema_1_windows,
#             'edema_unlabeled': edema_unlabeled_windows,
#             'subtype_1': subtype_1_windows,
#             'subtype_2': subtype_2_windows,
#             'subtype_unlabeled': subtype_unlabeled_windows,
#             'total': total_windows
#         }

#     else:
#         # Legacy distribution (3-class)
#         cardio_windows = 0
#         noncardio_windows = 0
#         negative_windows = 0
#         unlabeled_windows = 0

#         for meta in dataset.label_metadata:
#             for label in meta['label_series']:
#                 if label == 2:
#                     cardio_windows += 1
#                 elif label == 1:
#                     noncardio_windows += 1
#                 elif label == 0:
#                     negative_windows += 1
#                 else:  # -1 or NaN
#                     unlabeled_windows += 1

#         total_windows = cardio_windows + noncardio_windows + negative_windows + unlabeled_windows

#         print(f"\n{'='*60}")
#         print(f"[{dataset_name}] Window-level Label Distribution")
#         print(f"{'='*60}")
#         print(f"Cardio windows:        {cardio_windows:>6} ({cardio_windows/total_windows*100:>5.2f}%)")
#         print(f"Non-cardio windows:    {noncardio_windows:>6} ({noncardio_windows/total_windows*100:>5.2f}%)")
#         print(f"Negative windows:      {negative_windows:>6} ({negative_windows/total_windows*100:>5.2f}%)")
#         print(f"Unlabeled windows:     {unlabeled_windows:>6} ({unlabeled_windows/total_windows*100:>5.2f}%)")
#         print(f"{'─'*60}")
#         print(f"Total windows:         {total_windows:>6}")
#         print(f"{'='*60}\n")

#         return {
#             'cardio': cardio_windows,
#             'noncardio': noncardio_windows,
#             'negative': negative_windows,
#             'unlabeled': unlabeled_windows,
#             'total': total_windows
#         }


# def calculate_modality_distribution(dataset, dataset_name="Dataset"):
#     """
#     - 데이터셋의 window-level 모달리티 조합 분포를 계산하고 출력
#     - 각 window가 어떤 모달리티 조합을 가지고 있는지 분석

#     카테고리:
#     1. ts_only: 시계열만 (이미지 X, 텍스트 X)
#     2. ts_img: 시계열 + 이미지 (텍스트 X)
#     3. ts_text: 시계열 + 텍스트 (이미지 X)
#     4. ts_img_text: 시계열 + 이미지 + 텍스트
#     """
#     ts_only = 0
#     ts_img = 0
#     ts_text = 0
#     ts_img_text = 0

#     # 모든 환자 데이터를 순회하며 window별 모달리티 조합 분석
#     for stay_id in dataset.stay_ids:
#         stay_data = dataset.stay_groups.get_group(stay_id).sort_values('hour_slot')

#         hour_slots = stay_data['hour_slot'].to_numpy()
#         L = len(stay_data)

#         # 각 hour_slot에 이미지와 텍스트가 있는지 확인
#         img_index_series = [t if t in dataset.image_map[stay_id] else -1 for t in hour_slots]
#         text_index_series = [t if t in dataset.text_map[stay_id] else -1 for t in hour_slots]

#         # Sliding window 생성 (dataset의 __getitem__과 동일한 로직)
#         if L >= dataset.window_size + dataset.prediction_horizon:
#             max_start_idx = L - dataset.window_size - dataset.prediction_horizon

#             for i in range(0, max_start_idx + 1, dataset.stride):
#                 window_img = img_index_series[i:i + dataset.window_size]
#                 window_text = text_index_series[i:i + dataset.window_size]

#                 # window 내에 이미지/텍스트가 하나라도 있는지 확인
#                 has_img = any(x != -1 for x in window_img)
#                 has_text = any(x != -1 for x in window_text)

#                 # 모달리티 조합에 따라 카운트
#                 if has_img and has_text:
#                     ts_img_text += 1
#                 elif has_img:
#                     ts_img += 1
#                 elif has_text:
#                     ts_text += 1
#                 else:
#                     ts_only += 1

#     total_windows = ts_only + ts_img + ts_text + ts_img_text

#     print(f"\n{'='*60}")
#     print(f"[{dataset_name}] Window-level Modality Distribution")
#     print(f"{'='*60}")
#     print(f"TS only:               {ts_only:>6} ({ts_only/total_windows*100:>5.2f}%)")
#     print(f"TS + Image:            {ts_img:>6} ({ts_img/total_windows*100:>5.2f}%)")
#     print(f"TS + Text:             {ts_text:>6} ({ts_text/total_windows*100:>5.2f}%)")
#     print(f"TS + Image + Text:     {ts_img_text:>6} ({ts_img_text/total_windows*100:>5.2f}%)")
#     print(f"{'─'*60}")
#     print(f"Total windows:         {total_windows:>6}")
#     print(f"{'='*60}\n")

#     return {
#         'ts_only': ts_only,
#         'ts_img': ts_img,
#         'ts_text': ts_text,
#         'ts_img_text': ts_img_text,
#         'total': total_windows
#     }


# def analyze_batch_label_distribution(dataloader, dataset_name="Dataset", num_batches=None):
#     """
#     - DataLoader의 각 배치별 라벨 분포를 분석하고 통계를 출력
#     - num_batches: 분석할 배치 수 (None이면 전체)
#     """
#     batch_stats = []

#     print(f"\n{'='*80}")
#     print(f"[{dataset_name}] Batch-level Label Distribution Analysis")
#     print(f"{'='*80}")

#     for batch_idx, batch in enumerate(dataloader):
#         if num_batches and batch_idx >= num_batches:
#             break

#         labels = batch['labels']  # [B, W]

#         valid_labels = (labels != -1)

#         if len(valid_labels) == 0:
#             continue

#         # 각 라벨별 개수 계산
#         cardio_count = (valid_labels == 2).sum().item()
#         noncardio_count = (valid_labels == 1).sum().item()
#         negative_count = (valid_labels == 0).sum().item()
#         total_valid = len(valid_labels)

#         batch_stats.append({
#             'batch_idx': batch_idx,
#             'cardio': cardio_count,
#             'noncardio': noncardio_count,
#             'negative': negative_count,
#             'total': total_valid,
#             'cardio_pct': cardio_count / total_valid * 100 if total_valid > 0 else 0,
#             'noncardio_pct': noncardio_count / total_valid * 100 if total_valid > 0 else 0,
#             'negative_pct': negative_count / total_valid * 100 if total_valid > 0 else 0
#         })

#     if not batch_stats:
#         print(f"[Warning] No valid batches found in {dataset_name}")
#         return

#     # 전체 배치 통계
#     total_cardio = sum(stat['cardio'] for stat in batch_stats)
#     total_noncardio = sum(stat['noncardio'] for stat in batch_stats)
#     total_negative = sum(stat['negative'] for stat in batch_stats)
#     total_windows = sum(stat['total'] for stat in batch_stats)

#     # 배치별 비율의 평균 및 표준편차
#     cardio_pcts = [stat['cardio_pct'] for stat in batch_stats]
#     noncardio_pcts = [stat['noncardio_pct'] for stat in batch_stats]
#     negative_pcts = [stat['negative_pct'] for stat in batch_stats]

#     cardio_mean = np.mean(cardio_pcts)
#     cardio_std = np.std(cardio_pcts)
#     noncardio_mean = np.mean(noncardio_pcts)
#     noncardio_std = np.std(noncardio_pcts)
#     negative_mean = np.mean(negative_pcts)
#     negative_std = np.std(negative_pcts)

#     print(f"\n전체 배치 통계 (분석 배치 수: {len(batch_stats)})")
#     print(f"{'─'*80}")
#     print(f"Total windows across all batches: {total_windows:,}")
#     print(f"  Cardio:     {total_cardio:>6,} ({total_cardio/total_windows*100:>5.2f}%)")
#     print(f"  Non-cardio: {total_noncardio:>6,} ({total_noncardio/total_windows*100:>5.2f}%)")
#     print(f"  Negative:   {total_negative:>6,} ({total_negative/total_windows*100:>5.2f}%)")

#     print(f"\n배치별 라벨 비율 (평균 ± 표준편차)")
#     print(f"{'─'*80}")
#     print(f"  Cardio:     {cardio_mean:>5.2f}% ± {cardio_std:>4.2f}%")
#     print(f"  Non-cardio: {noncardio_mean:>5.2f}% ± {noncardio_std:>4.2f}%")
#     print(f"  Negative:   {negative_mean:>5.2f}% ± {negative_std:>4.2f}%")

#     # 배치 크기 통계
#     batch_sizes = [stat['total'] for stat in batch_stats]
#     print(f"\n배치당 유효 window 수")
#     print(f"{'─'*80}")
#     print(f"  평균: {np.mean(batch_sizes):.1f}")
#     print(f"  최소: {np.min(batch_sizes)}")
#     print(f"  최대: {np.max(batch_sizes)}")
#     print(f"  표준편차: {np.std(batch_sizes):.1f}")

#     print(f"{'='*80}\n")

#     return batch_stats


# def split_dataset(merged_df, train_ratio, val_ratio, random_seed=0):
#     """
#     환자 레벨에서 층화추출을 수행하여 train/val/test split
#     Multi-task learning: Edema 라벨을 기준으로 층화추출
#     (Main task: Edema detection, Sub task: Subtype classification)
#     """
#     stay_ids = merged_df['stay_id'].unique()

#     print("[Dataset Split] Using Edema-based stratification for multi-task learning")
#     # 각 환자의 대표 Edema 라벨 결정 (우선순위: edema=1 > edema=0 > unlabeled)
#     stay_labels = []
#     for stay_id in stay_ids:
#         stay_data = merged_df[merged_df['stay_id'] == stay_id]
#         edema_labels = stay_data['Edema'].to_numpy()

#         if np.any(edema_labels == 1):  # Has edema
#             stay_labels.append(1)
#         elif np.any(edema_labels == 0):  # No edema
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

#         edema_labels = stay_data['Edema'].to_numpy()
#         if np.any(edema_labels == 1):
#             temp_stay_labels.append(1)
#         elif np.any(edema_labels == 0):
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

#     for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
#         edema_0 = (split_df['Edema'] == 0).sum()
#         edema_1 = (split_df['Edema'] == 1).sum()
#         edema_unlabeled = (split_df['Edema'] == -1).sum()

#         # Among edema=1, count subtype distribution
#         edema_1_df = split_df[split_df['Edema'] == 1]
#         subtype_0 = (edema_1_df['subtype_label'] == 0).sum() if 'subtype_label' in split_df.columns else 0
#         subtype_1 = (edema_1_df['subtype_label'] == 1).sum() if 'subtype_label' in split_df.columns else 0
#         subtype_2 = (edema_1_df['subtype_label'] == 2).sum() if 'subtype_label' in split_df.columns else 0

#         print(f"\n{'='*80}")
#         print(f"{split_name} Set:")
#         print(f"  Edema Negative={edema_0}, Edema Positive={edema_1}, Unlabeled+Uncertain={edema_unlabeled}")
#         print(f"  Subtype (P(subtype|edema=1)): Intermediate={subtype_0}, Non-cardio={subtype_1}, Cardio={subtype_2}")
#         print(f"{'='*80}")

#     return train_df, val_df, test_df


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


# class StratifiedPatientSampler(Sampler):
#     """
#     층화추출 기반 Patient-level 배치 샘플러 (오버샘플링 없음)
#     - 전체 환자를 윈도우 라벨 비율에 맞게 섞어서 배치 구성
#     - B=batch_size (한 배치에 여러 환자의 윈도우)
#     - 윈도우 라벨 비율을 자연스럽게 유지
#     """
#     def __init__(self, dataset, batch_size=32, accelerator=None, shuffle=True, drop_last=True, seed=42, split=None):
#         super().__init__()
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.accelerator = accelerator
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#         self.base_seed = seed
#         self.split = split

#         # 전체 환자 인덱스
#         self.patient_indices = list(range(len(dataset)))

#         # DDP 설정 (현재 미사용)
#         if self.accelerator is not None:
#             self.world_size = self.accelerator.num_processes
#             self.rank = self.accelerator.process_index
#         else:
#             self.world_size = dist.get_world_size() if dist.is_initialized() else 1
#             self.rank = dist.get_rank() if dist.is_initialized() else 0

#         self.set_epoch(0)

#     def set_epoch(self, epoch):
#         """Epoch마다 환자 순서를 섞고 배치 생성"""
#         random.seed(self.base_seed + epoch)
#         patients = self.patient_indices.copy()

#         if self.shuffle:
#             random.shuffle(patients)

#         # batch_size명의 환자를 하나의 미니 배치로 생성함.
#         self.batches = []
#         for i in range(0, len(patients), self.batch_size):
#             batch = patients[i:i + self.batch_size]
#             if len(batch) == self.batch_size or not self.drop_last:
#                 self.batches.append(batch)

#         # DDP alignment
#         remainder = len(self.batches) % self.world_size

#         if self.drop_last and remainder != 0:
#             self.batches = self.batches[:len(self.batches) - remainder]
#             if self.accelerator is None or self.accelerator.is_main_process:
#                 print(f"[StratifiedPatientSampler] drop_last=True: {remainder} batches removed for DDP alignment")
#         elif not self.drop_last and remainder != 0:
#             pad_need = self.world_size - remainder
#             for _ in range(pad_need):
#                 self.batches.append(random.choice(self.batches))
#             if self.accelerator is None or self.accelerator.is_main_process:
#                 print(f"[StratifiedPatientSampler] drop_last=False: {pad_need} batches padded for DDP alignment")

#         if self.accelerator is None or self.accelerator.is_main_process:
#             split_tag = f"[{self.split.upper()}]" if self.split else ""
#             print(f"\n[StratifiedPatientSampler]{split_tag} Epoch {epoch} initialized")
#             print(f"Batch size (patients per batch): {self.batch_size}")
#             print(f"Total batches: {len(self.batches)}")
#             print(f"Batches per GPU: {len(self.batches) // self.world_size}")
#             print(f"Total patients: {len(patients)}")

#     def __iter__(self):
#         """각 GPU에 배치를 균등하게 분배"""
#         my_batches = [self.batches[i] for i in range(self.rank, len(self.batches), self.world_size)]

#         if self.accelerator is None or self.accelerator.is_main_process:
#             print(f"[StratifiedPatientSampler][Rank {self.rank}] Yielding {len(my_batches)} batches")

#         for batch in my_batches:
#             yield batch

#     def __len__(self):
#         """각 GPU가 처리할 배치 수"""
#         return len(self.batches) // self.world_size

#     def get_actual_class_distribution(self):
#         """
#         - 실제 배치 구성에서의 윈도우 레벨 클래스 분포 반환
#         - 오버샘플링이 없으므로 데이터셋 원본 분포와 동일
#         - Unlabeled 윈도우(label=-1)는 Loss에서 ignore되므로 제외하고 계산
#         """
#         # Check if multi-task
#         is_multitask = len(self.dataset.label_metadata) > 0 and 'edema_label_series' in self.dataset.label_metadata[0]

#         if is_multitask:
#             # Multi-task: count edema distribution
#             edema_0_windows = 0
#             edema_1_windows = 0
#             unlabeled_windows = 0

#             for batch in self.batches:
#                 for patient_idx in batch:
#                     meta = self.dataset.label_metadata[patient_idx]
#                     for label in meta['edema_label_series']:
#                         if label == 0:
#                             edema_0_windows += 1
#                         elif label == 1:
#                             edema_1_windows += 1
#                         else:
#                             unlabeled_windows += 1

#             valid_total = edema_0_windows + edema_1_windows

#             if valid_total == 0:
#                 return {'edema_0': 0.5, 'edema_1': 0.5, 'edema_0_count': 0, 'edema_1_count': 0}

#             distribution = {
#                 'edema_0': edema_0_windows / valid_total,
#                 'edema_1': edema_1_windows / valid_total,
#                 'edema_0_count': edema_0_windows,
#                 'edema_1_count': edema_1_windows,
#             }

#             if self.accelerator is None or self.accelerator.is_main_process:
#                 print(f"\n[StratifiedPatientSampler] Window-level edema distribution (유효 라벨만):")
#                 print(f"  No edema (0): {edema_0_windows:,} windows ({distribution['edema_0']:.2%})")
#                 print(f"  Has edema (1): {edema_1_windows:,} windows ({distribution['edema_1']:.2%})")

#         else:
#             # Legacy: count 3-class distribution
#             cardio_windows = 0
#             noncardio_windows = 0
#             negative_windows = 0
#             unlabeled_windows = 0

#             for batch in self.batches:
#                 for patient_idx in batch:
#                     meta = self.dataset.label_metadata[patient_idx]
#                     for label in meta['label_series']:
#                         if label == 2:
#                             cardio_windows += 1
#                         elif label == 1:
#                             noncardio_windows += 1
#                         elif label == 0:
#                             negative_windows += 1
#                         else:
#                             unlabeled_windows += 1

#             valid_total = cardio_windows + noncardio_windows + negative_windows

#             if valid_total == 0:
#                 return {'cardio': 0.33, 'noncardio': 0.33, 'negative': 0.34, 'cardio_count': 0, 'noncardio_count': 0, 'negative_count': 0}

#             distribution = {
#                 'cardio': cardio_windows / valid_total,
#                 'noncardio': noncardio_windows / valid_total,
#                 'negative': negative_windows / valid_total,
#                 'negative_count': negative_windows,
#                 'noncardio_count': noncardio_windows,
#                 'cardio_count': cardio_windows
#             }

#             if self.accelerator is None or self.accelerator.is_main_process:
#                 print(f"\n[StratifiedPatientSampler] Window-level class distribution (유효 라벨만):")
#                 print(f"  Cardio: {cardio_windows:,} windows ({distribution['cardio']:.2%})")
#                 print(f"  Non-cardio: {noncardio_windows:,} windows ({distribution['noncardio']:.2%})")
#                 print(f"  Negative: {negative_windows:,} windows ({distribution['negative']:.2%})")
#             print(f"  Unlabeled (CE에서 제외됨): {unlabeled_windows:,} windows")
#             print(f"  Total valid: {valid_total:,} windows")

#         return distribution