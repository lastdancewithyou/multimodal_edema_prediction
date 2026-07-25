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
    def __init__(self, args, merged_df, stride=None, cxr_flag_true=False):
        self.args = args
        self.window_size = args.window_size
        self.stride = stride
        self.cxr_flag_true = cxr_flag_true

        self.merged_df = merged_df
        self.stay_groups = self.merged_df.groupby('stay_id') # stay_id 식별자를 기준으로 grouping
        self.stay_ids = list(self.stay_groups.groups.keys())

        exclude_cols = [
            'subject_id', 'hadm_id', 'stay_id', 'slot_idx',
            # New Label
            'Edema', 'Pneumonia', 'Edema_soft', 'subtype_label', 'subtype_mask', 'p_mixed', 'p_ncpe', 'p_cpe',
            # Img
            'cxr_flag', 'raddino_emb_path', 'hybrid_emb_path',
            # Text
            'text_flag', 'text_embed_path',
        ]

        self.ts_features = [col for col in self.merged_df.columns if col not in exclude_cols]
        print(f"[Dataset] Total features: {len(self.ts_features)}")

        # ── Load PatchTST SSL pretrain scaler ──
        import joblib
        scaler_path = args.ts_scaler_path
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"[TS Scaler] not found at '{scaler_path}'. "
                f"Run SSL pretrain first (Dataset_ICU auto-saves it) or set --ts_scaler_path."
            )
        scaler_info = joblib.load(scaler_path)
        self.ts_scaler = scaler_info['scaler']
        scaler_cols    = scaler_info['feature_cols']

        if scaler_cols != self.ts_features:
            if set(scaler_cols) == set(self.ts_features):
                print(f"[TS Scaler] reordering ts_features to match scaler column order")
                self.ts_features = list(scaler_cols)
            else:
                missing = set(scaler_cols) - set(self.ts_features)
                extra   = set(self.ts_features) - set(scaler_cols)
                raise ValueError(
                    f"[TS Scaler] feature set mismatch — SSL과 downstream 변수가 다릅니다.\n"
                    f"  scaler_cols ({len(scaler_cols)}): {scaler_cols}\n"
                    f"  ts_features ({len(self.ts_features)}): {self.ts_features}\n"
                    f"  missing in ts_features: {missing}\n"
                    f"  extra in ts_features:   {extra}\n"
                    f"  → SSL pretrain과 downstream의 변수 set을 일치시켜야 합니다."
                )
        print(f"[TS Scaler] loaded from {scaler_path} ({len(scaler_cols)} cols, order verified)")

        # ========== image / text / clinical_prompt mapping 사전 구축 ==========
        # collate_fn에서 배치를 구성할 때, 중복 이미지/텍스트/프롬프트를 제거하고 unique한 것만 인코딩하기 위함
        self.image_map = {}
        self.segment_map = {}
        self.text_map = {}

        for stay_id in self.stay_ids:
            stay_data = self.stay_groups.get_group(stay_id)

            # cxr_flag == 1인 slot_idx(30분 슬롯)만 매핑에 추가함.
            img_dict = {t: p for t, p, flag in zip(stay_data['slot_idx'], stay_data['raddino_emb_path'], stay_data['cxr_flag']) if flag == 1}
            segment_dict = {t: p for t, p, flag in zip(stay_data['slot_idx'], stay_data['hybrid_emb_path'], stay_data['cxr_flag']) if flag == 1}
            # 향후 lesion dict 추가

            # text_flag == 1인 slot_idx만 매핑에 추가함
            text_dict = {t: p for t, p, flag in zip(stay_data['slot_idx'], stay_data['text_embed_path'], stay_data['text_flag']) if flag == 1}

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
            emb = torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            raise RuntimeError(f"[ERROR] Failed to load embedding: {path}")
        cache[path] = emb
        return emb

    def load_raddino_emb(self, path):
        """Lung + heart ROI-pooled raddino tokens (shape [2, 768]).
        Global CLS deliberately dropped — fusion uses anatomy-focused tokens only."""
        payload = self._load_emb(path, self.image_emb_cache)
        return torch.stack([
            payload['lung_raddino'].float(),    # [768]
            payload['heart_raddino'].float(),   # [768]
        ], dim=0)                                # [2, 768]

    def load_segment_emb(self, _path):
        """Deprecated: lung/heart are now part of raddino payload. Returns empty."""
        return torch.empty(0)

    def load_text_emb(self, path):
        return self._load_emb(path, self.text_emb_cache)
    ################################################################################################

    def _build_window_index(self):
        """
        전체 windows를 flat list로 구성, 각 window를 독립적인 샘플로 취급하여 window-level batching 지원.

        Detection setup (no prediction horizon):
        - L >= window_size: 기존 sliding window. 모든 slot이 real (real_len = window_size).
        - 1 <= L < window_size: 단일 padded window. 앞 L slot real, 뒤 zero-padded. label은 마지막 valid slot(L-1)에서 가져옴.
        - L == 0: 스킵.

        Each window_index entry is (stay_id, start_idx, real_len).
        real_len < window_size 인 경우 __getitem__에서 ts_valid_mask 생성.
        """
        self.window_index = []
        self.window_labels = {}

        # Valid patient filtering
        valid_patients = []
        skipped_patients = 0
        short_stays_used = 0

        for stay_id in self.stay_ids:
            stay_data = self.stay_groups.get_group(stay_id)
            L = len(stay_data)

            if L == 0:
                skipped_patients += 1
                continue

            valid_patients.append(stay_id)

            edema_soft_labels = stay_data['Edema_soft'].to_numpy()
            edema_hard_labels = stay_data['Edema'].to_numpy()      # Track A
            cxr_flags = stay_data['cxr_flag'].to_numpy()           # Track A anchor

            # Subtype soft labels (3-class: mixed / NCPE / CPE), hard label, per-slot mask
            p_mixed_arr       = stay_data['p_mixed'].to_numpy()
            p_ncpe_arr        = stay_data['p_ncpe'].to_numpy()
            p_cpe_arr         = stay_data['p_cpe'].to_numpy()
            subtype_label_arr = stay_data['subtype_label'].to_numpy()   # hard 0/1/2 (eval-only target)
            subtype_mask_arr  = stay_data['subtype_mask'].to_numpy()

            if L >= self.window_size:
                # Long stay: sliding windows, no padding
                max_start = L - self.window_size
                window_specs = [
                    (i, i + self.window_size - 1, self.window_size)
                    for i in range(0, max_start + 1, self.stride)
                ]
            else:
                # Short stay: single padded window (zero-pad slots [L, window_size))
                window_specs = [(0, L - 1, L)]
                short_stays_used += 1

            for start_idx, label_idx, real_len in window_specs:
                edema_soft_label = edema_soft_labels[label_idx]

                if pd.isna(edema_soft_label):
                    continue

                edema_hard_label = edema_hard_labels[label_idx]

                cxr_flag_val = cxr_flags[label_idx]
                cxr_anchor = int(cxr_flag_val) if pd.notna(cxr_flag_val) else 0

                # cxr_flag_true 모드: 라벨 슬롯에 실제 CXR이 있는 윈도우만 keep
                if self.cxr_flag_true and cxr_anchor != 1:
                    continue

                # Subtype slot — keep [p_mixed, p_ncpe, p_cpe] only if subtype_mask==1.
                # Otherwise zero-fill so loss is well-defined; the mask itself excludes it from reduction.
                sub_mask_val = subtype_mask_arr[label_idx]
                if pd.notna(sub_mask_val) and int(sub_mask_val) == 1:
                    subtype_soft = [
                        float(p_mixed_arr[label_idx]),
                        float(p_ncpe_arr[label_idx]),
                        float(p_cpe_arr[label_idx]),
                    ]
                    subtype_mask_val = 1
                    hard_raw = subtype_label_arr[label_idx]
                    subtype_label_val = int(hard_raw) if pd.notna(hard_raw) else -1
                else:
                    subtype_soft = [0.0, 0.0, 0.0]
                    subtype_mask_val = 0
                    subtype_label_val = -1

                window_id = len(self.window_index)
                self.window_index.append((stay_id, start_idx, real_len))

                self.window_labels[window_id] = {
                    'edema_soft': float(edema_soft_label),
                    'edema_hard': int(edema_hard_label) if pd.notna(edema_hard_label) else -1,
                    'cxr_anchor': cxr_anchor,
                    'subtype_soft': subtype_soft,
                    'subtype_label': subtype_label_val,
                    'subtype_mask': subtype_mask_val,
                }

        self.stay_ids = valid_patients # Update stay_ids to only include valid patients

        if skipped_patients > 0:
            print(f"[Dataset] Skipped {skipped_patients} patients with 0 slots")
        if short_stays_used > 0:
            print(f"[Dataset] Included {short_stays_used} short stays (L < window_size={self.window_size}) with zero-padding + ts_valid_mask")
        print(f"[Dataset] Built window index: {len(self.window_index):,} windows from {len(self.stay_ids):,} patients")


    def __getitem__(self, idx):
        """
        단일 window 반환 (24h 고정, 30분 슬롯 → window_size=48, window-level batching).
        real_len < window_size 인 짧은 stay는 ts_features 뒤쪽을 0으로 패딩하고
        ts_valid_mask로 real/padded slot을 표시한다.
        """
        # Window index에서 (stay_id, start_idx, real_len) 조회
        stay_id, start_idx, real_len = self.window_index[idx]
        stay_data = self.stay_groups.get_group(stay_id)

        # 실제 데이터가 있는 구간만 추출
        real_window = stay_data.iloc[start_idx:start_idx + real_len]
        real_slot_indices = real_window['slot_idx'].to_numpy()
        real_ts = real_window[self.ts_features].astype(np.float32).to_numpy()  # [real_len, D]
        real_ts = self.ts_scaler.transform(real_ts).astype(np.float32)         # SSL과 동일 z-score 적용
        real_ts = np.clip(real_ts, -5.0, 5.0)                                  # SSL과 동일한 outlier 클리핑

        D = real_ts.shape[1]
        # 고정 길이로 zero-padding (real_len == window_size면 padding 0)
        ts_features = torch.zeros((self.window_size, D), dtype=torch.float32)
        ts_features[:real_len] = torch.from_numpy(real_ts)

        # slot_indices도 padded slot은 -1로 표시 (downstream에서 안전)
        slot_indices = np.full(self.window_size, -1, dtype=np.int64)
        slot_indices[:real_len] = real_slot_indices

        # ts_valid_mask: 1 = real, 0 = padded
        ts_valid_mask = torch.zeros(self.window_size, dtype=torch.float32)
        ts_valid_mask[:real_len] = 1.0

        # image/text index lookup은 real slot만
        img_indices_real = [t if t in self.image_map[stay_id] else -1 for t in real_slot_indices]
        text_indices_real = [t if t in self.text_map[stay_id] else -1 for t in real_slot_indices]
        # padded slot은 -1로 (이미지/텍스트 없음)
        img_indices = img_indices_real + [-1] * (self.window_size - real_len)
        text_indices = text_indices_real + [-1] * (self.window_size - real_len)

        # Window sequence 생성 (collate_fn에서 사용; padded slot의 img/txt index는 -1)
        window_sequence = [
            {
                'time_step': int(slot_indices[j]),
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
            'time_steps': slot_indices,
            'ts_features': ts_features,
            'ts_valid_mask': ts_valid_mask,
            'window_sequence': window_sequence,
            'img_indices': img_indices,
            'text_indices': text_indices,
            'has_cxr': [int(x != -1) for x in img_indices],
            'has_text': [int(x != -1) for x in text_indices],
            'edema_soft': labels['edema_soft'],
            'edema_hard': labels['edema_hard'],
            'cxr_anchor': labels['cxr_anchor'],
            'subtype_soft': labels['subtype_soft'],
            'subtype_label': labels['subtype_label'],
            'subtype_mask': labels['subtype_mask'],
        }
    
    def collate_fn(self, batch):
        """
        Window-level batching용 collate function (고정 길이, 패딩 불필요)
        - 각 window는 24h 고정 길이 (30분 슬롯 기준 window_size=48)
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
                # 이미지 처리 (slot_idx 기준 매핑)
                img_slot = step['img_index']
                if img_slot != -1:
                    img_path = self.image_map[stay_id][img_slot]
                    if img_path not in img_path_to_idx:
                        img_path_to_idx[img_path] = len(unique_img_paths)
                        unique_img_paths.append(img_path)
                    img_index_tensor[i, t] = img_path_to_idx[img_path]

                    # Lung, heart segment embed
                    segment_path = self.segment_map[stay_id][img_slot]
                    if segment_path not in segment_path_to_idx:
                        segment_path_to_idx[segment_path] = len(unique_segment_paths)
                        unique_segment_paths.append(segment_path)
                    segment_index_tensor[i, t] = segment_path_to_idx[segment_path]

                # 텍스트 처리 (slot_idx 기준 매핑)
                txt_slot = step['txt_index']
                if txt_slot != -1:
                    txt_path = self.text_map[stay_id][txt_slot]
                    if txt_path not in text_path_to_idx:
                        text_path_to_idx[txt_path] = len(unique_text_paths)
                        unique_text_paths.append(txt_path)
                    text_index_tensor[i, t] = text_path_to_idx[txt_path]

        unique_img_embs = (
            torch.stack([self.load_raddino_emb(p) for p in unique_img_paths])
            if unique_img_paths else torch.empty(0)
        )

        # Lung/heart now bundled into raddino payload (load_raddino_emb returns [2, 768]).
        unique_segment_embs = torch.empty(0)

        # Text는 이제 input_ids/attention_mask 쌍을 저장 (BERT 모델 내부 forward용)
        if unique_text_paths:
            loaded = [self.load_text_emb(p) for p in unique_text_paths]
            unique_text_input_ids = torch.stack([x['input_ids'].long()     for x in loaded])   # [N, 128]
            unique_text_attn_mask = torch.stack([x['attention_mask'].long() for x in loaded])  # [N, 128]
        else:
            unique_text_input_ids = torch.empty(0, 128, dtype=torch.long)
            unique_text_attn_mask = torch.empty(0, 128, dtype=torch.long)

        # ==================== 라벨 텐서 ====================
        edema_soft_labels = torch.tensor([item['edema_soft'] for item in batch], dtype=torch.float32)
        edema_hard_labels = torch.tensor([item['edema_hard'] for item in batch], dtype=torch.long)
        cxr_anchor_mask = torch.tensor([item['cxr_anchor'] for item in batch], dtype=torch.long)
        # Subtype soft targets [B, 3], hard label [B] (-1 = invalid), per-sample mask [B]
        subtype_soft_labels = torch.tensor([item['subtype_soft'] for item in batch], dtype=torch.float32)
        subtype_label = torch.tensor([item['subtype_label'] for item in batch], dtype=torch.long)
        subtype_mask = torch.tensor([item['subtype_mask'] for item in batch], dtype=torch.float32)

        # Window-level batching: 각 배치 아이템이 하나의 window
        ts_tensor = torch.stack([item['ts_features'] for item in batch])  # [B, window_size, D]
        ts_valid_mask = torch.stack([item['ts_valid_mask'] for item in batch])  # [B, window_size], 1=real
        time_steps_tensor = torch.stack([torch.tensor(item['time_steps'], dtype=torch.float32) for item in batch])  # [B, window_size]

        return {
            'stay_ids': [item['stay_id'] for item in batch],
            'ts_tensor': ts_tensor,
            'ts_valid_mask': ts_valid_mask,
            'time_steps': time_steps_tensor,
            'img_index_tensor': img_index_tensor,
            'segment_index_tensor': segment_index_tensor,
            'text_index_tensor': text_index_tensor,
            'unique_img_embs': unique_img_embs,
            'unique_segment_embs': unique_segment_embs,
            'unique_text_input_ids': unique_text_input_ids,
            'unique_text_attn_mask': unique_text_attn_mask,
            'edema_soft_labels': edema_soft_labels,
            'edema_hard_labels': edema_hard_labels,
            'cxr_anchor_mask': cxr_anchor_mask,
            'subtype_soft_labels': subtype_soft_labels,
            'subtype_label': subtype_label,
            'subtype_mask': subtype_mask,
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
            print(f"\n[DDPWindowSampler]{split_tag}")
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
        
        train_dataset = SCL_Multi_Dataset(
            args, train_df,
            stride=train_stride,
            cxr_flag_true=bool(args.train_on_cxr_only),   # 학습 윈도우만 cxr-confirmed로 제한 (옵션)
        )
        val_dataset  = SCL_Multi_Dataset(args, val_df,  stride=eval_stride, cxr_flag_true=False)
        test_dataset = SCL_Multi_Dataset(args, test_df, stride=eval_stride, cxr_flag_true=False)

    # ── Label distribution & patient-level overlap sanity check (Edema only) ──
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        def _summarize(name, ds):
            labels = ds.window_labels
            n = len(labels)
            es = np.array([labels[i]['edema_soft'] for i in range(n)], dtype=np.float32)
            eh = np.array([labels[i]['edema_hard'] for i in range(n)], dtype=np.int64)

            es_pos = int((es >= 0.5).sum())
            es_neg = int((es < 0.5).sum())
            eh_valid = eh[eh != -1]
            eh_pos = int((eh_valid == 1).sum())
            eh_neg = int((eh_valid == 0).sum())

            print(f"\n[Label/{name}] windows={n:,}, patients={len(ds.stay_ids):,}")
            print(f"  edema_soft (≥0.5):   pos={es_pos:>7,} ({100*es_pos/max(n,1):5.2f}%)  "
                  f"neg={es_neg:>7,}  mean={es.mean():.4f}")
            print(f"  edema_hard (0/1):    pos={eh_pos:>7,} ({100*eh_pos/max(len(eh_valid),1):5.2f}%)  "
                  f"neg={eh_neg:>7,}  NaN={n - len(eh_valid):,}")

        _summarize("Train", train_dataset)
        _summarize("Val",   val_dataset)
        _summarize("Test",  test_dataset)

        # Patient-level disjointness check
        tr_p = set(train_dataset.stay_ids)
        va_p = set(val_dataset.stay_ids)
        te_p = set(test_dataset.stay_ids)
        tv = tr_p & va_p; tt = tr_p & te_p; vt = va_p & te_p
        print(f"\n[Split check] train∩val = {len(tv)} stay_ids, train∩test = {len(tt)}, val∩test = {len(vt)}")
        if tv or tt or vt:
            print(f"  ⚠️  Patient-level leak detected — split may not be stay-level disjoint!")
        else:
            print(f"  ✓  stay_id-level disjoint across train/val/test")

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