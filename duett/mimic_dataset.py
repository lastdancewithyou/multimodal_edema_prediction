# /home/DAHS1/gangmin/my_research/clinical_multimodal_learning/duett/mimic_dataset.py
"""MIMIC ICU dataset for DuETT SSL pretraining and fine-tuning.

새 파이프라인:
  - 노트북은 raw icu_events_df + 전체 static_df + 상수만 저장
  - train 스크립트가 --n_timesteps에 맞춰 filter/dense grid/split/stats를 모두 처리
"""
import os, pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessed')
# ART_DIR = '/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/subject_data/mimic_3_1/' # subject_data
ART_DIR = '/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/mimic_3_1/' # full data


def load_meta(art_dir=ART_DIR):
    with open(os.path.join(art_dir, 'meta.pkl'), 'rb') as f:
        return pickle.load(f)


def load_raw(art_dir=ART_DIR):
    icu = pd.read_feather(os.path.join(art_dir, 'icu_events_raw.ftr'))
    sta = pd.read_feather(os.path.join(art_dir, 'static_full.ftr'))
    return icu, sta


# ==========================================================
# 텐서 빌더 (학습 중 환자별 호출)
# ==========================================================
def build_stay_tensor(df_stay, means, stds, n_timesteps, all_vars, all_counts):
    n_vars = len(all_vars)
    x_ts = torch.zeros((n_timesteps, n_vars * 2))
    for _, row in df_stay.iterrows():
        t = int(row['slot_idx'])
        if t >= n_timesteps:
            continue
        for j, (var, cnt_col) in enumerate(zip(all_vars, all_counts)):
            count = row[cnt_col]
            if count > 0:
                val_norm = (row[var] - means[var]) / (stds[var] + 1e-7)
                x_ts[t, j]          = val_norm
                x_ts[t, j + n_vars] = count
    return x_ts


def encode_static(row, age_mean, age_std, onehot_static):
    age = (float(row['age_at_intime']) - age_mean) / (age_std + 1e-7)
    age = float(np.nan_to_num(age, nan=0.0))
    onehot = row[onehot_static].astype(float).values
    return torch.tensor([age, *onehot], dtype=torch.float32)


# ==========================================================
# Dataset
# ==========================================================
class MIMICDataset(torch.utils.data.Dataset):
    def __init__(self, stay_ids, icu_df, static_df, meta):
        self.stay_ids    = list(stay_ids)
        self.icu_df      = icu_df.set_index('stay_id').sort_index()
        self.static_df   = static_df.drop_duplicates('stay_id').set_index('stay_id')
        self.meta        = meta
        self.n_timesteps = meta['N_TIMESTEPS']
        self.label_col   = meta['LABEL_COL']
        self.bin_ends = torch.arange(1, self.n_timesteps + 1).float() / 24.0

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, i):
        sid = self.stay_ids[i]
        df_stay  = self.icu_df.loc[[sid]].reset_index()
        x_ts     = build_stay_tensor(
            df_stay, self.meta['means'], self.meta['stds'],
            self.n_timesteps, self.meta['ALL_VARS'], self.meta['ALL_COUNTS'])
        x_static = encode_static(
            self.static_df.loc[sid],
            self.meta['age_mean'], self.meta['age_std'],
            self.meta['ONEHOT_STATIC'])
        y = float(self.static_df.loc[sid, self.label_col])
        return (x_ts, x_static, self.bin_ends), y

    def d_static_num(self):      return self.meta['D_STATIC']
    def d_time_series_num(self): return len(self.meta['ALL_VARS'])
    def d_target(self):          return 1
    def pos_frac(self):
        s = self.static_df.loc[self.stay_ids, self.label_col].astype(float)
        return float(s.mean())


def collate_into_seqs(batch):
    xs, ys = zip(*batch)
    return tuple(zip(*xs)), ys


# ==========================================================
# Sliding window SSL Dataset
#   - 한 환자에서 stride 간격으로 여러 윈도우를 sample로
#   - n_timesteps 길이 윈도우, max_stay_hours까지만 사용
# ==========================================================
class MIMICSlidingDataset(torch.utils.data.Dataset):
    def __init__(self, stay_ids, icu_df, static_df, meta, stride=12):
        self.icu_df    = icu_df.set_index('stay_id').sort_index()
        self.static_df = static_df.drop_duplicates('stay_id').set_index('stay_id')
        self.meta      = meta
        self.n_timesteps = meta['N_TIMESTEPS']
        self.label_col   = meta['LABEL_COL']
        self.bin_ends = torch.arange(1, self.n_timesteps + 1).float() / 24.0

        # 환자별 dense grid 길이 (== 마지막 slot_idx + 1)
        stay_len = self.icu_df.groupby(level=0).size()

        # (stay_id, start_offset) 페어 생성: window가 stay 안에 완전히 들어가는 start
        self.samples = []
        for sid in stay_ids:
            L = int(stay_len.get(sid, 0))
            max_start = L - self.n_timesteps
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, stride):
                self.samples.append((sid, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sid, start = self.samples[i]
        df_stay = self.icu_df.loc[[sid]].reset_index()
        df_win = df_stay[(df_stay['slot_idx'] >= start) &
                         (df_stay['slot_idx'] < start + self.n_timesteps)].copy()
        df_win['slot_idx'] -= start

        x_ts = build_stay_tensor(df_win, self.meta['means'], self.meta['stds'],
                                  self.n_timesteps,
                                  self.meta['ALL_VARS'], self.meta['ALL_COUNTS'])
        x_static = encode_static(self.static_df.loc[sid],
                                  self.meta['age_mean'], self.meta['age_std'],
                                  self.meta['ONEHOT_STATIC'])
        # SSL은 라벨을 안 쓰지만 형식 통일을 위해 반환
        y = float(self.static_df.loc[sid, self.label_col])
        return (x_ts, x_static, self.bin_ends), y

    # ── DuETT 모델이 init 인자로 받는 메타 메서드들 ──
    #    SSL pretrain에서 pos_frac은 모델 attribute로 저장만 되고 loss에는 안 쓰임
    #    (training_step의 pretrain 분기는 pos_frac을 사용하지 않음)
    def d_static_num(self):      return self.meta['D_STATIC']
    def d_time_series_num(self): return len(self.meta['ALL_VARS'])
    def d_target(self):          return 1
    def pos_frac(self):
        # 환자 단위 (윈도우 단위가 아닌, 동일 환자가 여러 window를 가져도 1번만)
        unique_sids = list({sid for sid, _ in self.samples})
        s = self.static_df.loc[unique_sids, self.label_col].astype(float)
        return float(s.mean())


# ==========================================================
# Sliding SSL용 prepare: dense grid를 첫 n_timesteps에 자르지 않고 max_stay_hours까지 유지
# ==========================================================
def prepare_for_sliding_ssl(art_dir=ART_DIR, n_timesteps=48, stride=12,
                             max_stay_hours=336, split_seed=42, count_clip=15):
    """Sliding window SSL용. 환자당 여러 윈도우 sample 가능.

    - cohort: stay 길이 ≥ n_timesteps
    - 데이터: max_stay_hours까지 사용 (긴 stay는 잘라냄, 예: 14일)
    - dense grid: 환자별 실제 길이까지

    Returns:
        ds_train, ds_val, ds_test, meta
    """
    print(f'[sliding-prep] n_timesteps={n_timesteps}, stride={stride}, '
          f'max_stay={max_stay_hours}h, split_seed={split_seed}')

    meta = load_meta(art_dir)
    icu, sta = load_raw(art_dir)
    ALL_VARS    = meta['ALL_VARS']
    ALL_COUNTS  = meta['ALL_COUNTS']
    STD_VARS    = meta['STD_VARS']
    EXTRA       = meta['EXTRA']
    uniq_counts = list(dict.fromkeys(ALL_COUNTS))

    print(f'[vars]   time-series ({len(ALL_VARS)}): {ALL_VARS}')
    print(f'[static] D_STATIC={meta["D_STATIC"]} '
          f'(numeric {len(meta["NUM_STATIC"])} + onehot {len(meta["ONEHOT_STATIC"])})')

    # ─── (1) cohort 필터 + max_stay_hours cap ───
    stay_len_h = icu.groupby('stay_id')['slot_idx'].max() + 1
    keep_sids = stay_len_h[stay_len_h >= n_timesteps].index
    before = len(sta)
    sta = sta[sta.stay_id.isin(keep_sids)].reset_index(drop=True)
    icu = icu[icu.stay_id.isin(keep_sids) & (icu['slot_idx'] < max_stay_hours)].copy()
    print(f'[filter] cohort: {before} → {len(sta)} stays  '
          f'(median stay={stay_len_h.loc[keep_sids].median():.0f}h, '
          f'max used={min(stay_len_h.loc[keep_sids].max(), max_stay_hours)}h)')

    # ─── (2) 환자별 dense grid (각 stay의 실제 길이까지) ───
    actual_len = icu.groupby('stay_id')['slot_idx'].max() + 1   # cap 이후
    grids = []
    for sid in sta['stay_id'].unique():
        L = int(actual_len.get(sid, n_timesteps))
        grids.append(pd.DataFrame({'stay_id': sid, 'slot_idx': range(L)}))
    full_grid = pd.concat(grids, ignore_index=True)

    icu = full_grid.merge(icu, on=['stay_id', 'slot_idx'], how='left')
    icu[ALL_VARS]    = icu[ALL_VARS].fillna(0.0)
    icu[uniq_counts] = (icu[uniq_counts]
                       .fillna(0).astype(int).clip(lower=0, upper=count_clip))
    print(f'[grid]   icu shape = {icu.shape}')

    # ─── (3) patient-level split ───
    subj = sta[['subject_id', 'stay_id']].drop_duplicates()
    all_s = subj['subject_id'].unique()
    tr_s, tmp_s = train_test_split(all_s, test_size=0.30, random_state=split_seed)
    va_s, te_s  = train_test_split(tmp_s, test_size=0.50, random_state=split_seed)
    train_ids = subj.loc[subj.subject_id.isin(tr_s), 'stay_id'].values
    val_ids   = subj.loc[subj.subject_id.isin(va_s), 'stay_id'].values
    test_ids  = subj.loc[subj.subject_id.isin(te_s), 'stay_id'].values
    print(f'[split]  train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)}')

    # ─── (4) train split 정규화 통계 (모든 관측 slot에서) ───
    df_tr = icu[icu['stay_id'].isin(train_ids)]
    means, stds = {}, {}
    for v in ALL_VARS:
        cnt = ('count_' + v) if v in STD_VARS else EXTRA[v]
        vals = df_tr.loc[df_tr[cnt] > 0, v].dropna().astype(float)
        means[v] = float(vals.mean())
        stds[v]  = float(vals.std())
    a = sta.loc[sta['stay_id'].isin(train_ids), 'age_at_intime'].astype(float)
    age_mean, age_std = float(a.mean()), float(a.std())
    print(f'[stats]  age_mean: {age_mean:.3f} / age_std: {age_std:.3f}')

    # ─── (5) meta 완성 + 3개 Dataset ───
    meta = {**meta,
            'N_TIMESTEPS': n_timesteps,
            'means': means, 'stds': stds,
            'age_mean': age_mean, 'age_std': age_std,
            'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}

    ds_train = MIMICSlidingDataset(train_ids, icu, sta, meta, stride=stride)
    ds_val   = MIMICSlidingDataset(val_ids,   icu, sta, meta, stride=stride)
    ds_test  = MIMICSlidingDataset(test_ids,  icu, sta, meta, stride=stride)

    print(f'[windows] train={len(ds_train)} ({len(ds_train)/len(train_ids):.1f}/stay), '
          f'val={len(ds_val)} ({len(ds_val)/len(val_ids):.1f}/stay), '
          f'test={len(ds_test)} ({len(ds_test)/len(test_ids):.1f}/stay)')

    return ds_train, ds_val, ds_test, meta


# ==========================================================
# 핵심: raw → 가공 → Dataset 3개 + 완성된 meta
# ==========================================================
def prepare_from_raw(art_dir=ART_DIR, n_timesteps=24, split_seed=42,
                     count_clip=15):
    """N_TIMESTEPS에 맞춰 cohort 필터 + dense grid + split + stats까지 한 번에.

    Returns:
        ds_train, ds_val, ds_test, meta(완성된 dict)
    """
    print(f'[prepare_from_raw] n_timesteps={n_timesteps}, split_seed={split_seed}')

    meta = load_meta(art_dir)
    icu, sta = load_raw(art_dir)
    ALL_VARS    = meta['ALL_VARS']
    ALL_COUNTS  = meta['ALL_COUNTS']
    STD_VARS    = meta['STD_VARS']
    EXTRA       = meta['EXTRA']
    uniq_counts = list(dict.fromkeys(ALL_COUNTS))

    # ─── (1) cohort 필터: 실제 관측 길이 ≥ n_timesteps 인 stay만 ───
    stay_len_h = icu.groupby('stay_id')['slot_idx'].max() + 1
    keep_sids = stay_len_h[stay_len_h >= n_timesteps].index
    before = len(sta)
    sta = sta[sta.stay_id.isin(keep_sids)].reset_index(drop=True)
    icu = icu[icu.stay_id.isin(keep_sids) & (icu['slot_idx'] < n_timesteps)].copy()
    print(f'[filter] cohort: {before} → {len(sta)} stays')

    # ─── 변수 목록 출력 (sanity) ───
    print(f'[vars]   time-series ({len(ALL_VARS)}): {ALL_VARS}')
    print(f'[static] D_STATIC={meta["D_STATIC"]} '
        f'(numeric {len(meta["NUM_STATIC"])}: {meta["NUM_STATIC"]} '
        f'+ onehot {len(meta["ONEHOT_STATIC"])})')


    # ─── (2) dense grid (누락 bin을 0으로) ───
    all_stay = sta['stay_id'].unique()
    full_grid = (pd.MultiIndex
                 .from_product([all_stay, range(n_timesteps)], names=['stay_id', 'slot_idx'])
                 .to_frame(index=False))
    icu = full_grid.merge(icu, on=['stay_id', 'slot_idx'], how='left')
    icu[ALL_VARS]    = icu[ALL_VARS].fillna(0.0)
    icu[uniq_counts] = (icu[uniq_counts]
                       .fillna(0).astype(int).clip(lower=0, upper=count_clip))
    print(f'[grid]   icu shape = {icu.shape} '
          f'({len(all_stay)}×{n_timesteps}={len(all_stay)*n_timesteps})')

    # ─── (3) patient-level split ───
    subj = sta[['subject_id', 'stay_id']].drop_duplicates()
    all_s = subj['subject_id'].unique()
    tr_s, tmp_s = train_test_split(all_s, test_size=0.30, random_state=split_seed)
    va_s, te_s  = train_test_split(tmp_s, test_size=0.50, random_state=split_seed)
    train_ids = subj.loc[subj.subject_id.isin(tr_s), 'stay_id'].values
    val_ids   = subj.loc[subj.subject_id.isin(va_s), 'stay_id'].values
    test_ids  = subj.loc[subj.subject_id.isin(te_s), 'stay_id'].values
    print(f'[split]  train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)}')

    # ─── (4) train split 기준 정규화 통계 ───
    df_tr = icu[icu['stay_id'].isin(train_ids)]
    means, stds = {}, {}
    for v in ALL_VARS:
        cnt = ('count_' + v) if v in STD_VARS else EXTRA[v]
        vals = df_tr.loc[df_tr[cnt] > 0, v].dropna().astype(float)
        means[v] = float(vals.mean())
        stds[v]  = float(vals.std())
    a = sta.loc[sta['stay_id'].isin(train_ids), 'age_at_intime'].astype(float)
    age_mean, age_std = float(a.mean()), float(a.std())
    print(f'[stats]  age_mean: {age_mean:.3f} / age_std: {age_std:.3f}')

    # ─── (5) meta 완성 → Dataset 3개 생성 ───
    meta = {**meta,
            'N_TIMESTEPS': n_timesteps,
            'means': means, 'stds': stds,
            'age_mean': age_mean, 'age_std': age_std,
            'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}

    ds_train = MIMICDataset(train_ids, icu, sta, meta)
    ds_val   = MIMICDataset(val_ids,   icu, sta, meta)
    ds_test  = MIMICDataset(test_ids,  icu, sta, meta)
    return ds_train, ds_val, ds_test, meta
