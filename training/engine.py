import os
import warnings
warnings.filterwarnings('ignore', message='Spectral initialisation failed')

import torch
import torch.nn.functional as F

from utils.utils import timer


# 단일 배치 학습 함수
def train_batch(
        args, model, batch, loss_module, device, accelerator,
        disable_cxr=False, disable_txt=False,
        bce_weight=None, cardio_weight=0.0, pneumo_weight=0.0, info_weight=0.0,
        # contrast_weight=0.0, mask_prob=0.15,
        current_epoch=0, is_training=True,
    ):

    if is_training:
        model.train()

    # ==================== 1. 배치 데이터 준비 ====================
    with timer("Batch Data preparation", accelerator):
        edema_soft_labels = batch['edema_soft_labels']
        edema_hard_labels = batch['edema_hard_labels']
        cxr_anchor_mask = batch['cxr_anchor_mask']
        cardiomegaly_labels = batch['cardiomegaly_labels']
        pneumonia_labels = batch['pneumonia_labels']
        # subtype_labels = batch['subtype_labels']

        img_index_tensor = batch['img_index_tensor']
        txt_index_tensor = batch['text_index_tensor']
        has_cxr = (img_index_tensor != -1).long()   # [B, T]
        has_text = (txt_index_tensor != -1).long()  # [B, T]

    # ==================== 2. 모델 입력 데이터 전처리 ====================
    with timer("데이터 전처리 작업", accelerator):
        ts_series, cxr_data, text_data, has_cxr, has_text = prepare_multiview_inputs(
            batch, has_cxr, has_text,
            disable_cxr=disable_cxr,
            disable_txt=disable_txt,
        )

    # ==================== 3. Forward Pass 및 Loss 계산 ====================
    use_info_nce = is_training and info_weight > 0.0
    
    with timer("Batch별 Embedding 추출 및 Loss 연산 총", accelerator):
        with accelerator.autocast():
            time_steps = batch['time_steps']

            model_outputs = model(
                args, ts_series, cxr_data, text_data,
                has_cxr, has_text,
                time_steps=time_steps,
            )

            edema_logits = model_outputs['edema_logits']
            cardiomegaly_logits = model_outputs['cardiomegaly_logits']
            pneumonia_logits = model_outputs['pneumonia_logits']

            info_loss_t = torch.tensor(0.0, device=device)
        
            if use_info_nce:
                proj_clinical = model_outputs['proj_emb']
                proj_text = model_outputs['proj_text_emb']
                valid_txt_mask = model_outputs['valid_txt_mask']

                if valid_txt_mask.sum() > 1:
                    info_loss_t = loss_module.info_nce_loss(
                        proj_clinical[valid_txt_mask], 
                        proj_text[valid_txt_mask]
                    )
            
            # cardio/pneumo 라벨은 CXR이 있는 window에서만 의미 있음 → cxr_anchor=0이면 NaN으로 마스킹
            cxr_mask = (cxr_anchor_mask == 1)
            cardio_targets = torch.where(cxr_mask, cardiomegaly_labels, torch.full_like(cardiomegaly_labels, float('nan')))
            pneumo_targets = torch.where(cxr_mask, pneumonia_labels, torch.full_like(pneumonia_labels, float('nan')))

            with timer("Main loss 연산", accelerator):
                total_batch_loss, bce_loss_t, ce_loss_t, cardio_loss_t, pneumo_loss_t, final_info_loss_t, loss_counts = loss_module(
                    edema_logits=edema_logits,
                    edema_soft_labels=edema_soft_labels,
                    cardiomegaly_logits=cardiomegaly_logits, cardiomegaly_labels=cardio_targets, cardio_weight=cardio_weight,
                    pneumonia_logits=pneumonia_logits, pneumonia_labels=pneumo_targets, pneumo_weight=pneumo_weight,
                    # subtype_logits=subtype_logits, subtype_labels=subtype_labels, ce_weight=...,
                    bce_weight=bce_weight,
                    info_loss_t=info_loss_t, info_weight=info_weight,
                    # contrast_weight=contrast_weight if use_contrastive else 0.0, proj_view1=proj_view1, proj_view2=proj_view2,
                    device=device, accelerator=accelerator,
                )

    # ==================== 4. Metrics 수집 ====================
    batch_bce = float(bce_loss_t.detach().item())
    batch_ce = float(ce_loss_t.detach().item())  # 현재는 0으로만 처리됨.(향후 cardio/noncardio task용)
    batch_cardio = float(cardio_loss_t.detach().item())
    batch_pneumo = float(pneumo_loss_t.detach().item())
    batch_info = float(final_info_loss_t.detach().item())

    batch_outputs = {
        'edema_soft_labels': edema_soft_labels,
        'edema_hard_labels': edema_hard_labels,
        'cxr_anchor_mask': cxr_anchor_mask,
        'edema_logits': edema_logits,
        'cardiomegaly_logits': cardiomegaly_logits,
        'pneumonia_logits': pneumonia_logits,
        'cardiomegaly_labels': cardiomegaly_labels,
        'pneumonia_labels': pneumonia_labels,
        # 'subtype_labels': subtype_labels,
        # 'subtype_logits': subtype_logits,
    }

    batch_counts = {
        'bce_count': loss_counts['bce_count'],
        'ce_count': loss_counts['ce_count'],
        'cardio_count': loss_counts['cardio_count'],
        'pneumo_count': loss_counts['pneumo_count'],
        'info_count': loss_counts['info_count'],
    }
    return total_batch_loss, batch_bce, batch_ce, batch_cardio, batch_pneumo, batch_info, batch_outputs, batch_counts


def prepare_multiview_inputs(batch, has_cxr, has_text, disable_cxr=False, disable_txt=False):
    """
    사전 임베딩(.pt) 기반. 인코딩/패딩/augmentation 없음.
    """
    ts_series = batch['ts_tensor']
    img_index_tensor = batch['img_index_tensor']
    txt_index_tensor = batch['text_index_tensor']

    # ==================== CXR ====================
    if not disable_cxr:
        valid_positions = (img_index_tensor != -1).nonzero(as_tuple=False)
        if len(valid_positions) > 0:
            unique_indices = img_index_tensor[valid_positions[:, 0], valid_positions[:, 1]]
        else:
            unique_indices = torch.empty(0, dtype=torch.long)
        cxr_data = {
            'unique_embs': batch['unique_img_embs'],
            'unique_segment_embs': batch['unique_segment_embs'],
            'unique_ctr': batch['unique_ctr'],
            'segment_index_tensor': batch['segment_index_tensor'],
            'unique_indices': unique_indices,
            'positions': valid_positions,
        }
    else:
        cxr_data = {
            'unique_embs': torch.empty(0),
            'unique_segment_embs': torch.empty(0),
            'unique_ctr': torch.empty(0),
            'segment_index_tensor': torch.empty(0, dtype=torch.long),
            'unique_indices': torch.empty(0, dtype=torch.long),
            'positions': torch.empty(0, 2, dtype=torch.long),
        }

    # ==================== TEXT ====================
    if not disable_txt:
        valid_positions = (txt_index_tensor != -1).nonzero(as_tuple=False)
        if len(valid_positions) > 0:
            unique_indices = txt_index_tensor[valid_positions[:, 0], valid_positions[:, 1]]
        else:
            unique_indices = torch.empty(0, dtype=torch.long)
        text_data = {
            'unique_embs': batch['unique_text_embs'],
            'unique_indices': unique_indices,
            'positions': valid_positions,
        }
    else:
        text_data = {
            'unique_embs': torch.empty(0),
            'unique_indices': torch.empty(0, dtype=torch.long),
            'positions': torch.empty(0, 2, dtype=torch.long),
        }

    return ts_series, cxr_data, text_data, has_cxr, has_text




# def build_two_views(ts_series, value_col_idx, flag_col_idx, mask_prob=0.15):
#     """
#     ts_series: [B, T, D] — value 컬럼과 flag 컬럼이 한 텐서에 함께 들어있음
#     value_col_idx: [Dv] long — value 컬럼의 last-dim 인덱스
#     flag_col_idx:  [Dv] long — 위 value에 대응하는 '<var>_flag' 컬럼 인덱스 (순서 일치)
#     mask_prob: flag==1 셀 중 얼마나 가릴지

#     return: (view1, view2) — 둘 다 [B, T, D]. 마스킹된 위치는 value=0, flag=0.
#     같은 mask_prob로 독립 샘플링되어 두 개의 augmentation view를 만든다.
#     """
#     ts_values = ts_series.index_select(-1, value_col_idx)  # [B, T, Dv]
#     ts_flags = ts_series.index_select(-1, flag_col_idx)    # [B, T, Dv]

#     v1_vals, v1_flags = feature_masking(ts_values, ts_flags, mask_prob=mask_prob)
#     v2_vals, v2_flags = feature_masking(ts_values, ts_flags, mask_prob=mask_prob)

#     view1 = ts_series.clone()
#     view1[..., value_col_idx] = v1_vals
#     view1[..., flag_col_idx] = v1_flags.to(view1.dtype)

#     view2 = ts_series.clone()
#     view2[..., value_col_idx] = v2_vals
#     view2[..., flag_col_idx] = v2_flags.to(view2.dtype)

#     return view1, view2


# def feature_masking(ts_tensor, flag_tensor, mask_prob=0.15):
#     """
#     ts_tensor: [B, T, D] - 스케일링된 시계열 데이터
#     flag_tensor: [B, T, D] - 각 변수 및 시간대별 측정 여부 (1: 존재, 0: 결측)
#     mask_prob: 유효한 데이터 중 몇 %를 인위적으로 가릴 것인가 (기본 15%)
#     """
#     # 1. 원본 훼손 방지를 위한 복제
#     aug_ts = ts_tensor.clone()
#     aug_flags = flag_tensor.clone()

#     # 2. 0~1 사이의 난수 생성 [B, T, D]
#     rand_matrix = torch.rand_like(aug_ts)

#     # 3. [핵심] 인위적 결측 타겟 선정
#     # 조건 A: 원래 데이터가 존재하는 곳 (flag_tensor == 1)
#     # 조건 B: 난수가 mask_prob 보다 작은 곳 (예: 15% 확률)
#     # 두 조건을 모두 만족하는 위치만 True가 되는 Boolean Mask 생성
#     target_mask = (flag_tensor == 1) & (rand_matrix < mask_prob)

#     # 4. 타겟 위치 마스킹 수행
#     # 선택된 위치의 시계열 값을 0.0 (또는 평균값)으로 만듦
#     aug_ts[target_mask] = 0.0
    
#     # 해당 위치의 플래그도 0으로 변경하여, 모델이 이 값이 결측되었음을 알게 함
#     aug_flags[target_mask] = 0 

#     return aug_ts, aug_flags