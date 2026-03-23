import os
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore', message='Spectral initialisation failed')

import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils.utils import timer

# 단일 배치 학습 함수
def train_batch(args, model, batch, loss_module, device, accelerator, dataset, disable_cxr=False, disable_txt=False, max_length=256,
                bce_weight=None, ce_weight=None
    ):
    model.train()

    # ==================== 1. 배치 데이터 GPU 전송 ====================
    with timer("Batch Data preparation", accelerator):
        # New multi-task format
        for k in ['edema_labels', 'subtype_labels', 'window_mask', 'valid_seq_mask']:
            batch[k] = batch[k].to(device, non_blocking=True)
        edema_labels = batch['edema_labels']
        subtype_labels = batch['subtype_labels']



        img_index_tensor = batch['img_index_tensor']
        txt_index_tensor = batch['text_index_tensor']
        has_cxr = (img_index_tensor != -1).long().to(device, non_blocking=True)   # [B, W, T]
        has_text = (txt_index_tensor != -1).long().to(device, non_blocking=True)  # [B, W, T]

        window_mask = batch['window_mask']
        seq_valid_mask = batch['valid_seq_mask']

    # ==================== 2. 모델 입력 데이터 전처리 ====================
    with timer("데이터 전처리 작업", accelerator):
        ts_series, cxr_views, text_series, has_cxr, has_text = prepare_multiview_inputs_v2(
            batch, device, has_cxr, has_text,
            dataset=dataset,
            disable_cxr=disable_cxr,
            disable_txt=disable_txt,
            max_length=max_length,
        )

    # ==================== 3. Forward Pass 및 Loss 계산 ====================
    with timer("Batch별 Embedding 추출 및 Loss 연산 총", accelerator):
        with accelerator.autocast():
            time_steps = batch.get('time_steps').to(device, non_blocking=True)

            # Prepare prompt data
            prompt_data = {
                'unique_prompt_texts': batch['unique_prompt_texts'],
                'prompt_index_tensor': batch['prompt_index_tensor'].to(device, non_blocking=True)
            }

            model_outputs = model(
                args, ts_series, cxr_views, text_series, prompt_data, has_cxr, has_text,
                window_mask, seq_valid_mask, time_steps=time_steps
            )

            edema_logits = model_outputs['edema_logits']                # [B, W, 1]
            subtype_logits = model_outputs['subtype_logits']            # [B, W, 2]
            valid_embeddings = model_outputs['valid_embeddings']        # [Nwin, 256]
            window_time_indices = model_outputs['window_time_indices']  # [Nwin]
            batch_indices = model_outputs['batch_indices']              # [Nwin]

            with timer("Main loss 연산", accelerator):
                total_batch_loss, bce_loss_t, ce_loss_t, loss_counts = loss_module(
                    edema_logits = edema_logits,
                    subtype_logits = subtype_logits,
                    valid_embeddings = valid_embeddings,
                    window_time_indices = window_time_indices,
                    batch_indices = batch_indices,
                    edema_labels=edema_labels,
                    subtype_labels=subtype_labels,
                    window_mask=window_mask,
                    bce_weight=bce_weight,
                    ce_weight=ce_weight,
                    device=device,
                    accelerator=accelerator
                )

    # ==================== 5. Metrics 수집 ====================
    window_count = window_mask.sum().item()
    batch_bce = float(bce_loss_t.detach().item())
    batch_ce = float(ce_loss_t.detach().item())

    batch_outputs = {
        'edema_labels': edema_labels,
        'subtype_labels': subtype_labels,
        'edema_logits': edema_logits,
        'subtype_logits': subtype_logits
    }

    batch_counts = {
        'window_count': window_count,
        'bce_count': loss_counts['bce_count'],
        'ce_count': loss_counts['ce_count'],
    }

    return total_batch_loss, batch_bce, batch_ce, batch_outputs, batch_counts


def prepare_multiview_inputs_v2(batch, device, has_cxr, has_text, dataset, disable_cxr=False, disable_txt=False, max_length=256):

    ts = batch['ts_tensor']
    img_index_tensor = batch['img_index_tensor']
    txt_index_tensor = batch['text_index_tensor']
    unique_img_paths = batch['unique_img_paths']
    unique_txt_keys = batch['unique_txt_keys']

    # ==================== TEXT PREPARATION ====================
    if not disable_txt:
        text_ids_list, text_masks_list = [], []
        for stay_id, hour in unique_txt_keys:
            token = dataset.text_map[stay_id][hour]
            ids = torch.tensor(token['input_ids'], dtype=torch.long)
            mask = torch.tensor(token['attention_mask'], dtype=torch.long)

            current_len = len(ids)
            if current_len > max_length:
                ids = ids[:max_length]
                mask = mask[:max_length]
            elif current_len < max_length:
                pad_len = max_length - current_len
                ids = F.pad(ids, (0, pad_len), value=0)
                mask = F.pad(mask, (0, pad_len), value=0)

            text_ids_list.append(ids)
            text_masks_list.append(mask)

        if len(unique_txt_keys) > 0:
            unique_text_ids = torch.stack(text_ids_list, dim=0).to(device, non_blocking=True)
            unique_text_masks = torch.stack(text_masks_list, dim=0).to(device, non_blocking=True)
        else:
            unique_text_ids = torch.empty(0, max_length, dtype=torch.long, device=device)
            unique_text_masks = torch.empty(0, max_length, dtype=torch.long, device=device)
    else:
        unique_text_ids = torch.empty(0, max_length, dtype=torch.long, device=device)
        unique_text_masks = torch.empty(0, max_length, dtype=torch.long, device=device)

    # ==================== IMG PREPARATION ====================
    if not disable_cxr:
        if len(unique_img_paths) > 0:
            unique_imgs = torch.stack(
                [dataset.load_image_cached(path) for path in unique_img_paths],
                dim=0
            ).to(device, non_blocking=True)
        else:
            num_channels = 3 if dataset.to_3ch else 1
            unique_imgs = torch.empty(0, num_channels, 224, 224, device=device)
    else:
        num_channels = 3 if dataset.to_3ch else 1
        unique_imgs = torch.empty(0, num_channels, 224, 224, device=device)

    ts_series = ts.to(device, non_blocking=True)  # [B, W, T, D]

    # Text data structure
    if not disable_txt:
        valid_positions = (txt_index_tensor != -1).nonzero(as_tuple=False)
        if len(valid_positions) > 0:
            unique_indices = txt_index_tensor[valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]]
            text_data = {
                'unique_input_ids': unique_text_ids,
                'unique_attention_mask': unique_text_masks,
                'unique_indices': unique_indices,
                'positions': valid_positions
            }
        else:
            text_data = {
                'unique_input_ids': torch.empty(0, max_length, dtype=torch.long, device=device),
                'unique_attention_mask': torch.empty(0, max_length, dtype=torch.long, device=device),
                'unique_indices': torch.empty(0, dtype=torch.long, device=device),
                'positions': torch.empty(0, 3, dtype=torch.long, device=device)
            }
    else:
        text_data = {
            'unique_input_ids': torch.empty(0, max_length, dtype=torch.long, device=device),
            'unique_attention_mask': torch.empty(0, max_length, dtype=torch.long, device=device),
            'unique_indices': torch.empty(0, dtype=torch.long, device=device),
            'positions': torch.empty(0, 3, dtype=torch.long, device=device)
        }

    # CXR data structure
    if not disable_cxr:
        valid_positions = (img_index_tensor != -1).nonzero(as_tuple=False)
        if len(valid_positions) > 0:
            unique_indices = img_index_tensor[valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]]
            cxr_data = {
                'unique_images': unique_imgs,
                'unique_indices': unique_indices,
                'positions': valid_positions
            }
        else:
            num_channels = 3 if dataset.to_3ch else 1
            cxr_data = {
                'unique_images': torch.empty(0, num_channels, 224, 224, device=device),
                'unique_indices': torch.empty(0, dtype=torch.long, device=device),
                'positions': torch.empty(0, 3, dtype=torch.long, device=device)
            }
    else:
        num_channels = 3 if dataset.to_3ch else 1
        cxr_data = {
            'unique_images': torch.empty(0, num_channels, 224, 224, device=device),
            'unique_indices': torch.empty(0, dtype=torch.long, device=device),
            'positions': torch.empty(0, 3, dtype=torch.long, device=device)
        }

    return ts_series, cxr_data, text_data, has_cxr, has_text