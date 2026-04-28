import os
import warnings
warnings.filterwarnings('ignore', message='Spectral initialisation failed')

import torch
import torch.nn.functional as F

from utils.utils import timer


# 단일 배치 학습 함수
def train_batch(
        args, model, batch, loss_module, device, accelerator, dataset, disable_cxr=False, disable_txt=False, max_length=256,
        bce_weight=None, ce_weight=None, current_epoch=0, is_training=True
    ):

    if is_training:
        model.train()

    # ==================== 1. 배치 데이터 준비 ====================
    with timer("Batch Data preparation", accelerator):
        edema_labels = batch['edema_labels']
        subtype_labels = batch['subtype_labels']
        # score_diff_targets = batch['score_diff_targets']

        img_index_tensor = batch['img_index_tensor']
        txt_index_tensor = batch['text_index_tensor']
        has_cxr = (img_index_tensor != -1).long()   # [B, T]
        has_text = (txt_index_tensor != -1).long()  # [B, T]

    # ==================== 2. 모델 입력 데이터 전처리 ====================
    with timer("데이터 전처리 작업", accelerator):
        ts_series, cxr_views, text_series, has_cxr, has_text = prepare_multiview_inputs(
            batch, has_cxr, has_text,
            dataset=dataset,
            disable_cxr=disable_cxr,
            disable_txt=disable_txt,
            max_length=max_length,
            is_training=True
        )

    # ==================== 3. Forward Pass 및 Loss 계산 ====================
    with timer("Batch별 Embedding 추출 및 Loss 연산 총", accelerator):
        with accelerator.autocast():
            time_steps = batch['time_steps']

            prompt_data = {
                'unique_prompt_texts': batch['unique_prompt_texts'],
                'prompt_index_tensor': batch['prompt_index_tensor']
            }

            model_outputs = model(args, ts_series, cxr_views, text_series, prompt_data, has_cxr, has_text,time_steps=time_steps,
                                current_epoch=current_epoch, total_epoch=args.epochs)

            edema_logits = model_outputs['edema_logits']                
            subtype_logits = model_outputs['subtype_logits']            
            align_loss = model_outputs['align_loss']

            with timer("Main loss 연산", accelerator):
                total_batch_loss, bce_loss_t, ce_loss_t, loss_counts = loss_module(
                    edema_logits=edema_logits,
                    subtype_logits=subtype_logits,
                    edema_labels=edema_labels,
                    subtype_labels=subtype_labels,
                    bce_weight=bce_weight,
                    ce_weight=ce_weight,
                    device=device,
                    accelerator=accelerator
                )
                total_batch_loss = total_batch_loss + args.align_weight * align_loss

    # ==================== 5. Metrics 수집 ====================
    batch_bce = float(bce_loss_t.detach().item())
    batch_ce = float(ce_loss_t.detach().item())

    batch_outputs = {
        'edema_labels': edema_labels,
        'subtype_labels': subtype_labels,
        'edema_logits': edema_logits,
        'subtype_logits': subtype_logits,
        'align_loss': align_loss.detach().item(),
    }

    batch_counts = {
        'bce_count': loss_counts['bce_count'],
        'ce_count': loss_counts['ce_count'],
    }
    return total_batch_loss, batch_bce, batch_ce, batch_outputs, batch_counts


def prepare_multiview_inputs(batch, has_cxr, has_text, dataset, disable_cxr=False, disable_txt=False, max_length=256, is_training=False):
    ts_data = batch['ts_tensor']
    img_index_tensor = batch['img_index_tensor']
    txt_index_tensor = batch['text_index_tensor']
    unique_img_paths = batch['unique_img_paths']
    unique_txt_keys = batch['unique_txt_keys']

    # ==================== TEXT PREPARATION ====================
    if not disable_txt and len(unique_txt_keys) > 0:
        all_ids = []
        all_masks = []

        for stay_id, hour in unique_txt_keys:
            token = dataset.text_map[stay_id][hour]
            all_ids.append(token['input_ids'])
            all_masks.append(token['attention_mask'])

        text_ids_list = []
        text_masks_list = []

        for ids, mask in zip(all_ids, all_masks):
            current_len = len(ids)

            if current_len > max_length:
                # Truncate
                ids_tensor = torch.tensor(ids[:max_length], dtype=torch.long)
                mask_tensor = torch.tensor(mask[:max_length], dtype=torch.long)
            elif current_len < max_length:
                # Pad
                ids_tensor = torch.tensor(ids, dtype=torch.long)
                mask_tensor = torch.tensor(mask, dtype=torch.long)
                pad_len = max_length - current_len
                ids_tensor = F.pad(ids_tensor, (0, pad_len), value=0)
                mask_tensor = F.pad(mask_tensor, (0, pad_len), value=0)
            else:
                # Exact length
                ids_tensor = torch.tensor(ids, dtype=torch.long)
                mask_tensor = torch.tensor(mask, dtype=torch.long)

            text_ids_list.append(ids_tensor)
            text_masks_list.append(mask_tensor)

        unique_text_ids = torch.stack(text_ids_list, dim=0)
        unique_text_masks = torch.stack(text_masks_list, dim=0)
    else:
        unique_text_ids = torch.empty(0, max_length, dtype=torch.long)
        unique_text_masks = torch.empty(0, max_length, dtype=torch.long)

    # ==================== IMG PREPARATION ====================
    if not disable_cxr:
        if len(unique_img_paths) > 0:
            unique_imgs = torch.stack([dataset.load_image_cached(path) for path in unique_img_paths],dim=0)
            unique_imgs = unique_imgs.to(torch.float32)

            # Apply CPU-based CenterCrop for validation (non-training)
            if not is_training:
                unique_imgs = cxr_val_transform(unique_imgs)
            # For training, GPU augmentation will be applied after moving to device
        else:
            num_channels = 3 if dataset.to_3ch else 1
            unique_imgs = torch.empty(0, num_channels, 224, 224)
    else:
        num_channels = 3 if dataset.to_3ch else 1
        unique_imgs = torch.empty(0, num_channels, 224, 224)

    ts_series = ts_data

    # Text data structure
    if not disable_txt:
        valid_positions = (txt_index_tensor != -1).nonzero(as_tuple=False)
        if len(valid_positions) > 0:
            unique_indices = txt_index_tensor[valid_positions[:, 0], valid_positions[:, 1]]
            text_data = {
                'unique_input_ids': unique_text_ids,
                'unique_attention_mask': unique_text_masks,
                'unique_indices': unique_indices,
                'positions': valid_positions
            }
        else:
            text_data = {
                'unique_input_ids': torch.empty(0, max_length, dtype=torch.long),
                'unique_attention_mask': torch.empty(0, max_length, dtype=torch.long),
                'unique_indices': torch.empty(0, dtype=torch.long),
                'positions': torch.empty(0, 2, dtype=torch.long)
            }
    else:
        text_data = {
            'unique_input_ids': torch.empty(0, max_length, dtype=torch.long),
            'unique_attention_mask': torch.empty(0, max_length, dtype=torch.long),
            'unique_indices': torch.empty(0, dtype=torch.long),
            'positions': torch.empty(0, 2, dtype=torch.long)
        }

    # Img data structure
    if not disable_cxr:
        valid_positions = (img_index_tensor != -1).nonzero(as_tuple=False)
        if len(valid_positions) > 0:
            unique_indices = img_index_tensor[valid_positions[:, 0], valid_positions[:, 1]]

            # Apply GPU augmentation for training AFTER moving to device
            # This will be done in cxr_data dict, so model can apply it on GPU
            cxr_data = {
                'unique_images': unique_imgs,
                'unique_indices': unique_indices,
                'positions': valid_positions,
                'is_training': is_training  # Pass training flag to model
            }
        else:
            num_channels = 3 if dataset.to_3ch else 1
            cxr_data = {
                'unique_images': torch.empty(0, num_channels, 224, 224),
                'unique_indices': torch.empty(0, dtype=torch.long),
                'positions': torch.empty(0, 2, dtype=torch.long),
                'is_training': is_training
            }
    else:
        num_channels = 3 if dataset.to_3ch else 1
        cxr_data = {
            'unique_images': torch.empty(0, num_channels, 224, 224),
            'unique_indices': torch.empty(0, dtype=torch.long),
            'positions': torch.empty(0, 2, dtype=torch.long),
            'is_training': is_training
        }

    return ts_series, cxr_data, text_data, has_cxr, has_text



#################### IMG augmentation #########################
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import kornia.augmentation as K

background_val = 0.0

# CPU-based transform for validation (CenterCrop only)
cxr_val_transform = T.CenterCrop(224)

# GPU-based augmentation using Kornia for training
cxr_train_transform_gpu = torch.nn.Sequential(
    K.RandomCrop((224, 224)),
    K.RandomRotation(degrees=5.0, p=1.0),
    K.RandomAffine(degrees=0.0, translate=(0.05, 0.05), p=1.0),
    K.ColorJitter(brightness=0.1, contrast=0.1, p=1.0), 
)
############################################################