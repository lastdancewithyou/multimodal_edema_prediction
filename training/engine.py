import warnings
warnings.filterwarnings('ignore', message='Spectral initialisation failed')

import torch

from utils.utils import timer


# 단일 배치 학습 함수 (LUPI dual-stream)
def train_batch(
        args, model, batch, loss_module, device, accelerator,
        disable_cxr=False, disable_txt=False,
        is_training=True, eval_path='deploy',
        disable_txt_priv=False,
    ):

    if is_training:
        model.train()

    # ==================== 1. 배치 데이터 준비 ====================
    with timer("Batch Data preparation", accelerator):
        edema_soft_labels = batch['edema_soft_labels']
        edema_hard_labels = batch['edema_hard_labels']
        cxr_anchor_mask = batch['cxr_anchor_mask']
        subtype_soft_labels = batch['subtype_soft_labels']      # [B, 3]
        subtype_label       = batch['subtype_label']            # [B] long (-1 = invalid)
        subtype_mask        = batch['subtype_mask']             # [B] float

        img_index_tensor = batch['img_index_tensor']
        txt_index_tensor = batch['text_index_tensor']
        has_cxr = (img_index_tensor != -1).long()
        has_text = (txt_index_tensor != -1).long()

    # ==================== 2. 모델 입력 데이터 전처리 ====================
    with timer("데이터 전처리 작업", accelerator):
        ts_series, cxr_data, text_data, has_cxr, has_text = prepare_multiview_inputs(
            batch, has_cxr, has_text,
            disable_cxr=disable_cxr,
            disable_txt=disable_txt,
        )

    # ==================== 3. Forward Pass (Dual-stream) ====================
    with timer("Dual-stream forward + Loss 연산 총", accelerator):
        with accelerator.autocast():
            time_steps = batch['time_steps']

            # 평가 시 priv path를 보고 싶으면 lupi_mode=True 강제하여 두 path 모두 계산
            need_priv = is_training or eval_path == 'priv'
            model_outputs = model(
                args, ts_series, cxr_data, text_data,
                has_cxr, has_text,
                time_steps=time_steps,
                ts_valid_mask=batch['ts_valid_mask'],
                lupi_mode=need_priv,
                disable_txt_priv=disable_txt_priv,   # Modality Dropout on teacher
            )

            logit_priv   = model_outputs['logit_priv']
            logit_deploy = model_outputs['logit_deploy']
            fused_priv   = model_outputs['fused_priv']
            fused_deploy = model_outputs['fused_deploy']
            feat_priv    = model_outputs.get('feat_priv',   None)
            feat_deploy  = model_outputs.get('feat_deploy', None)
            subtype_logits_deploy = model_outputs.get('subtype_logits_deploy', None)
            subtype_logits_priv   = model_outputs.get('subtype_logits_priv',   None)

            # 평가 시 어떤 path의 logit을 메트릭에 쓸지 선택
            eval_logit = logit_priv if (not is_training and eval_path == 'priv') else logit_deploy
            eval_subtype_logits = (
                subtype_logits_priv if (not is_training and eval_path == 'priv')
                else subtype_logits_deploy
            )

            if is_training:
                with timer("LUPI Loss 연산", accelerator):
                    (
                        total_batch_loss,
                        bce_priv_t, bce_deploy_t,
                        fd_loss_t, rd_loss_t, kd_loss_t, cov_loss_t,
                        subtype_loss_t,
                        loss_counts,
                    ) = loss_module(
                        logit_priv=logit_priv,
                        logit_deploy=logit_deploy,
                        fused_priv=fused_priv,
                        fused_deploy=fused_deploy,
                        feat_priv=feat_priv,
                        feat_deploy=feat_deploy,
                        edema_soft_labels=edema_soft_labels,
                        subtype_logits_priv=subtype_logits_priv,
                        subtype_logits_deploy=subtype_logits_deploy,
                        subtype_target_probs=subtype_soft_labels,
                        subtype_mask=subtype_mask,
                        device=device, accelerator=accelerator,
                    )
            else:
                # eval/inference: 선택된 path에 BCE only (메트릭 추적용)
                bce_deploy_t, bce_count = loss_module.deploy_bce(eval_logit, edema_soft_labels)
                total_batch_loss = bce_deploy_t
                bce_priv_t = torch.tensor(0.0, device=device)
                fd_loss_t  = torch.tensor(0.0, device=device)
                rd_loss_t  = torch.tensor(0.0, device=device)
                kd_loss_t  = torch.tensor(0.0, device=device)
                cov_loss_t = torch.tensor(0.0, device=device)
                subtype_loss_t = torch.tensor(0.0, device=device)
                loss_counts = {
                    'bce_count': bce_count, 'rd_cos': 0.0, 'rd_mse': 0.0,
                    'subtype_count': 0,
                    'subtype_loss_priv': 0.0, 'subtype_loss_deploy': 0.0,
                }

    # ==================== 4. Metrics ====================
    batch_bce_priv   = float(bce_priv_t.detach().item())
    batch_bce_deploy = float(bce_deploy_t.detach().item())
    batch_fd         = float(fd_loss_t.detach().item())
    batch_rd         = float(rd_loss_t.detach().item())
    batch_kd         = float(kd_loss_t.detach().item())
    batch_cov        = float(cov_loss_t.detach().item())
    batch_subtype    = float(subtype_loss_t.detach().item())

    # ── Privileged vs Deploy 경로 차이 진단 ──
    diag_fused_diff = 0.0
    diag_logit_diff = 0.0
    if is_training and fused_priv is not None and fused_deploy is not None:
        with torch.no_grad():
            diag_fused_diff = float((fused_priv - fused_deploy).abs().mean().item())
            diag_logit_diff = float((logit_priv - logit_deploy).abs().mean().item())

    # 평가/추론용 logit
    metric_logit = logit_priv if (not is_training and eval_path == 'priv') else logit_deploy
    batch_outputs = {
        'edema_soft_labels': edema_soft_labels,
        'edema_hard_labels': edema_hard_labels,
        'cxr_anchor_mask':   cxr_anchor_mask,
        'edema_logits':      metric_logit,
        'logit_priv':        logit_priv,       # 진단용
        # ── Subtype eval outputs (eval_path에 맞춰 priv/deploy 라우팅) ──
        'subtype_logits':  eval_subtype_logits,      # [B, 3] or None
        'subtype_label':   subtype_label,            # [B] long, -1 where mask=0
        'subtype_mask':    subtype_mask,             # [B] float
    }

    batch_counts = {
        'bce_count':       loss_counts['bce_count'],
        'bce_priv_loss':   batch_bce_priv,
        'bce_deploy_loss': batch_bce_deploy,
        'fd_loss':         batch_fd,
        'rd_loss':         batch_rd,
        'rd_cos':          loss_counts.get('rd_cos', 0.0),
        'rd_mse':          loss_counts.get('rd_mse', 0.0),
        'kd_loss':         batch_kd,
        'cov_loss':        batch_cov,
        'subtype_loss':        batch_subtype,
        'subtype_loss_priv':   loss_counts.get('subtype_loss_priv',   0.0),
        'subtype_loss_deploy': loss_counts.get('subtype_loss_deploy', 0.0),
        'subtype_count':       loss_counts.get('subtype_count', 0),
        'diag_fused_diff': diag_fused_diff,
        'diag_logit_diff': diag_logit_diff,
    }
    return total_batch_loss, batch_bce_priv, batch_bce_deploy, batch_outputs, batch_counts


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
            'segment_index_tensor': batch['segment_index_tensor'],
            'unique_indices': unique_indices,
            'positions': valid_positions,
        }
    else:
        cxr_data = {
            'unique_embs': torch.empty(0),
            'unique_segment_embs': torch.empty(0),
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
            'unique_input_ids':  batch['unique_text_input_ids'],   # [N_uniq, 128] long
            'unique_attn_mask':  batch['unique_text_attn_mask'],   # [N_uniq, 128] long
            'unique_indices':    unique_indices,
            'positions':         valid_positions,
        }
    else:
        text_data = {
            'unique_input_ids':  torch.empty(0, 128, dtype=torch.long),
            'unique_attn_mask':  torch.empty(0, 128, dtype=torch.long),
            'unique_indices':    torch.empty(0, dtype=torch.long),
            'positions':         torch.empty(0, 2, dtype=torch.long),
        }

    return ts_series, cxr_data, text_data, has_cxr, has_text
