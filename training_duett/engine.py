from __future__ import annotations

import torch
import torch.nn as nn


def _move_lists(batch: dict, device: torch.device) -> dict:
    """Move list-of-tensors entries (x_ts, x_static, bin_ends) to device."""
    out = {
        "x_ts": tuple(t.to(device) for t in batch["x_ts"]),
        "x_static": tuple(t.to(device) for t in batch["x_static"]),
        "bin_ends": tuple(t.to(device) for t in batch["bin_ends"]),
        "y": batch["y"].to(device),
    }
    if "pixel_values" in batch:
        out["pixel_values"] = batch["pixel_values"].to(device)
    if "y_multi" in batch:
        out["y_multi"] = batch["y_multi"].to(device)
        out["y_multi_mask"] = batch["y_multi_mask"].to(device)
    return out


# =============================================================================
# Teacher
# =============================================================================
def train_teacher_batch(batch, teacher, loss_fn, optimizer, device, accelerator=None, aux_alpha: float = 0.0):
    """Teacher 1-step. TeacherModel이 tuple (main, aux)를 반환하면 aux BCE를 함께 계산."""
    teacher.train()
    b = _move_lists(batch, device)
    out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"])

    aux_loss_val = 0.0
    if isinstance(out, tuple):
        main_logit, aux_logit = out
        main_loss = loss_fn(main_logit, b["y"].float())
        aux_loss  = loss_fn(aux_logit,  b["y"].float())
        loss = main_loss + aux_alpha * aux_loss
        aux_loss_val = aux_loss.detach().item()
    else:
        main_logit = out
        main_loss = loss_fn(main_logit, b["y"].float())
        loss = main_loss

    optimizer.zero_grad()
    if accelerator is not None:
        accelerator.backward(loss)
    else:
        loss.backward()
    optimizer.step()

    return {
        "loss": loss.detach().item(),
        "main_loss": main_loss.detach().item(),
        "aux_loss": aux_loss_val,
        "logits": main_logit.detach(),
        "y": b["y"].detach(),
    }


@torch.no_grad()
def eval_teacher_batch(batch, teacher, loss_fn, device):
    teacher.eval()
    b = _move_lists(batch, device)
    out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"])
    if isinstance(out, dict):
        main_logit = out["main_logit"]
    elif isinstance(out, tuple):
        main_logit = out[0]
    else:
        main_logit = out
    loss = loss_fn(main_logit, b["y"].float())
    return {"loss": loss.item(), "logits": main_logit, "y": b["y"]}


# =============================================================================
# Teacher — Pathology mode (multi-label BCE on stage2 + stage4)
# =============================================================================
def train_teacher_pathology_batch(batch, teacher, path_loss_fn, optimizer, device, accelerator=None):
    """PathologyPerceiver 학습 1-step.

    teacher(...)는 dict(main_logit, stage2_logits, stage4_logits)를 반환한다고 가정.
    path_loss_fn 은 PathologyMultiLabelLoss.
    """
    teacher.train()
    b = _move_lists(batch, device)
    out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"])
    if not isinstance(out, dict):
        raise RuntimeError("pathology 모드인데 TeacherModel이 dict를 반환하지 않았습니다.")

    losses = path_loss_fn(
        out["stage2_logits"], out["stage4_logits"],
        b["y_multi"], b["y_multi_mask"])

    optimizer.zero_grad()
    if accelerator is not None:
        accelerator.backward(losses["total"])
    else:
        losses["total"].backward()
    optimizer.step()

    return {
        "loss":         losses["total"].detach().item(),
        "stage2_total": losses["stage2_total"].item(),
        "stage4_total": losses["stage4_total"].item(),
        "stage2_per":   losses["stage2_per"].cpu(),      # [K]
        "stage4_per":   losses["stage4_per"].cpu(),      # [K]
        "main_logit":   out["main_logit"].detach(),
        "stage2_logits": out["stage2_logits"].detach(),
        "stage4_logits": out["stage4_logits"].detach(),
        "y":            b["y"].detach(),
        "y_multi":      b["y_multi"].detach(),
        "y_multi_mask": b["y_multi_mask"].detach(),
    }


# =============================================================================
# Teacher — Dual Pathology mode (image / ts / fusion 3-way BCE)
# =============================================================================
def train_teacher_dual_pathology_batch(batch, teacher, path_loss_fn, optimizer, device, accelerator=None):
    teacher.train()
    b = _move_lists(batch, device)
    out = teacher(b["x_ts"], b["x_static"], b["bin_ends"], b["pixel_values"])
    if not isinstance(out, dict):
        raise RuntimeError("dual_pathology 모드인데 TeacherModel이 dict를 반환하지 않았습니다.")

    losses = path_loss_fn(out["img_logits"], out["ts_logits"], out["fusion_logits"], b["y_multi"], b["y_multi_mask"])

    optimizer.zero_grad()
    if accelerator is not None:
        accelerator.backward(losses["total"])
    else:
        losses["total"].backward()
    optimizer.step()

    return {
        "loss":       losses["total"].detach().item(),
        "img_total":  losses["img_total"].item(),
        "ts_total":   losses["ts_total"].item(),
        "fus_total":  losses["fus_total"].item(),
        "img_per":    losses["img_per"].cpu(),        # [K]
        "ts_per":     losses["ts_per"].cpu(),         # [K]
        "fus_per":    losses["fus_per"].cpu(),        # [K]
        "main_logit":    out["main_logit"].detach(),
        "img_logits":    out["img_logits"].detach(),
        "ts_logits":     out["ts_logits"].detach(),
        "fusion_logits": out["fusion_logits"].detach(),
        "y":             b["y"].detach(),
        "y_multi":       b["y_multi"].detach(),
        "y_multi_mask":  b["y_multi_mask"].detach(),
    }


# =============================================================================
# Student (KD)
# =============================================================================
def train_student_batch(batch_stu, batch_tea, student, teacher, kd_loss_fn,
                        optimizer, device, accelerator=None):
    """Batch inputs are already paired (same anchor rows in same order).

    batch_stu: student loader batch (no pixel_values)
    batch_tea: teacher loader batch (with pixel_values), same y
    """
    student.train()
    teacher.eval()

    b_s = _move_lists(batch_stu, device)
    b_t = _move_lists(batch_tea, device)

    with torch.no_grad():
        z_t = teacher(b_t["x_ts"], b_t["x_static"], b_t["bin_ends"], b_t["pixel_values"])["main_logit"]
    z_s = student(b_s["x_ts"], b_s["x_static"], b_s["bin_ends"])
    losses = kd_loss_fn(z_s, z_t, b_s["y"])

    optimizer.zero_grad()
    if accelerator is not None:
        accelerator.backward(losses["total"])
    else:
        losses["total"].backward()
    optimizer.step()

    return {
        "loss": losses["total"].detach().item(),
        "bce": losses["bce"].item(),
        "kd": losses["kd"].item(),
        "logits": z_s.detach(),
        "y": b_s["y"].detach(),
    }


@torch.no_grad()
def eval_student_batch(batch, student, device):
    student.eval()
    b = _move_lists(batch, device)
    z = student(b["x_ts"], b["x_static"], b["bin_ends"])
    return {"logits": z, "y": b["y"]}