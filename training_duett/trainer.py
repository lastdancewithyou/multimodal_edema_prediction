from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

from .data_processing import (AnchorConfig, build_datasets, duett_kd_collate,
                              DEFAULT_PATHOLOGY_LABELS)
from .engine import (train_teacher_batch, train_student_batch, eval_teacher_batch, eval_student_batch,
                     train_teacher_pathology_batch, train_teacher_dual_pathology_batch,
                     train_teacher_dual_pathology_lp_batch)
from .evaluator import (evaluate_binary, make_teacher_forward,
                          make_teacher_aux_forward, make_student_forward,
                          evaluate_pathology, format_pathology_gap_table,
                          evaluate_dual_pathology, format_dual_pathology_gap_table)

from models.main_architecture_duett import (
    DuettFeatureExtractor, load_duett_backbone,
    CXREncoder, PatchDualPathologyPerceiver,
    TeacherModel, StudentModel,
)
# Legacy 클래스들은 사용자가 주석 처리할 수 있음 — soft import (없으면 해당 mode 사용 시 명확 에러)
try:
    from models.main_architecture_duett import TemporalPerceiver
except ImportError:
    TemporalPerceiver = None
try:
    from models.main_architecture_duett import PathologyPerceiver
except ImportError:
    PathologyPerceiver = None
try:
    from models.main_architecture_duett import DualPathologyPerceiver
except ImportError:
    DualPathologyPerceiver = None
from loss.losses_duett import StudentKDLoss, PathologyMultiLabelLoss, DualPathologyLoss

warnings.filterwarnings("ignore", message=r".*weights_only.*")
warnings.filterwarnings("ignore", message=r".*rotary embedding dimension.*")


# =============================================================================
# Shared helpers
# =============================================================================
def _make_loader(ds, batch_size, shuffle, num_workers, mode):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda b: duett_kd_collate(b, mode),
        drop_last=shuffle,
    )


def _save_ckpt(path, model, optimizer, epoch, metric, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metric": metric,
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
    }, path)


# =============================================================================
# Differential LR + Warmup/Cosine scheduler
# =============================================================================
def _make_param_groups(model, args):
    """LR groups (dual_patch residual mode 대응):

        backbone(DuETT / CXR)           : args.lr × backbone_lr_mult (기본 0.2)
        pathology_queries (shared query): args.lr × query_lr_mult    (기본 0.2)
        correction_head                 : args.lr × correction_lr_mult (기본 5.0)
        rest (head/perceiver/proj/...)  : args.lr

    query 는 image/ts branch 가 공유하므로 급격히 흔들리면 두 branch 모두 손상 → 낮은 LR.
    correction_head 는 zero-init 라 초기에 gradient 신호가 미미 → 높은 LR 로 빠르게 살림.
    """
    backbone_prefixes = ("duett.", "cxr.")
    backbone, correction, queries, rest = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(backbone_prefixes):
            backbone.append(p)
        elif "correction_head" in name or name.endswith(".beta") or name == "beta":
            # beta 는 correction_head 와 함께 correction group 으로 (correction_lr_mult 공유).
            correction.append(p)
        elif name.endswith("_queries"):
            # pathology_queries (single) / image_queries, temporal_queries (dual_patch)
            queries.append(p)
        else:
            rest.append(p)

    corr_mult  = float(getattr(args, "correction_lr_mult", 5.0))
    query_mult = float(getattr(args, "query_lr_mult", 0.2))

    groups = []
    if backbone:
        groups.append({"params": backbone,   "lr": args.lr * args.backbone_lr_mult, "name": "backbone"})
    if queries:
        groups.append({"params": queries,    "lr": args.lr * query_mult, "name": "pathology_queries"})
    if correction:
        groups.append({"params": correction, "lr": args.lr * corr_mult, "name": "correction_head"})
    if rest:
        groups.append({"params": rest,       "lr": args.lr, "name": "rest"})
    return groups


def _make_scheduler(optimizer, total_steps: int, args):
    """Linear warmup (0 → base lr) → cosine anneal (base lr → base lr × min_lr_ratio)."""
    warmup = max(int(args.warmup_steps), 1)
    cosine_steps = max(int(total_steps) - warmup, 1)
    warmup_sched = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=args.lr * args.min_lr_ratio)
    return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])


def _steps_per_epoch(train_loader, args) -> int:
    n = len(train_loader)
    if args.limit_batches > 0:
        n = min(n, args.limit_batches)
    return n


# =============================================================================
# wandb helpers (main-process only, no-op if disabled)
# =============================================================================
def _wandb_init(args, accelerator, stage: str):
    if args.wandb_disabled or not accelerator.is_main_process:
        return None
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or stage,
        config=vars(args),
    )
    print(f"[wandb] project={args.wandb_project}  run={args.wandb_run_name or stage}")
    return wandb


def _wandb_log(wb, data: dict, step: Optional[int] = None):
    if wb is None:
        return
    if step is None:
        wb.log(data)
    else:
        wb.log(data, step=step)


def _wandb_finish(wb):
    if wb is None:
        return
    wb.finish()


# =============================================================================
# Linear Probing helpers (correction-only mode)
# =============================================================================
def _apply_lp_setup(teacher_unwrapped, ckpt_path: str, correction_dropout: float,
                    logger=print):
    """LP 진입 시 1회 호출.
        1) best ckpt 를 로드 (strict=True — 아키텍처 mismatch 는 명시적 에러).
        2) 모든 파라미터 freeze → correction_head + beta 만 unfreeze.
        3) correction_head 내부 nn.Dropout.p 를 correction_dropout 으로 override (>0 일 때).
    반환: (unfrozen_names, dropout_reset_count).
    """
    if not ckpt_path:
        raise ValueError("--lp_only_correction 요구: --lp_ckpt <path> 를 지정해 주세요.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"--lp_ckpt not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    if "model" not in state:
        raise KeyError(f"ckpt 에 'model' key 가 없음: {ckpt_path}")
    teacher_unwrapped.load_state_dict(state["model"], strict=True)
    logger(f"[teacher LP] loaded ckpt from {ckpt_path}  (saved epoch={state.get('epoch','?')}, "
           f"metric={state.get('metric','?')})")

    per = teacher_unwrapped.perceiver
    if not hasattr(per, "correction_head") or not hasattr(per, "beta"):
        raise RuntimeError("LP 는 residual fusion (correction_head + beta) 구조 전제. "
                           f"perceiver={type(per).__name__} 에 필요 attr 없음.")

    for p in teacher_unwrapped.parameters():
        p.requires_grad = False

    unfrozen = []
    for name, p in per.correction_head.named_parameters():
        p.requires_grad = True
        unfrozen.append(f"perceiver.correction_head.{name}")
    per.beta.requires_grad = True
    unfrozen.append("perceiver.beta")

    n_dropout = 0
    if correction_dropout > 0.0:
        for m in per.correction_head.modules():
            if isinstance(m, nn.Dropout):
                m.p = float(correction_dropout)
                n_dropout += 1
    return unfrozen, n_dropout


# =============================================================================
# Teacher training
# =============================================================================
def train_teacher(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    accelerator.print(f"[teacher] device={device}  procs={accelerator.num_processes}")

    if accelerator.is_main_process:
        # exist_ok=False: 이전 run 결과 실수로 덮어쓰는 걸 방지
        os.makedirs(args.ckpt_dir, exist_ok=False)
        accelerator.print(f"[teacher] run dir → {args.ckpt_dir}")

    processor = AutoImageProcessor.from_pretrained(args.cxr_model_name)

    # Pathology labels: schema 성격이라 CLI 대신 data_processing.py의 상수 참조.
    # Student 파이프라인과 통일. 변경 시 DEFAULT_PATHOLOGY_LABELS만 수정.
    pathology_labels_tuple = DEFAULT_PATHOLOGY_LABELS
    if pathology_labels_tuple[0] != args.label_col:
        raise ValueError(f"DEFAULT_PATHOLOGY_LABELS[0]={pathology_labels_tuple[0]!r} != "
                         f"label_col={args.label_col!r} (data_processing.py에서 순서 확인)")

    # Perceiver 모드 dispatch
    perceiver_type = args.perceiver_type   # "legacy" | "single" | "dual" | "dual_patch"
    is_pathology            = (perceiver_type == "single")
    is_dual_pathology       = (perceiver_type == "dual")
    is_patch_dual_pathology = (perceiver_type == "dual_patch")
    is_query_based          = is_pathology or is_dual_pathology or is_patch_dual_pathology
    # dual과 dual_patch는 loss/evaluator/train step에서 동일하게 작동 (3-branch img/ts/fus 출력)
    uses_dual_loss          = is_dual_pathology or is_patch_dual_pathology
    accelerator.print(f"[teacher] perceiver_type = {perceiver_type}")

    cfg = AnchorConfig(
        final_df_path=args.final_df_path,
        static_path=args.static_path,
        meta_path=args.meta_path,
        label_col=args.label_col,
        n_timesteps=args.n_timesteps,
        split_seed=args.split_seed,
        # Aligned split 사용 — val_size/test_size는 pretrained head의 70/15/15 상속
        pathology_labels=pathology_labels_tuple,
    )
    bundle = build_datasets(cfg, image_processor=processor, include_cxr=True)
    ds = bundle["datasets"]

    train_loader = _make_loader(ds["train"], args.batch_size, True, args.num_workers, "teacher")
    val_loader   = _make_loader(ds["val"],   args.batch_size, False, args.num_workers, "teacher")
    test_loader  = _make_loader(ds["test"],  args.batch_size, False, args.num_workers, "teacher")
    # 매 epoch train subset(deterministic 앞쪽 N배치) gap/AUROC 계산용 loader
    train_eval_loader = None
    if args.eval_train_batches > 0:
        from torch.utils.data import Subset
        n_eval_samples = min(args.eval_train_batches * args.batch_size, len(ds["train"]))
        train_eval_ds = Subset(ds["train"], list(range(n_eval_samples)))
        train_eval_loader = _make_loader(train_eval_ds, args.batch_size, False,
                                         args.num_workers, "teacher")
        accelerator.print(f"[teacher] train_eval subset: n={n_eval_samples} "
                          f"({args.eval_train_batches} batches × {args.batch_size})")

    meta = bundle["meta"]
    backbone = load_duett_backbone(
        ckpt_path=args.duett_ckpt,
        d_static_num=int(meta["D_STATIC"]),
        d_time_series_num=len(bundle["ts_vars"]),
        n_timesteps=args.n_timesteps,
        freeze=args.freeze_duett,
        aug_noise=args.aug_noise,
        aug_mask=args.aug_mask,
        transformer_dropout=args.transformer_dropout,
    )
    # 새 dual은 CLS만 필요 (patches 미사용). single(=PathologyPerceiver), aux_cxr, 그리고
    # dual_patch(=PatchDualPathologyPerceiver)는 patches 필요.
    cxr_return_patches = args.use_aux_cxr or is_pathology or is_patch_dual_pathology
    cxr_enc = CXREncoder(model_name=args.cxr_model_name,
                          freeze=not args.unfreeze_cxr,
                          return_patches=cxr_return_patches)

    if is_dual_pathology:
        if DualPathologyPerceiver is None:
            raise ImportError("DualPathologyPerceiver is commented out in models/main_architecture_duett.py. "
                              "Un-comment the class to use --perceiver_type dual.")
        perceiver = DualPathologyPerceiver(
            n_pathologies=len(pathology_labels_tuple),
            d_ts=backbone.d_representation,
            d_latent=args.d_latent,
            n_heads=args.n_perceiver_heads,
            dropout=args.perceiver_dropout,
        )
        accelerator.print(f"[teacher] DualPathologyPerceiver: K={len(pathology_labels_tuple)}  "
                          f"labels={pathology_labels_tuple}")
    elif is_patch_dual_pathology:
        perceiver = PatchDualPathologyPerceiver(
            n_pathologies=len(pathology_labels_tuple),
            d_ts=backbone.d_representation,
            d_latent=args.d_latent,
            n_heads=args.n_perceiver_heads,
            dropout=args.perceiver_dropout,
            n_timesteps=args.n_timesteps,
            d_event_embedding=backbone.d_embedding,
        )
        accelerator.print(f"[teacher] PatchDualPathologyPerceiver: K={len(pathology_labels_tuple)}  "
                          f"labels={pathology_labels_tuple}  (patches → cross-attn)")
    elif is_pathology:
        if PathologyPerceiver is None:
            raise ImportError("PathologyPerceiver is commented out in models/main_architecture_duett.py. "
                              "Un-comment the class to use --perceiver_type single.")
        perceiver = PathologyPerceiver(
            n_pathologies=len(pathology_labels_tuple),
            d_ts=backbone.d_representation,
            d_latent=args.d_latent,
            n_heads=args.n_perceiver_heads,
            dropout=args.perceiver_dropout,
        )
        accelerator.print(f"[teacher] PathologyPerceiver: K={len(pathology_labels_tuple)}  "
                          f"labels={pathology_labels_tuple}")
    else:
        if TemporalPerceiver is None:
            raise ImportError("TemporalPerceiver is commented out in models/main_architecture_duett.py. "
                              "Un-comment the class to use --perceiver_type legacy.")
        perceiver = TemporalPerceiver(
            d_ts=backbone.d_representation,
            d_img=cxr_enc.d_out,
            d_latent=args.d_latent,
            n_latents=args.n_latents,
            n_layers=args.n_perceiver_layers,
            n_heads=args.n_perceiver_heads,
            dropout=args.perceiver_dropout,
        )

    teacher = TeacherModel(backbone, cxr_enc, perceiver,
                           head_hidden=args.head_hidden,
                           head_dropout=args.head_dropout,
                           cxr_return_patches=cxr_return_patches,
                           d_img=cxr_enc.d_out,
                           use_aux_cxr=args.use_aux_cxr,
                           pathology_mode=is_pathology,
                           dual_pathology_mode=is_dual_pathology,
                           patch_dual_pathology_mode=is_patch_dual_pathology,
                           pretrained_cxr_head_ckpt=(args.pretrained_cxr_head_ckpt
                                                     if is_dual_pathology else None),
                           pathology_labels=(pathology_labels_tuple
                                             if is_dual_pathology else None))
    teacher.to(device)

    # ─── Correction-only Linear Probing setup (dual_patch 전용) ────────────
    # best ckpt 로드 → correction_head + beta 만 unfreeze → correction_head dropout override.
    # 이후 param_groups 는 이 requires_grad 상태를 기준으로 build 됨.
    if args.lp_only_correction:
        if not is_patch_dual_pathology:
            raise ValueError("--lp_only_correction 은 --perceiver_type dual_patch 에서만 지원됩니다. "
                             f"현재 perceiver_type={args.perceiver_type!r}")
        unfrozen, n_dropout_reset = _apply_lp_setup(
            teacher, args.lp_ckpt, args.lp_correction_dropout,
            logger=accelerator.print,
        )
        accelerator.print(f"[teacher LP] unfrozen params ({len(unfrozen)}): {unfrozen}")
        accelerator.print(f"[teacher LP] correction_head dropout override "
                          f"→ p={args.lp_correction_dropout}  (updated {n_dropout_reset} module(s))")
        accelerator.print(f"[teacher LP] regularizers: beta_l2={args.lp_beta_l2}  "
                          f"corr_l2={args.lp_corr_l2}")

    # Trainable parameter 수 출력
    n_total = sum(p.numel() for p in teacher.parameters())
    n_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    accelerator.print(f"[teacher] params: total={n_total:>12,d}  "
                      f"trainable={n_trainable:>12,d}  "
                      f"({100*n_trainable/n_total:.2f}%)")

    # Backbone(낮은 lr) vs 신규 초기화(base lr) 분리 + AdamW
    param_groups = _make_param_groups(teacher, args)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Loss 구성 — dual / single / legacy
    path_loss_fn = None       # single (PathologyMultiLabelLoss)
    dual_loss_fn = None       # dual   (DualPathologyLoss)
    loss_fn = None            # legacy (BCEWithLogitsLoss)
    if is_query_based:
        label_weights = torch.tensor(
            [float(w) for w in args.label_weights.split(",")], dtype=torch.float32)
        if label_weights.numel() != len(pathology_labels_tuple):
            raise ValueError(f"label_weights len ({label_weights.numel()}) `!= "
                             f"pathology_labels len ({len(pathology_labels_tuple)})")
        # pos_weight는 unimodal(CXR/TS) probe들과 통일하기 위해 사용하지 않음.
        # 파이프라인 전체가 동일 loss config여야 fusion 이득이 순수 intervention 효과로 해석됨.
        accelerator.print(f"[teacher] pathology label_weights = {label_weights.tolist()}  "
                          f"(pos_weight OFF — unimodal probes와 통일)")
        if uses_dual_loss:
            dual_loss_fn = DualPathologyLoss(
                label_weights=label_weights,
                pos_weight=None,
                alpha_img=args.aux_img_alpha,
                alpha_ts=args.aux_ts_alpha,
                alpha_fus=args.aux_fus_alpha,
            ).to(device)
            accelerator.print(f"[teacher] DualPathologyLoss alphas: "
                              f"img={args.aux_img_alpha} ts={args.aux_ts_alpha} fus={args.aux_fus_alpha}")
        else:
            path_loss_fn = PathologyMultiLabelLoss(
                label_weights=label_weights,
                pos_weight=None,
                alpha_stage2=args.aux_stage2_alpha,
                alpha_stage4=args.aux_stage4_alpha,
            ).to(device)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    # Warmup + Cosine LR 스케줄러
    total_steps = _steps_per_epoch(train_loader, args) * args.epochs
    scheduler = _make_scheduler(optimizer, total_steps, args)
    accelerator.print(f"[teacher] total_steps={total_steps}  warmup={args.warmup_steps}  "
                       f"backbone_lr={args.lr * args.backbone_lr_mult:.2e}  head_lr={args.lr:.2e}")

    teacher, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        teacher, optimizer, train_loader, val_loader, test_loader, scheduler)
    if train_eval_loader is not None:
        train_eval_loader = accelerator.prepare(train_eval_loader)

    # dual_patch residual: 학습 시작 시점의 shared query 스냅샷 (query_cos_to_init 계산용).
    # optimizer.step() 호출 전에 detach().clone() 로 고정.
    query_ref = None
    if is_patch_dual_pathology:
        _per = accelerator.unwrap_model(teacher).perceiver
        if hasattr(_per, "pathology_queries"):
            query_ref = _per.pathology_queries.detach().float().clone()
            accelerator.print(f"[teacher] query_ref snapshot: shape={tuple(query_ref.shape)} "
                              f"norm={float(query_ref.norm(dim=-1).mean()):.4f}")

    wb = _wandb_init(args, accelerator, stage="teacher")

    best_auroc = -1.0
    best_ckpt = os.path.join(args.ckpt_dir, "best.pt")
    global_step = 0
    epochs_since_best = 0
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"[teacher ep{epoch}]")
        running_loss = 0.0
        running_aux  = 0.0
        running_s2   = 0.0
        running_s4   = 0.0
        running_img  = 0.0
        running_ts   = 0.0
        running_fus  = 0.0
        n = 0
        for step, batch in enumerate(pbar):
            if uses_dual_loss and args.lp_only_correction:
                out = train_teacher_dual_pathology_lp_batch(
                    batch, teacher, dual_loss_fn, optimizer, device, accelerator,
                    beta_l2=args.lp_beta_l2, corr_l2=args.lp_corr_l2,
                    aux_residual_alpha=args.aux_residual_alpha)
            elif uses_dual_loss:
                out = train_teacher_dual_pathology_batch(
                    batch, teacher, dual_loss_fn, optimizer, device, accelerator,
                    aux_residual_alpha=args.aux_residual_alpha)
            elif is_pathology:
                out = train_teacher_pathology_batch(
                    batch, teacher, path_loss_fn, optimizer, device, accelerator)
            else:
                out = train_teacher_batch(
                    batch, teacher, loss_fn, optimizer, device, accelerator,
                    aux_alpha=(args.aux_cxr_alpha if args.use_aux_cxr else 0.0))
            scheduler.step()
            bs = out["y"].shape[0]
            running_loss += out["loss"] * bs
            n += bs
            global_step += 1
            if uses_dual_loss:
                running_img += out["img_total"] * bs
                running_ts  += out["ts_total"]  * bs
                running_fus += out["fus_total"] * bs
            elif is_pathology:
                running_s2 += out["stage2_total"] * bs
                running_s4 += out["stage4_total"] * bs
            else:
                running_aux += out["aux_loss"] * bs

            if accelerator.is_main_process and step % args.log_every == 0:
                avg = float(running_loss / max(n, 1))
                cur_lr = optimizer.param_groups[-1]["lr"]
                parts = [f"loss={avg:.4f}"]
                log = {
                    "train/loss": avg,
                    "train/lr_head": cur_lr,
                    "train/lr_backbone": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                }
                if uses_dual_loss:
                    avg_img = float(running_img / max(n, 1))
                    avg_ts  = float(running_ts  / max(n, 1))
                    avg_fus = float(running_fus / max(n, 1))
                    parts.append(f"fus={avg_fus:.4f}")
                    parts.append(f"img={avg_img:.4f}")
                    parts.append(f"ts={avg_ts:.4f}")
                    log["train/img_loss"] = avg_img
                    log["train/ts_loss"]  = avg_ts
                    log["train/fus_loss"] = avg_fus
                    if args.aux_residual_alpha > 0.0:
                        aux_res = float(out.get("aux_residual", 0.0))
                        parts.append(f"aux_res={aux_res:.4f}")
                        log["train/aux_residual_loss"] = aux_res
                    # LP regularizer 계측 (마지막 batch 값 그대로 — 스무딩 없이 최신치를 보여 β/correction magnitude 변화 감지)
                    if args.lp_only_correction:
                        reg_b = float(out.get("reg_beta_l2", 0.0))
                        reg_c = float(out.get("reg_corr_l2", 0.0))
                        parts.append(f"reg_β={reg_b:.4f}")
                        parts.append(f"reg_c={reg_c:.4f}")
                        log["train/lp_reg_beta_l2"] = reg_b
                        log["train/lp_reg_corr_l2"] = reg_c
                        unwrapped = accelerator.unwrap_model(teacher)
                        beta_vec = unwrapped.perceiver.beta.detach().float().cpu()
                        log["train/lp_beta_mean_abs"] = float(beta_vec.abs().mean())
                        log["train/lp_beta_max_abs"]  = float(beta_vec.abs().max())
                elif is_pathology:
                    avg_s2 = float(running_s2 / max(n, 1))
                    avg_s4 = float(running_s4 / max(n, 1))
                    parts.append(f"s2={avg_s2:.4f}")
                    parts.append(f"s4={avg_s4:.4f}")
                    log["train/stage2_loss"] = avg_s2
                    log["train/stage4_loss"] = avg_s4
                elif args.use_aux_cxr:
                    avg_aux = float(running_aux / max(n, 1))
                    parts.append(f"aux={avg_aux:.4f}")
                    log["train/aux_loss"] = avg_aux
                parts.append(f"lr={cur_lr:.2e}")
                pbar.set_postfix_str(", ".join(parts))
                _wandb_log(wb, log, step=global_step)
            if args.limit_batches and step + 1 >= args.limit_batches:
                break

        # ─── Validation ────────────────────────────────────────────────
        improved = False
        if uses_dual_loss:
            val_p = evaluate_dual_pathology(teacher, val_loader, device, pathology_labels_tuple,
                                             query_ref=query_ref)
            val_m = {"auroc": val_p["main_auroc"], "auprc": val_p["main_auprc"], "n": val_p["n"]}
            if accelerator.is_main_process:
                print(f"[teacher ep{epoch}] Val main(Edema fusion) "
                      f"AUROC={val_m['auroc']:.4f} AUPRC={val_m['auprc']:.4f} n={val_m['n']}")
                print(format_dual_pathology_gap_table(val_p))
                log_val = {
                    "val/auroc": val_m["auroc"],
                    "val/auprc": val_m["auprc"],
                    "val/epoch": epoch,
                }
                for r in val_p["per_label"]:
                    nm = r["name"]
                    log_val[f"val/{nm}/img_auroc"] = r["img_auroc"]
                    log_val[f"val/{nm}/ts_auroc"]  = r["ts_auroc"]
                    log_val[f"val/{nm}/fus_auroc"] = r["fus_auroc"]
                    log_val[f"val/{nm}/gap_i2f"]   = r["gap_i2f"]
                    log_val[f"val/{nm}/gap_t2f"]   = r["gap_t2f"]
                    log_val[f"val/{nm}/img_auprc"] = r["img_auprc"]
                    log_val[f"val/{nm}/ts_auprc"]  = r["ts_auprc"]
                    log_val[f"val/{nm}/fus_auprc"] = r["fus_auprc"]
                _wandb_log(wb, log_val, step=global_step)
                if val_m["auroc"] > best_auroc:
                    best_auroc = val_m["auroc"]
                    improved = True
                    unwrapped = accelerator.unwrap_model(teacher)
                    _save_ckpt(best_ckpt, unwrapped, optimizer, epoch, val_m, args)
                    print(f"[teacher] saved best ckpt (main AUROC={best_auroc:.4f}) → {best_ckpt}")
                    _wandb_log(wb, {"val/best_auroc": best_auroc}, step=global_step)
        elif is_pathology:
            val_p = evaluate_pathology(teacher, val_loader, device, pathology_labels_tuple)
            val_m = {"auroc": val_p["main_auroc"], "auprc": val_p["main_auprc"], "n": val_p["n"]}
            if accelerator.is_main_process:
                print(f"[teacher ep{epoch}] main(Edema stage4) "
                      f"AUROC={val_m['auroc']:.4f} AUPRC={val_m['auprc']:.4f} n={val_m['n']}")
                print(format_pathology_gap_table(val_p))
                log_val = {
                    "val/auroc": val_m["auroc"],
                    "val/auprc": val_m["auprc"],
                    "val/epoch": epoch,
                }
                for r in val_p["per_label"]:
                    nm = r["name"]
                    log_val[f"val/{nm}/stage2_auroc"] = r["stage2_auroc"]
                    log_val[f"val/{nm}/stage4_auroc"] = r["stage4_auroc"]
                    log_val[f"val/{nm}/gap_auroc"]   = r["gap_auroc"]
                    log_val[f"val/{nm}/stage2_auprc"] = r["stage2_auprc"]
                    log_val[f"val/{nm}/stage4_auprc"] = r["stage4_auprc"]
                    log_val[f"val/{nm}/gap_auprc"]   = r["gap_auprc"]
                _wandb_log(wb, log_val, step=global_step)
                if val_m["auroc"] > best_auroc:
                    best_auroc = val_m["auroc"]
                    improved = True
                    unwrapped = accelerator.unwrap_model(teacher)
                    _save_ckpt(best_ckpt, unwrapped, optimizer, epoch, val_m, args)
                    print(f"[teacher] saved best ckpt (main AUROC={best_auroc:.4f}) → {best_ckpt}")
                    _wandb_log(wb, {"val/best_auroc": best_auroc}, step=global_step)
        else:
            fwd = make_teacher_forward()
            val_m = evaluate_binary(teacher, val_loader, device, fwd)
            val_aux = None
            if args.use_aux_cxr:
                fwd_aux = make_teacher_aux_forward()
                val_aux = evaluate_binary(teacher, val_loader, device, fwd_aux)
            if accelerator.is_main_process:
                print(f"[teacher ep{epoch}] val AUROC={val_m['auroc']:.4f}  "
                      f"AUPRC={val_m['auprc']:.4f}  n={val_m['n']}")
                if val_aux is not None:
                    print(f"[teacher ep{epoch}]  ⤷ aux(CXR-only) AUROC={val_aux['auroc']:.4f}  "
                          f"AUPRC={val_aux['auprc']:.4f}")
                log_val = {
                    "val/auroc": val_m["auroc"], "val/auprc": val_m["auprc"],
                    "val/epoch": epoch,
                }
                if val_aux is not None:
                    log_val["val/aux_auroc"] = val_aux["auroc"]
                    log_val["val/aux_auprc"] = val_aux["auprc"]
                _wandb_log(wb, log_val, step=global_step)
                if val_m["auroc"] > best_auroc:
                    best_auroc = val_m["auroc"]
                    improved = True
                    unwrapped = accelerator.unwrap_model(teacher)
                    _save_ckpt(best_ckpt, unwrapped, optimizer, epoch, val_m, args)
                    print(f"[teacher] saved best ckpt (AUROC={best_auroc:.4f}) → {best_ckpt}")
                    _wandb_log(wb, {"val/best_auroc": best_auroc}, step=global_step)

        # ─── Train-set evaluation (overfit 진단용) ─────────────────────
        if train_eval_loader is not None:
            if uses_dual_loss:
                train_p = evaluate_dual_pathology(teacher, train_eval_loader, device, pathology_labels_tuple,
                                                   query_ref=query_ref)
                if accelerator.is_main_process:
                    print(f"[teacher ep{epoch}] TRAIN main(Edema fusion) "
                          f"AUROC={train_p['main_auroc']:.4f} AUPRC={train_p['main_auprc']:.4f} "
                          f"n={train_p['n']}")
                    print(format_dual_pathology_gap_table(train_p))
                    log_train = {
                        "train_eval/auroc": train_p["main_auroc"],
                        "train_eval/auprc": train_p["main_auprc"],
                        "train_eval/epoch": epoch,
                        "train_eval/main_gap_over_val": train_p["main_auroc"] - val_m["auroc"],
                    }
                    for r in train_p["per_label"]:
                        nm = r["name"]
                        log_train[f"train_eval/{nm}/img_auroc"] = r["img_auroc"]
                        log_train[f"train_eval/{nm}/ts_auroc"]  = r["ts_auroc"]
                        log_train[f"train_eval/{nm}/fus_auroc"] = r["fus_auroc"]
                        log_train[f"train_eval/{nm}/gap_i2f"]   = r["gap_i2f"]
                        log_train[f"train_eval/{nm}/gap_t2f"]   = r["gap_t2f"]
                    _wandb_log(wb, log_train, step=global_step)
            elif is_pathology:
                train_p = evaluate_pathology(teacher, train_eval_loader, device, pathology_labels_tuple)
                if accelerator.is_main_process:
                    print(f"[teacher ep{epoch}] TRAIN main(Edema stage4) "
                          f"AUROC={train_p['main_auroc']:.4f} AUPRC={train_p['main_auprc']:.4f} "
                          f"n={train_p['n']}")
                    print(format_pathology_gap_table(train_p))
                    log_train = {
                        "train_eval/auroc": train_p["main_auroc"],
                        "train_eval/auprc": train_p["main_auprc"],
                        "train_eval/epoch": epoch,
                    }
                    # main task train-val gap = overfit 정량 지표
                    log_train["train_eval/main_gap_over_val"] = train_p["main_auroc"] - val_m["auroc"]
                    for r in train_p["per_label"]:
                        nm = r["name"]
                        log_train[f"train_eval/{nm}/stage2_auroc"] = r["stage2_auroc"]
                        log_train[f"train_eval/{nm}/stage4_auroc"] = r["stage4_auroc"]
                        log_train[f"train_eval/{nm}/gap_auroc"]   = r["gap_auroc"]
                        log_train[f"train_eval/{nm}/stage2_auprc"] = r["stage2_auprc"]
                        log_train[f"train_eval/{nm}/stage4_auprc"] = r["stage4_auprc"]
                        log_train[f"train_eval/{nm}/gap_auprc"]   = r["gap_auprc"]
                    _wandb_log(wb, log_train, step=global_step)
            else:
                fwd_te = make_teacher_forward()
                train_m = evaluate_binary(teacher, train_eval_loader, device, fwd_te)
                if accelerator.is_main_process:
                    print(f"[teacher ep{epoch}] TRAIN AUROC={train_m['auroc']:.4f}  "
                          f"AUPRC={train_m['auprc']:.4f}  n={train_m['n']}  "
                          f"(train-val gap = {train_m['auroc'] - val_m['auroc']:+.4f})")
                    _wandb_log(wb, {
                        "train_eval/auroc": train_m["auroc"],
                        "train_eval/auprc": train_m["auprc"],
                        "train_eval/main_gap_over_val": train_m["auroc"] - val_m["auroc"],
                    }, step=global_step)

        # ─── Gradient flow diagnostics (dual_patch, 3 epoch 마다, main rank only) ──
        if (is_patch_dual_pathology and dual_loss_fn is not None
                and accelerator.is_main_process
                and args.grad_diag_every > 0
                and (epoch % args.grad_diag_every == 0)):
            try:
                from analysis.grad_flow_diagnostics import (
                    run_dual_gradient_diagnostics, format_gradient_diagnostics,
                    gradient_diagnostics_to_log_dict,
                )
                diag_loader = train_eval_loader if train_eval_loader is not None else val_loader
                accelerator.print(f"[teacher ep{epoch}] running gradient flow diagnostics "
                                  f"(max_batches={args.grad_diag_batches})")
                grad_report = run_dual_gradient_diagnostics(
                    accelerator.unwrap_model(teacher), diag_loader, dual_loss_fn, device,
                    max_batches=args.grad_diag_batches,
                    label_names=list(pathology_labels_tuple),
                )
                print(format_gradient_diagnostics(grad_report))
                _wandb_log(wb, gradient_diagnostics_to_log_dict(grad_report), step=global_step)
            except Exception as exc:  # 진단은 학습에 영향 없어야 함
                accelerator.print(f"[teacher] grad_diag skipped: {exc}")

        # Early stopping (모든 rank에서 동일하게 판단해야 hang 안 남)
        improved_t = torch.tensor(int(improved), device=device)
        if accelerator.num_processes > 1:
            from accelerate.utils import broadcast
            improved_t = broadcast(improved_t, from_process=0)
        epochs_since_best = 0 if improved_t.item() else epochs_since_best + 1
        if args.patience > 0 and epochs_since_best >= args.patience:
            accelerator.print(f"[teacher] early stop — val AUROC {args.patience} epoch 개선 실패"
                               f"(best={best_auroc:.4f})")
            break

    # ─── Final test (load best) ────────────────────────────────────
    if accelerator.is_main_process and os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        accelerator.unwrap_model(teacher).load_state_dict(state["model"])
    if uses_dual_loss:
        test_p = evaluate_dual_pathology(teacher, test_loader, device, pathology_labels_tuple,
                                          query_ref=query_ref)
        if accelerator.is_main_process:
            print(f"[teacher] test main(Edema fusion) AUROC={test_p['main_auroc']:.4f}  "
                  f"AUPRC={test_p['main_auprc']:.4f}")
            print(format_dual_pathology_gap_table(test_p))
            test_log = {"test/auroc": test_p["main_auroc"], "test/auprc": test_p["main_auprc"]}
            for r in test_p["per_label"]:
                nm = r["name"]
                test_log[f"test/{nm}/img_auroc"] = r["img_auroc"]
                test_log[f"test/{nm}/ts_auroc"]  = r["ts_auroc"]
                test_log[f"test/{nm}/fus_auroc"] = r["fus_auroc"]
                test_log[f"test/{nm}/gap_i2f"]   = r["gap_i2f"]
                test_log[f"test/{nm}/gap_t2f"]   = r["gap_t2f"]
                test_log[f"test/{nm}/img_auprc"] = r["img_auprc"]
                test_log[f"test/{nm}/ts_auprc"]  = r["ts_auprc"]
                test_log[f"test/{nm}/fus_auprc"] = r["fus_auprc"]
            _wandb_log(wb, test_log, step=global_step)
    elif is_pathology:
        test_p = evaluate_pathology(teacher, test_loader, device, pathology_labels_tuple)
        if accelerator.is_main_process:
            print(f"[teacher] test main(Edema stage4) AUROC={test_p['main_auroc']:.4f}  "
                  f"AUPRC={test_p['main_auprc']:.4f}")
            print(format_pathology_gap_table(test_p))
            test_log = {"test/auroc": test_p["main_auroc"], "test/auprc": test_p["main_auprc"]}
            for r in test_p["per_label"]:
                nm = r["name"]
                test_log[f"test/{nm}/stage2_auroc"] = r["stage2_auroc"]
                test_log[f"test/{nm}/stage4_auroc"] = r["stage4_auroc"]
                test_log[f"test/{nm}/gap_auroc"]   = r["gap_auroc"]
                test_log[f"test/{nm}/stage2_auprc"] = r["stage2_auprc"]
                test_log[f"test/{nm}/stage4_auprc"] = r["stage4_auprc"]
                test_log[f"test/{nm}/gap_auprc"]   = r["gap_auprc"]
            _wandb_log(wb, test_log, step=global_step)
    else:
        fwd = make_teacher_forward()
        test_m = evaluate_binary(teacher, test_loader, device, fwd)
        if accelerator.is_main_process:
            print(f"[teacher] test AUROC={test_m['auroc']:.4f}  AUPRC={test_m['auprc']:.4f}")
            _wandb_log(wb, {"test/auroc": test_m["auroc"], "test/auprc": test_m["auprc"]},
                       step=global_step)
    _wandb_finish(wb)


# =============================================================================
# Teacher reconstruction from checkpoint (for student KD)
# =============================================================================
def _build_teacher_from_ckpt(teacher_state, meta, ts_vars_len,
                              duett_ckpt, n_timesteps, cxr_model_name_fallback):
    """Checkpoint에 저장된 teacher `args`로 dual-mode TeacherModel을 복원.

    Teacher 하이퍼(d_latent/head_*/pathology_labels 등)는 모두 checkpoint에서 읽어와
    student CLI에는 teacher 아키텍처 관련 인자를 반복 지정할 필요가 없다.
    """
    t_args = teacher_state["args"]
    if t_args.get("perceiver_type") != "dual":
        raise NotImplementedError(
            f"student KD는 현재 dual perceiver teacher만 지원합니다. "
            f"checkpoint의 perceiver_type={t_args.get('perceiver_type')!r}")

    pathology_labels = tuple(s.strip() for s in t_args["pathology_labels"].split(","))

    # DuETT backbone: 구조만 잡고 weight는 teacher state_dict가 덮어씀.
    backbone = load_duett_backbone(
        ckpt_path=duett_ckpt,
        d_static_num=int(meta["D_STATIC"]),
        d_time_series_num=ts_vars_len,
        n_timesteps=n_timesteps,
        freeze=True,
        aug_noise=0.0, aug_mask=0.0, transformer_dropout=0.0,
    )
    cxr_enc = CXREncoder(
        model_name=t_args.get("cxr_model_name", cxr_model_name_fallback),
        freeze=True,
        return_patches=False,   # 새 dual은 CLS만 사용
    )
    perceiver = DualPathologyPerceiver(
        n_pathologies=len(pathology_labels),
        d_ts=backbone.d_representation,
        d_latent=int(t_args["d_latent"]),
        n_heads=int(t_args["n_perceiver_heads"]),
        dropout=float(t_args["perceiver_dropout"]),
    )
    teacher = TeacherModel(
        backbone, cxr_enc, perceiver,
        head_hidden=int(t_args["head_hidden"]),
        head_dropout=float(t_args["head_dropout"]),
        cxr_return_patches=False,
        d_img=cxr_enc.d_out,
        use_aux_cxr=False,
        pathology_mode=False,
        dual_pathology_mode=True,
        pretrained_cxr_head_ckpt=t_args["pretrained_cxr_head_ckpt"],
        pathology_labels=pathology_labels,
    )
    teacher.load_state_dict(teacher_state["model"])
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


# =============================================================================
# Student KD training
# =============================================================================
def train_student(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    accelerator.print(f"[student] device={device}  procs={accelerator.num_processes}")

    if accelerator.is_main_process:
        os.makedirs(args.ckpt_dir, exist_ok=False)
        accelerator.print(f"[student] run dir → {args.ckpt_dir}")

    processor = AutoImageProcessor.from_pretrained(args.cxr_model_name)

    cfg = AnchorConfig(
        final_df_path=args.final_df_path,
        static_path=args.static_path,
        meta_path=args.meta_path,
        label_col=args.label_col,
        n_timesteps=args.n_timesteps,
        split_seed=args.split_seed,
        # Aligned split 사용 — val_size/test_size는 pretrained head의 70/15/15 상속
    )

    bundle = build_datasets(cfg, image_processor=processor, include_cxr=True)
    meta = bundle["meta"]

    # --- Teacher (frozen) ---
    # Checkpoint에 저장된 teacher args로 아키텍처를 그대로 복원.
    teacher_state = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher = _build_teacher_from_ckpt(
        teacher_state, meta,
        ts_vars_len=len(bundle["ts_vars"]),
        duett_ckpt=args.duett_ckpt,
        n_timesteps=args.n_timesteps,
        cxr_model_name_fallback=args.cxr_model_name,
    )
    teacher.to(device)
    accelerator.print(f"[student] loaded dual teacher from {args.teacher_ckpt}")

    # --- Student ---
    stu_backbone = load_duett_backbone(
        ckpt_path=args.duett_ckpt,
        d_static_num=int(meta["D_STATIC"]),
        d_time_series_num=len(bundle["ts_vars"]),
        n_timesteps=args.n_timesteps,
        freeze=False,
        aug_noise=args.aug_noise,
        aug_mask=args.aug_mask,
        transformer_dropout=args.transformer_dropout,
    )
    student = StudentModel(stu_backbone, pool=args.student_pool,
                           head_hidden=args.head_hidden,
                           head_dropout=args.head_dropout)
    student.to(device)

    # Trainable parameter 수 출력
    n_total = sum(p.numel() for p in student.parameters())
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    accelerator.print(f"[student] params: total={n_total:>12,d}  "
                      f"trainable={n_trainable:>12,d}  "
                      f"({100*n_trainable/n_total:.2f}%)")

    train_loader = _make_loader(bundle["datasets"]["train"], args.batch_size, True,
                                args.num_workers, "teacher")
    val_loader   = _make_loader(bundle["datasets"]["val"],   args.batch_size, False,
                                args.num_workers, "teacher")
    test_loader  = _make_loader(bundle["datasets"]["test"],  args.batch_size, False,
                                args.num_workers, "teacher")

    kd_loss = StudentKDLoss(kd_name=args.kd_name, kd_T=args.kd_T,
                             kd_alpha=args.kd_alpha, pos_weight=None).to(device)

    # Optimizer
    param_groups = _make_param_groups(student, args)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Warmup + Cosine LR 스케줄러
    total_steps = _steps_per_epoch(train_loader, args) * args.epochs
    scheduler = _make_scheduler(optimizer, total_steps, args)
    accelerator.print(f"[student] total_steps={total_steps}  warmup={args.warmup_steps}  "
                       f"backbone_lr={args.lr * args.backbone_lr_mult:.2e}  head_lr={args.lr:.2e}")

    student, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        student, optimizer, train_loader, val_loader, test_loader, scheduler)

    wb = _wandb_init(args, accelerator, stage="student")

    best_auroc = -1.0
    best_ckpt = os.path.join(args.ckpt_dir, "best.pt")
    global_step = 0
    epochs_since_best = 0
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"[student ep{epoch}]")
        running = {"loss": 0.0, "bce": 0.0, "kd": 0.0, "n": 0}
        for step, batch in enumerate(pbar):
            out = train_student_batch(batch, batch, student, teacher, kd_loss, optimizer, device, accelerator)
            scheduler.step()
            bs = out["y"].shape[0]
            running["loss"] += out["loss"] * bs
            running["bce"] += out["bce"] * bs
            running["kd"] += out["kd"] * bs
            running["n"] += bs
            global_step += 1
            if accelerator.is_main_process and step % args.log_every == 0:
                nn_ = max(running["n"], 1)
                cur_lr = optimizer.param_groups[-1]["lr"]
                pbar.set_postfix(
                    loss=f"{running['loss']/nn_:.4f}",
                    bce=f"{running['bce']/nn_:.4f}",
                    kd=f"{running['kd']/nn_:.4f}",
                    lr=f"{cur_lr:.2e}",
                )
                _wandb_log(wb, {
                    "train/loss": running["loss"] / nn_,
                    "train/bce":  running["bce"]  / nn_,
                    "train/kd":   running["kd"]   / nn_,
                    "train/lr_head": cur_lr,
                    "train/lr_backbone": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                }, step=global_step)
            if args.limit_batches and step + 1 >= args.limit_batches:
                break

        fwd = make_student_forward()
        val_m = evaluate_binary(student, val_loader, device, fwd)
        improved = False
        if accelerator.is_main_process:
            print(f"[student ep{epoch}] val AUROC={val_m['auroc']:.4f}  "
                  f"AUPRC={val_m['auprc']:.4f}  n={val_m['n']}")
            _wandb_log(wb, {
                "val/auroc": val_m["auroc"], "val/auprc": val_m["auprc"],
                "val/epoch": epoch,
            }, step=global_step)
            if val_m["auroc"] > best_auroc:
                best_auroc = val_m["auroc"]
                improved = True
                unwrapped = accelerator.unwrap_model(student)
                _save_ckpt(best_ckpt, unwrapped, optimizer, epoch, val_m, args)
                print(f"[student] saved best ckpt (AUROC={best_auroc:.4f}) → {best_ckpt}")
                _wandb_log(wb, {"val/best_auroc": best_auroc}, step=global_step)

        # Early stopping (모든 rank에서 동일하게 판단해야 hang 안 남)
        improved_t = torch.tensor(int(improved), device=device)
        if accelerator.num_processes > 1:
            from accelerate.utils import broadcast
            improved_t = broadcast(improved_t, from_process=0)
        epochs_since_best = 0 if improved_t.item() else epochs_since_best + 1
        if args.patience > 0 and epochs_since_best >= args.patience:
            accelerator.print(f"[student] early stop — val AUROC {args.patience} epoch 개선 실패"
                               f"(best={best_auroc:.4f})")
            break

    if accelerator.is_main_process and os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        accelerator.unwrap_model(student).load_state_dict(state["model"])
    fwd = make_student_forward()
    test_m = evaluate_binary(student, test_loader, device, fwd)
    if accelerator.is_main_process:
        print(f"[student] test AUROC={test_m['auroc']:.4f}  AUPRC={test_m['auprc']:.4f}")
        _wandb_log(wb, {"test/auroc": test_m["auroc"], "test/auprc": test_m["auprc"]},
                   step=global_step)
    _wandb_finish(wb)
