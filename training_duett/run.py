"""Argparse for the DuETT KD teacher/student entry points."""
from __future__ import annotations

import argparse
import os
from datetime import datetime


REPO_DEFAULTS = {
    "final_df_path": "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/subject_data/mimic_3_1/final_df_20260713",
    "static_path": "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/data/full_data/mimic_3_1/static_full.ftr",
    "duett_ckpt": "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/duett/checkpoints/mimic_ssl/n24_s12/pretrain-epoch=130-val_loss=0.2717.ckpt",
    "pretrained_cxr_head_ckpt": "/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/checkpoints/cxr_linear_head/raddino_linear_head_20260715_053311.pt",
}


# Diff-tag에서 제외 — 실험 결과에 영향 없는 경로/로깅/런타임 인자
DEFAULT_TAG_EXCLUDE = frozenset({
    "final_df_path", "static_path", "meta_path", "duett_ckpt",
    "cxr_model_name", "ckpt_dir", "teacher_ckpt", "pretrained_cxr_head_ckpt",
    "wandb_project", "wandb_run_name", "wandb_disabled",
    "num_workers", "log_every", "mixed_precision", "limit_batches",
})


def make_diff_tag(parser: argparse.ArgumentParser, args: argparse.Namespace, exclude=DEFAULT_TAG_EXCLUDE) -> str:
    defaults = {a.dest: a.default for a in parser._actions
                if a.dest != "help" and a.dest not in exclude}
    diff = {k: v for k, v in vars(args).items()
            if k in defaults and v != defaults[k]}
    if not diff:
        return "default"
    return "_".join(f"{k}={v}" for k, v in sorted(diff.items()))


def _finalize_ckpt_dir(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """{ckpt_dir}/{timestamp}_{diff_tag}/ 형태로 유일 경로 확정."""
    tag = make_diff_tag(parser, args)
    args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + tag
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.run_id)
    return args


def _attach_meta_path(args: argparse.Namespace) -> argparse.Namespace:
    args.meta_path = os.path.join(os.path.dirname(args.duett_ckpt), "meta_with_stats.pkl")
    return args


def _add_common(p: argparse.ArgumentParser):

    # wandb
    p.add_argument("--wandb_project",  type=str, default="Teacher Modal Upgrade")
    p.add_argument("--wandb_run_name", type=str, default="correction_head_Q:T_K,V:I", help="비우면 stage명(teacher/student) 사용")
    p.add_argument("--wandb_disabled", action="store_true")


    # Data
    p.add_argument("--final_df_path", type=str, default=REPO_DEFAULTS["final_df_path"])
    p.add_argument("--static_path",   type=str, default=REPO_DEFAULTS["static_path"])
    p.add_argument("--label_col",     type=str, default="label_edema")
    p.add_argument("--n_timesteps",   type=int, default=24)
    p.add_argument("--split_seed",    type=int, default=42)

    # DuETT backbone
    p.add_argument("--duett_ckpt",    type=str, default=REPO_DEFAULTS["duett_ckpt"])
    p.add_argument("--aug_noise",     type=float, default=0.0)
    p.add_argument("--aug_mask",      type=float, default=0.0)
    p.add_argument("--transformer_dropout", type=float, default=0.0)

    # CXR encoder
    p.add_argument("--cxr_model_name", type=str, default="microsoft/rad-dino")

    # Perceiver + head
    p.add_argument("--d_latent",           type=int, default=256)
    p.add_argument("--n_latents",          type=int, default=16)
    p.add_argument("--n_perceiver_layers", type=int, default=1)
    p.add_argument("--n_perceiver_heads",  type=int, default=4)
    p.add_argument("--perceiver_dropout",  type=float, default=0.2)
    p.add_argument("--head_hidden",        type=int, default=128)
    p.add_argument("--head_dropout",       type=float, default=0.2)

    # Optim / training
    p.add_argument("--lr",             type=float, default=8e-5,
                   help="head/perceiver/proj 등 신규 파라미터의 base lr")
    p.add_argument("--backbone_lr_mult", type=float, default=0.2,
                   help="pretrained backbone(DuETT/CXR) lr = args.lr × 이 값")
    p.add_argument("--correction_lr_mult", type=float, default=1.0,
                   help="dual_patch residual mode: correction_head lr = args.lr × 이 값. "
                        "원점 baseline에서는 다른 신규 모듈과 같은 1.0을 사용.")
    p.add_argument("--query_lr_mult",   type=float, default=0.2,
                   help="dual_patch residual mode: shared pathology_queries lr = args.lr × 이 값. "
                        "queries 는 image/ts branches 와 attention 공유하므로 천천히 이동.")

    # Gradient flow diagnostics (dual_patch 전용)
    p.add_argument("--grad_diag_every", type=int, default=3,
                   help="dual_patch: N epoch 마다 gradient flow 진단 실행. 0 이면 비활성.")
    p.add_argument("--grad_diag_batches", type=int, default=8,
                   help="grad_diag 시 사용할 batch 수 (train_eval_loader 또는 val_loader 앞쪽).")
    p.add_argument("--weight_decay",   type=float, default=5e-2)
    p.add_argument("--batch_size",     type=int, default=128)
    p.add_argument("--num_workers",    type=int, default=8)
    p.add_argument("--epochs",         type=int, default=30)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--log_every",      type=int, default=20)
    p.add_argument("--limit_batches",  type=int, default=0,
                   help="if >0, cap steps/epoch (dev/dry-run)")

    # LR scheduler (linear warmup + cosine anneal)
    p.add_argument("--warmup_steps",   type=int, default=300,
                   help="linear warmup 스텝 수")
    p.add_argument("--min_lr_ratio",   type=float, default=0.01,
                   help="cosine anneal 최종 lr = args.lr × 이 값")

    # Early stopping
    p.add_argument("--patience",       type=int, default=5,
                   help="val AUROC 개선 없는 epoch가 이 값 이상이면 조기 종료. 0이면 비활성")

    # Auxiliary CXR-only head
    p.add_argument("--use_aux_cxr",    action="store_true",
                   help="CXR-only auxiliary head 활성화 — img_proj에 직접 gradient")
    p.add_argument("--aux_cxr_alpha",  type=float, default=0.0,
                   help="total_loss = main_bce + aux_cxr_alpha × aux_bce")

    # Query-based Perceiver (multi-label query-based fusion)
    p.add_argument("--perceiver_type", type=str, default="legacy",
                   choices=["legacy", "single", "dual", "dual_patch"],
                   help="legacy: TemporalPerceiver / single: PathologyPerceiver / "
                        "dual: DualPathologyPerceiver (frozen pretrained CXR head + residual fusion) / "
                        "dual_patch: PatchDualPathologyPerceiver (patches × pathology query "
                        "cross-attention; pretrained head 불필요)")

    # Single (PathologyPerceiver) 전용 loss alpha
    p.add_argument("--aux_stage2_alpha", type=float, default=1.0,
                   help="[single] image-only aux (stage2) 전체 가중치")
    p.add_argument("--aux_stage4_alpha", type=float, default=0.5,
                   help="[single] multimodal aux (stage4) 전체 가중치 — main task 포함")
    # Dual (DualPathologyPerceiver) 전용 3-branch loss alpha
    # 새 dual: image branch가 frozen pretrained head → 학습 대상 아님 → 기본 alpha_img=0.0 (logging only).
    p.add_argument("--aux_img_alpha",   type=float, default=0.5,
                   help="[dual] image branch loss 가중치 (frozen pretrained head라 gradient 0; logging 유지)")
    p.add_argument("--aux_ts_alpha",    type=float, default=0.5,
                   help="[dual] TS branch loss 가중치 (독립 gradient path 확보 위해 0.5+)")
    p.add_argument("--aux_fus_alpha",   type=float, default=1.0,
                   help="[dual] fusion branch loss 가중치 (main task)")
    p.add_argument("--aux_residual_alpha", type=float, default=0.0,
                   help="[dual_patch] Correction 이 image anchor + correction 을 통과한 확률을 "
                        "smoothed label 에 맞추도록 하는 KL divergence aux loss weight. 0 이면 비활성. "
                        "gradient 는 scaled_correction 만 통과 (img_logit 은 detach). "
                        "fusion loss 만으로 correction 방향 학습이 실패할 때 사용.")
    p.add_argument("--pretrained_cxr_head_ckpt", type=str,
                   default=REPO_DEFAULTS["pretrained_cxr_head_ckpt"],
                   help="[dual] 240k CXR로 학습된 frozen linear head checkpoint. "
                        "TeacherModel이 로드해 CLS→N-label logit→pathology_labels 순으로 slice.")

    # Train-time monitoring (overfitting 진단용)
    p.add_argument("--eval_train_batches", type=int, default=0,
                   help="매 epoch 끝에 train 앞쪽 N 배치(deterministic subset)에서 gap 표/AUROC "
                        "계산. 0=비활성. 100 (약 12.8k 샘플)이면 val보다 넓어 AUROC 통계 충분.")

    # Correction-only Linear Probing (dual_patch 전용, over-fit 진단/치료 stage)
    # best ckpt 를 불러와 correction_head + beta 만 재학습. 다른 모든 파라미터는 freeze +
    # eval mode 로 두어 backbone/perceiver 내부 dropout 이 재현성 있는 forward 를 만들도록 함.
    p.add_argument("--lp_only_correction", action="store_true",
                   help="linear probing: correction_head + beta 만 학습, 나머지 전부 freeze+eval. "
                        "dual_patch teacher 에서만 지원. --lp_ckpt 필수.")
    p.add_argument("--lp_ckpt", type=str, default="",
                   help="LP 시작점 checkpoint 경로 (train_teacher 가 저장한 best.pt).")
    p.add_argument("--lp_beta_l2", type=float, default=1e-3,
                   help="beta L2 penalty coefficient. loss += lp_beta_l2 * mean(beta**2).")
    p.add_argument("--lp_corr_l2", type=float, default=1e-2,
                   help="scaled_correction L2 penalty coefficient. "
                        "loss += lp_corr_l2 * mean((beta*ts_correction)**2) (batch mean).")
    p.add_argument("--lp_correction_dropout", type=float, default=0.3,
                   help="LP 시 correction_head 내부 nn.Dropout.p 를 이 값으로 override. "
                        "0 이면 override 하지 않음.")

    


def parse_teacher_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DuETT KD teacher training")
    _add_common(p)
    p.add_argument("--freeze_duett",   action="store_true",
                   help="freeze DuETT backbone during teacher training")
    p.add_argument("--unfreeze_cxr",   action="store_true",
                   help="unfreeze RAD-DINO (default: frozen)")
    p.add_argument("--ckpt_dir",       type=str,
                   default="/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/checkpoints_duett/teacher")
    args = _attach_meta_path(p.parse_args())
    return _finalize_ckpt_dir(p, args)


def parse_student_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DuETT KD student training")
    _add_common(p)
    p.add_argument("--teacher_ckpt",   type=str, required=True)
    p.add_argument("--student_pool",   type=str, default="mean",
                   choices=["mean", "rep_token"])
    p.add_argument("--kd_name",        type=str, default="vanilla_kl",
                   choices=["vanilla_kl"])
    p.add_argument("--kd_T",           type=float, default=4.0)
    p.add_argument("--kd_alpha",       type=float, default=0.5,
                   help="0.0 = KD only, 1.0 = BCE only")
    p.add_argument("--ckpt_dir",       type=str,
                   default="/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/checkpoints_duett/student")
    args = _attach_meta_path(p.parse_args())
    return _finalize_ckpt_dir(p, args)
