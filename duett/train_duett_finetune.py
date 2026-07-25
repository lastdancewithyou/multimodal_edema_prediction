# /home/DAHS1/gangmin/my_research/clinical_multimodal_learning/duett/train_duett_finetune.py
"""DuETT fine-tune + test for MIMIC mortality. Run inside tmux."""
import os, sys, argparse, warnings

warnings.filterwarnings("ignore", message=".*rotary embedding dimension.*")
warnings.filterwarnings("ignore", message=".*ambiguous collection.*")
try:
    from loguru import logger as _loguru
    _loguru.disable("x_transformers")
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

import duett
from mimic_dataset import collate_into_seqs, prepare_from_raw, ART_DIR


# ==========================================================
# 콜백 & 유틸
# ==========================================================
class WarmUpCallback(pl.callbacks.Callback):
    def __init__(self, steps=1000, invsqrt=True, decay=None):
        super().__init__()
        self.warmup_steps = steps
        self.decay = decay or steps
        self.state = {'steps': 0, 'base_lr': None}
        self.invsqrt = invsqrt

    def _set_lr(self, opt, lr):
        for pg in opt.param_groups:
            pg['lr'] = lr

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        opt = model.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        if self.state['base_lr'] is None:
            self.state['base_lr'] = opt.param_groups[0]['lr']
        s, base = self.state['steps'], self.state['base_lr']
        if s < self.warmup_steps:
            self._set_lr(opt, s / self.warmup_steps * base)
        elif self.invsqrt:
            self._set_lr(opt, base * (self.decay / (s - self.warmup_steps + self.decay)) ** 0.5)
        self.state['steps'] += 1


def average_models(models):
    """top-k 체크포인트 가중치 평균을 첫 모델에 로드해 반환."""
    models = list(models)
    sds = [m.state_dict() for m in models]
    avg = {k: sum(sd[k] for sd in sds) / len(sds) for k in sds[0]}
    models[0].load_state_dict(avg)
    return models[0]


# ==========================================================
# argparse
# ==========================================================
def parse_args():
    ap = argparse.ArgumentParser()
    # 경로
    ap.add_argument('--art_dir',     default=ART_DIR,
                    help='전처리 산출물 경로 (icu_events_raw.ftr, static_full.ftr, meta.pkl)')
    ap.add_argument('--pretrained',  required=True,
                    help='SSL pretrain checkpoint 경로 (.ckpt)')
    ap.add_argument('--ckpt_dir',    default=os.path.join(HERE, 'checkpoints/mimic_ft'),
                    help='base 경로. 실제 저장은 그 아래 n{N_TIMESTEPS}/seed{N}/ 폴더로 자동 분리')
    # 데이터
    ap.add_argument('--n_timesteps', type=int, default=24,
                    help='시간 슬롯 수 = 윈도우 길이(h). SSL pretrain과 반드시 동일')
    ap.add_argument('--split_seed',  type=int, default=42,
                    help='train/val/test split 재현용 (SSL과 동일해야 leakage 없음)')
    # 학습
    ap.add_argument('--batch_size',   type=int, default=256)
    ap.add_argument('--num_workers',  type=int, default=8)
    ap.add_argument('--max_epochs',   type=int, default=40)
    ap.add_argument('--warmup_steps', type=int, default=1000)
    ap.add_argument('--save_top_k',   type=int, default=5,
                    help='top-k checkpoint를 weight averaging해서 test')
    ap.add_argument('--seeds',        type=int, nargs='+',
                    default=[40, 41, 42],
                    help='반복 실행할 seed 목록 (3개 → 3-run mean±std)')
    ap.add_argument('--devices',      type=int, default=1)
    # 실험 식별
    ap.add_argument('--ft_tag',       default='',
                    help='fine-tune 실험에 추가 식별자 (예: lr1e4, freeze 등).')
    # wandb
    ap.add_argument('--wandb_project', default='duett-mimic-ft')
    ap.add_argument('--wandb_disabled', action='store_true')
    return ap.parse_args()


# ==========================================================
# Main
# ==========================================================
def main():
    args = parse_args()

    # ─── ckpt 자동 분리 ───
    # pretrain 경로의 부모 폴더 이름을 가져와 fine-tune 폴더에 그대로 사용
    # 예: mimic_ssl/n48_s12/pretrain-X.ckpt → mimic_ft/n48_s12/seed{N}/
    pretrain_tag = os.path.basename(os.path.dirname(args.pretrained))
    run_tag = pretrain_tag
    if args.ft_tag:
        run_tag += f'__{args.ft_tag}'

    # n_timesteps 일관성 검증 (pretrain 경로에서 n{X} 추출)
    import re
    m = re.match(r'n(\d+)', pretrain_tag)
    if m:
        ssl_n = int(m.group(1))
        if ssl_n != args.n_timesteps:
            print(f'⚠ Warning: pretrain n_timesteps={ssl_n} '
                  f'vs --n_timesteps={args.n_timesteps}. 일치시키세요.')

    args.ckpt_dir = os.path.join(args.ckpt_dir, run_tag)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print('=' * 60)
    print(f'  Fine-tune from SSL run: {pretrain_tag}')
    if args.ft_tag:
        print(f'  ft_tag: {args.ft_tag}')
    print(f'  pretrained: {os.path.basename(args.pretrained)}')
    print('=' * 60)
    print(f'[ckpt] saving to {args.ckpt_dir}')

    # raw → 가공 → Dataset
    ds_train, ds_val, ds_test, meta = prepare_from_raw(
        art_dir=args.art_dir,
        n_timesteps=args.n_timesteps,
        split_seed=args.split_seed,
    )
    print(f'pos_frac(train) = {ds_train.pos_frac():.4f}')

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 prefetch_factor=2, persistent_workers=args.num_workers > 0,
                 collate_fn=collate_into_seqs)
    dl_train = DataLoader(ds_train, shuffle=True,  **dl_kw)
    dl_val   = DataLoader(ds_val,   shuffle=False, **dl_kw)
    dl_test  = DataLoader(ds_test,  shuffle=False, **dl_kw)

    ft_common_kw = dict(
        d_static_num=ds_train.d_static_num(),
        d_time_series_num=ds_train.d_time_series_num(),
        d_target=ds_train.d_target(),
        pos_frac=ds_train.pos_frac(),
        masked_transform_timesteps=args.n_timesteps,
        max_len=args.n_timesteps,
    )

    all_results = []
    for seed in args.seeds:
        print(f'\n{"="*60}\nFine-tune seed = {seed}\n{"="*60}')
        pl.seed_everything(seed)

        # (1) pretrain ckpt → fine-tune 모델
        ft_model = duett.fine_tune_model(args.pretrained, seed=seed, **ft_common_kw)

        # (2) seed별 ckpt 폴더
        seed_ckpt_dir = os.path.join(args.ckpt_dir, f'seed{seed}')
        os.makedirs(seed_ckpt_dir, exist_ok=True)
        ckpt = ModelCheckpoint(
            dirpath=seed_ckpt_dir,
            save_top_k=args.save_top_k, save_last=False,
            monitor='val_auprc', mode='max',
            filename='ft-{epoch:03d}-{val_auprc:.4f}',
        )

        # (3) wandb logger (seed별 run, run_tag로 group)
        if args.wandb_disabled:
            logger = False
        else:
            logger = WandbLogger(
                project=args.wandb_project,
                name=f'{run_tag}__seed{seed}',
                group=run_tag,
                save_dir=seed_ckpt_dir,
                config={**vars(args), 'seed': seed,
                        'pretrain_tag': pretrain_tag},
            )

        # (4) 학습
        trainer = pl.Trainer(
            accelerator='gpu', devices=args.devices,
            max_epochs=args.max_epochs,
            gradient_clip_val=1.0,
            callbacks=[WarmUpCallback(steps=args.warmup_steps), ckpt],
            num_sanity_val_steps=2,
            logger=logger,
        )
        trainer.fit(ft_model, train_dataloaders=dl_train, val_dataloaders=dl_val)

        # (4) top-k weight averaging
        top_paths = list(ckpt.best_k_models.keys())
        print(f'  top-{len(top_paths)} ckpts saved at {seed_ckpt_dir}')
        final_model = average_models([
            duett.fine_tune_model(p, **ft_common_kw) for p in top_paths
        ])

        # (5) test
        test_res = trainer.test(final_model, dataloaders=dl_test)
        all_results.append({'seed': seed, 'res': test_res[0],
                            'best_path': ckpt.best_model_path})

        if not args.wandb_disabled:
            wandb.finish()

    # ── 최종 요약 ──
    print(f'\n{"="*60}\nFinal results across seeds {args.seeds}\n{"="*60}')
    aurocs = [r['res']['test_auroc'] for r in all_results]
    auprcs    = [r['res']['test_auprc']    for r in all_results]
    print(f'Test AUROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}')
    print(f'Test AUPRC    = {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}')
    for r in all_results:
        print(f"  seed {r['seed']}: AUROC={r['res']['test_auroc']:.4f}, "
              f"AUPRC={r['res']['test_auprc']:.4f}  best={r['best_path']}")


if __name__ == '__main__':
    main()
