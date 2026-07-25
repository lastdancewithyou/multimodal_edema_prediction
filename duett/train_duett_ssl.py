# /home/DAHS1/gangmin/my_research/clinical_multimodal_learning/duett/train_duett_ssl.py
"""Standalone SSL pretraining for DuETT on MIMIC. Run inside tmux."""
import os, sys, argparse, warnings, pickle

# 소음 억제
warnings.filterwarnings("ignore", message=".*ambiguous collection.*")
warnings.filterwarnings("ignore", message=".*rotary embedding dimension.*")
try:
    from loguru import logger as _loguru
    _loguru.disable("x_transformers")
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import duett
from mimic_dataset import (collate_into_seqs, prepare_from_raw,
                            prepare_for_sliding_ssl, ART_DIR)


class WarmUpCallback(pl.callbacks.Callback):
    def __init__(self, steps=2000, invsqrt=True, decay=None):
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art_dir',     default=ART_DIR)
    ap.add_argument('--ckpt_dir',    default=os.path.join(HERE, 'checkpoints/mimic_ssl'),
                    help='base 경로. 실제 저장은 그 아래 n{N_TIMESTEPS}/ 폴더로 자동 분리')
    ap.add_argument('--n_timesteps', type=int, default=24,
                    help='시간 슬롯 수 = 윈도우 길이(h). cohort/grid/split이 이 값에 의존')
    ap.add_argument('--split_seed',  type=int, default=42,
                    help='train/val/test split 재현용 (SSL/finetune 일관성 유지)')
    # sliding window SSL
    ap.add_argument('--sliding_stride', type=int, default=12,
                    help='0: 첫 n_timesteps만 (paper와 동일). '
                         '>0: stride 간격으로 sliding window SSL (예: 12).')
    ap.add_argument('--max_stay_hours', type=int, default=336,
                    help='sliding 모드에서 한 환자에서 사용할 최대 시간(h). 기본 336h(14일)')
    ap.add_argument('--batch_size',  type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--max_epochs',  type=int, default=3)
    ap.add_argument('--warmup_steps',type=int, default=2000)
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--devices',     type=int, default=1)
    # 실험 식별
    ap.add_argument('--ssl_tag',     default='',
                    help='SSL 실험에 추가 식별자 (예: bnp, extra_var 등). '
                         'run_tag 뒤에 __{ssl_tag}로 붙어 ckpt 폴더/finetune 매칭에 사용.')
    # wandb
    ap.add_argument('--wandb_project', default='duett-mimic-ssl',
                    help='wandb project 이름')
    ap.add_argument('--wandb_run_name', default='',
                    help='wandb run 이름. None이면 ckpt run_tag 사용 (예: n48_s12)')
    ap.add_argument('--wandb_disabled', action='store_true',
                    help='wandb 비활성화 (오프라인 또는 디버깅 시)')
    return ap.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # ─── SSL 모드 표시 (한눈에 보이게) ───
    print('=' * 60)
    if args.sliding_stride > 0:
        print('  ★ SLIDING WINDOW SSL ★')
        print(f'    n_timesteps  = {args.n_timesteps}')
        print(f'    stride       = {args.sliding_stride}')
        print(f'    max_stay_hrs = {args.max_stay_hours}')
    else:
        print(f'  SINGLE-WINDOW SSL (first {args.n_timesteps}h only)')
    print('=' * 60)

    # N_TIMESTEPS (+ sliding 시 stride)별로 ckpt 폴더 자동 분리
    run_tag = f'n{args.n_timesteps}'
    if args.sliding_stride > 0:
        run_tag += f'_s{args.sliding_stride}'
    if args.ssl_tag:
        run_tag += f'__{args.ssl_tag}'
    args.ckpt_dir = os.path.join(args.ckpt_dir, run_tag)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f'[ckpt] saving to {args.ckpt_dir}')

    # raw → 가공 → Dataset
    if args.sliding_stride > 0:
        ds_train, ds_val, ds_test, meta = prepare_for_sliding_ssl(
            art_dir=args.art_dir,
            n_timesteps=args.n_timesteps,
            stride=args.sliding_stride,
            max_stay_hours=args.max_stay_hours,
            split_seed=args.split_seed,
        )
    else:
        ds_train, ds_val, ds_test, meta = prepare_from_raw(
            art_dir=args.art_dir,
            n_timesteps=args.n_timesteps,
            split_seed=args.split_seed,
        )
    print(f'pos_frac(train)={ds_train.pos_frac():.4f}')

    # KD student가 재사용할 수 있도록 stats 포함 meta를 ckpt 폴더에 저장.
    # (means/stds/age_mean/age_std/N_TIMESTEPS/train_ids/val_ids/test_ids)
    meta_out = os.path.join(args.ckpt_dir, 'meta_with_stats.pkl')
    with open(meta_out, 'wb') as f:
        pickle.dump(meta, f)
    print(f'[meta] saved with train stats → {meta_out}')

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 prefetch_factor=2, persistent_workers=args.num_workers > 0,
                 collate_fn=collate_into_seqs)
    dl_train = DataLoader(ds_train, shuffle=True,  **dl_kw)
    dl_val   = DataLoader(ds_val,   shuffle=False, **dl_kw)

    model = duett.pretrain_model(
        d_static_num=ds_train.d_static_num(),
        d_time_series_num=ds_train.d_time_series_num(),
        d_target=ds_train.d_target(),
        pos_frac=ds_train.pos_frac(),
        masked_transform_timesteps=args.n_timesteps,
        max_len=args.n_timesteps,
        seed=args.seed,
    )

    ckpt = ModelCheckpoint(
        dirpath=args.ckpt_dir, save_last=True,
        monitor='val_loss', mode='min', save_top_k=1,
        filename='pretrain-{epoch:03d}-{val_loss:.4f}',
    )

    # ─── wandb logger 구성 ───
    if args.wandb_disabled:
        logger = False
        print('[wandb] disabled')
    else:
        wb_run_name = args.wandb_run_name or run_tag
        logger = WandbLogger(
            project=args.wandb_project,
            name=wb_run_name,
            save_dir=args.ckpt_dir,
            config={
                'mode': 'sliding' if args.sliding_stride > 0 else 'single',
                'n_timesteps': args.n_timesteps,
                'sliding_stride': args.sliding_stride,
                'max_stay_hours': args.max_stay_hours,
                'batch_size': args.batch_size,
                'max_epochs': args.max_epochs,
                'warmup_steps': args.warmup_steps,
                'seed': args.seed,
                'split_seed': args.split_seed,
                'train_samples': len(ds_train),
                'val_samples': len(ds_val),
                'pos_frac': ds_train.pos_frac(),
                'd_static_num': ds_train.d_static_num(),
                'd_time_series_num': ds_train.d_time_series_num(),
            },
        )
        print(f'[wandb] project={args.wandb_project}, run={wb_run_name}')

    trainer = pl.Trainer(
        accelerator='gpu', devices=args.devices,
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        callbacks=[WarmUpCallback(steps=args.warmup_steps), ckpt],
        num_sanity_val_steps=2,
        logger=logger,
    )
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)

    print('best ckpt:', ckpt.best_model_path)


if __name__ == '__main__':
    main()
