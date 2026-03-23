import os
from contextlib import contextmanager
import time
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist

from training.run import parse_arguments

# TIMER 전역 제어
TIME_ENABLED = False # True: 작동 / False: 미작동

def plot_latent_time_attention(attn, save_path=None):
    B, L, T = attn.shape

    fig, axes = plt.subplots(
        nrows=B,
        ncols=1,
        figsize=(12, 2.5 * B),
        sharex=True
    )

    if B == 1:
        axes = [axes]

    for b in range(B):
        sns.heatmap(
            attn[b].cpu().numpy(),
            ax=axes[b],
            cmap="coolwarm",
            cbar=(b == 0),
            xticklabels=range(T),
            yticklabels=[f"latent {i}" for i in range(L)]
        )
        # axes[b].set_ylabel(f"sample {b}")

    axes[-1].set_xlabel("ICU Window Time Steps")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()



# SEED CONTROL
def set_seed(seed: int = 42):
    """
    재현성을 위해 Python, NumPy, PyTorch의 시드를 고정하는 함수
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티-GPU 사용 시
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CUDNN deterministic 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎯 [SEED] All seeds set to {seed} for reproducibility")
    

def seed_worker(worker_id):
    """
    DataLoader의 멀티-프로세싱 워커들의 시드를 고정하는 함수
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

@contextmanager
def timer(name, accelerator=None):
    """
    Context manager for timing code execution.
    Only prints on main process when accelerator is provided.

    Args:
        name: Description of the timed block
        accelerator: Accelerate accelerator instance (optional)
    """
    if not TIME_ENABLED:
        yield
        return

    start = time.perf_counter()
    yield
    end = time.perf_counter()

    # Only log on main process
    if accelerator is not None:
        if accelerator.is_main_process:
            print(f"[Timer] {name}: {end - start:.4f}s")
    else:
        # Fallback to distributed rank check
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[Timer] {name}: {end - start:.4f}s")


def log_memory(tag=""):
    """
    Allocated: 실제로 모델이나 연산에 사용 중인 메모리 양 (=현재 GPU에서 pytorch tensor가 차지하는 메모리 총량)
    Reserved: Pytorch 메모리풀에 의해 예약된 메모리 양
    """
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    free_mem, total_mem = torch.cuda.mem_get_info()          # bytes
    free_mem_mb = free_mem / (1024 ** 2)
    total_mem_mb = total_mem / (1024 ** 2)
    used_mem_mb = total_mem_mb - free_mem_mb
    used_ratio = used_mem_mb / total_mem_mb * 100

    print(f"[GPU MEM - {tag}]")
    print(f"  Allocated: {allocated:.2f} MB")
    print(f"  Reserved : {reserved:.2f} MB")
    print(f"  Used     : {used_mem_mb:.2f} MB / {total_mem_mb:.2f} MB ({used_ratio:.1f}%)")
    print(f"  Free     : {free_mem_mb:.2f} MB")


def mask_tokenized_inputs(input_ids_tensor, attention_mask_tensor, mask_token_id=103, mask_prob=0.2, special_token_ids=[101, 102]):
    """
    101 = [CLS], 102 = [SEP], 103 = [MASK]
    """

    special_token_ids_tensor = torch.tensor(special_token_ids, dtype=input_ids_tensor.dtype, device=input_ids_tensor.device)
    is_special_token = torch.isin(input_ids_tensor, special_token_ids_tensor)
    is_padding = attention_mask_tensor == 0

    cannot_mask = is_special_token | is_padding
    random_mask = torch.rand_like(input_ids_tensor.float()) < mask_prob
    mask = random_mask & (~cannot_mask)

    masked_input_ids = input_ids_tensor.clone()
    masked_input_ids[mask] = mask_token_id
    return masked_input_ids


def padding_text(x, token_max_length=256):
    x = list(x)
    x = x[:token_max_length]
    if len(x) < token_max_length:
        x += [0] * (token_max_length - len(x))
    return x


def get_temperature(epoch, total_epochs, args):
    """
    decay: True일 경우 temperature를 조정하고, False일 경우 고정값 유지
    """
    if args.use_temp_decay:
        return args.init_temp * ((args.final_temp / args.init_temp) ** (epoch / total_epochs))
    else:
        return args.fixed_temp


def debug_tensor(name, tensor, print_stats=False): 
    if torch.isnan(tensor).any():
        print(f"[DEBUG] NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"[DEBUG] Inf detected in {name}")
    if print_stats:
        print(f"[DEBUG] {name} mean: {tensor.mean().item():.4f}, std: {tensor.std().item():.4f}")


class Earlystopping:
    def __init__(self, patience, start_epoch=0, save_path=None, experiment_id=None):
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_auroc = float('-inf')
        self.counter = 0
        self.experiment_id = experiment_id

        # experiment_id가 제공되면 실험별 폴더 생성
        if save_path is not None and experiment_id is not None:
            base_dir = os.path.dirname(save_path)
            filename = os.path.basename(save_path)
            self.save_path = os.path.join(base_dir, filename)
        else:
            self.save_path = save_path

    def __call__(self, args, auroc, model, epoch):
        early_stop = False

        # early stopping 시작 시점은 warm up 종료 시점
        if epoch < self.start_epoch:
            return False

        if auroc > self.best_auroc:
            self.best_auroc = auroc
            self.counter = 0

            if self.save_path is not None:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(model.state_dict(), self.save_path)
                print(f"[Epoch {epoch+1}] 🔥 성능 향상! Best AUROC: {self.best_auroc:.4f} ([공통 경로에 덮어씌웁니다.]: {self.save_path})")
        else:
            self.counter += 1
            print(f"📉 성능 개선 없음. patience 추가 {self.counter}")

            if self.counter >= self.patience:
                early_stop = True
        return early_stop

    def get_best_model_path(self):
        """학습 종료 후 best model 경로 반환"""
        return self.save_path

class stage2_Earlystopping:
    def __init__(self, patience, start_epoch=0, save_path=None, experiment_id=None):
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_auroc = float('-inf')
        self.counter = 0
        self.experiment_id = experiment_id

        # experiment_id가 제공되면 실험별 폴더 생성
        if save_path is not None and experiment_id is not None:
            base_dir = os.path.dirname(save_path)
            filename = os.path.basename(save_path)
            self.save_path = os.path.join(base_dir, f"experiment_{experiment_id}", filename)
        else:
            self.save_path = save_path

    def __call__(self, args, auroc, model, epoch):
        early_stop = False

        # early stopping 시작 시점은 warm up 종료 시점
        if epoch < self.start_epoch:
            return False

        if auroc > self.best_auroc:
            self.best_auroc = auroc
            self.counter = 0

            # 실험별 폴더에 저장
            if self.save_path is not None:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(model.state_dict(), self.save_path)
                print(f"[Epoch {epoch+1}] 🔥 성능 향상! Best AUROC: {self.best_auroc:.4f} (저장 완료: {self.save_path})")
        else:
            self.counter += 1
            print(f"📉 성능 개선 없음. patience 추가 {self.counter}")

            if self.counter >= self.patience:
                early_stop = True
        return early_stop

    def get_best_model_path(self):
        """학습 종료 후 best model 경로 반환"""
        return self.save_path


class stage1_earlystopping:
    """
    Stage 1용 Early Stopping: Loss 기반 (낮을수록 좋음)
    """
    def __init__(self, patience, start_epoch=0):
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_loss = float('inf')  # 낮을수록 좋으므로 inf로 초기화
        self.counter = 0

    def __call__(self, args, loss, model, epoch):
        """
        Args:
            loss: 실제 loss 값 (낮을수록 좋음)
        """
        early_stop = False

        # early stopping 시작 시점은 warm up 종료 시점
        if epoch < self.start_epoch:
            return False

        if loss < self.best_loss:  # Loss가 감소하면 개선
            self.best_loss = loss
            self.counter = 0
            print(f"[Epoch {epoch+1}] 🔥 Loss 개선! (Best Loss: {self.best_loss:.4f})")
        else:
            self.counter += 1
            print(f"📉 Loss 개선 없음. patience 추가 {self.counter} (Current: {loss:.4f}, Best: {self.best_loss:.4f})")

            if self.counter >= self.patience:
                early_stop = True
        return early_stop


def compute_class_weights(label_tensor, num_classes):
    counts = torch.bincount(label_tensor[label_tensor != -1], minlength=num_classes).float()
    total = counts.sum()
    weights = total / (counts + 1e-8)  # Avoid division by zero
    weights = weights / weights.sum() * num_classes  # Optional normalization
    return weights


def debug_unique_batches_gpu(loader, accelerator):
    """
    각 rank가 받은 stay_id 집합의 교집합을 확인해
    중복 배치가 없는지 검증한다. 교집합 크기가 0이면 OK.
    """
    # 1) 로컬 id 수집
    local_ids = []
    for batch in loader:
        local_ids.extend(batch['stay_ids'])
    local_ids = torch.tensor(list(set(local_ids)), device=accelerator.device)

    # 2) rank 간 gather
    gathered = accelerator.gather(local_ids)
    if accelerator.is_main_process:
        sets = [set(t.tolist()) for t in gathered]
        inter = set.intersection(*sets)
        print(f"[DEBUG] 🟢 intersection size = {len(inter)} (0이면 중복 없음 의미함.)")


def print_modality_stats(name, tensor):
    print(f"[{name}] mean: {tensor.abs().mean().item():.4e}, max: {tensor.abs().max().item():.4e}")


# ==================== GRADIENT AMPLIFICATION FOR SELECTIVE LAYERS ====================

def DEFINE_LAYER_GROUP(model, layer_patterns=None):
    """
    모델에서 지정된 패턴의 모듈을 찾아서 그룹 G로 정의 (모듈 레벨)
    선택된 모듈의 모든 파라미터가 함께 증폭됨
    나중에 다른 레이어 패턴도 추가 가능한 구조

    Args:
        model: 훈련 중인 모델 (Accelerator로 wrapping 가능)
        layer_patterns: 찾을 모듈 이름 패턴 리스트
        기본값: ['cxr_cross', 'text_cross']

    Returns:
        G: 선택된 모듈 객체 리스트 (in_proj, out_proj 등 포함)
        G_names: 해당 모듈의 이름 리스트 (디버깅용)
    """
    if layer_patterns is None:
        layer_patterns = ['cxr_cross', 'text_cross']

    G = []
    G_names = []

    unwrapped_model = getattr(model, "module", model)

    for name, module in unwrapped_model.named_modules():
        # 패턴과 일치하고 파라미터를 가진 모듈 찾기
        if any(pattern in name for pattern in layer_patterns):
            # 파라미터가 있는 모듈만 선택
            if sum(p.numel() for p in module.parameters()) > 0:
                G.append(module)
                G_names.append(name)

    # print(f"🎯 [Layer Group] Found {len(G)} modules with patterns {layer_patterns}:")
    # for name in sorted(G_names):
    #     print(f"   - {name}")

    return G, G_names


def GRADIENT_AMPLIFICATION(G, G_names, alpha=3.0, exclude_patterns=None):
    """
    지정된 모듈의 파라미터 그래디언트에 alpha를 곱하여 증폭
    제외할 파라미터 패턴을 유연하게 지정 가능

    Args:
        G: 선택된 모듈 객체 리스트
        G_names: 해당 모듈의 이름 리스트 (디버깅용)
        alpha: 그래디언트 증폭 배수 (default: 3.0)
        exclude_patterns: 제외할 파라미터 이름 패턴 리스트
    """
    if not G:
        return

    if exclude_patterns is None:
        exclude_patterns = []

    amplified_count = 0
    skipped_count = 0
    total_grad_before = 0.0
    total_grad_after = 0.0

    for module, module_name in zip(G, G_names):
        module_params_count = 0
        module_skipped_count = 0

        for param_name, param in module.named_parameters():
            # 제외 패턴 확인
            should_exclude = any(pattern in param_name for pattern in exclude_patterns)

            if should_exclude:
                skipped_count += 1
                module_skipped_count += 1
                continue

            if param.grad is not None:
                grad_norm_before = param.grad.norm().item()
                param.grad *= alpha
                grad_norm_after = param.grad.norm().item()

                total_grad_before += grad_norm_before
                total_grad_after += grad_norm_after
                amplified_count += 1
                module_params_count += 1

        if module_params_count > 0 or module_skipped_count > 0:
            skip_info = f" (excluded: {module_skipped_count})" if module_skipped_count > 0 else ""
            # print(f"   [AMP] {module_name}: {module_params_count} params amplified{skip_info}")

    # if amplified_count > 0:
        # print(f"[GRADIENT_AMPLIFICATION] Total {amplified_count} parameters (α={alpha:.1f}x)")
        # if exclude_patterns:
            # print(f"   Exclude patterns: {exclude_patterns} (skipped: {skipped_count})")
        # print(f"   Before: {total_grad_before:.4e} | After: {total_grad_after:.4e}")
    # else:
        # print("⚠️  Warning: No gradients found to amplify")