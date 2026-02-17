import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class uniform_loss(nn.Module):
    def __init__(self, t=0.07):
        super(uniform_loss, self).__init__()
        self.t = t

    def forward(self, x):
        return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()


def generate_optimal_target(num_classes, targets_per_class, embed_dim, temperature=0.3, n_iter=5000, save_dir='/home/DAHS1/gangmin/my_research/src/output/targets', device='cuda', visualize=True, use_multiple_targets=False):
    """
    - 클래스들이 특정 영역에 몰리지 않고 최대 거리를 유지하여 클래스 간의 결정 경계를 명확하게 만듦.
    - 이를 통해 feature space가 uniform하게 사용되며, 모든 클래스가 동등하게 표현될 수 있도록 함.
    - L_u는 클래스 목표들이 서로 가까울수록 값이 커지며, L_u를 최소화하는 것은 클래스 목표들이 서로 멀리 떨어지게 만드는 것이고, 이상적인 균일한 분포는 곧 L_u의 이론적인 최솟값에 해당됨.
    """
    print("="*60)
    print("🎯 Generating Optimal Target Prototypes")
    print("="*60)
    print(f"   Number of Classes (N): {num_classes}")
    print(f"   Targets per Class: {'Multiple' if use_multiple_targets else 'Single'} ({targets_per_class if use_multiple_targets else 1})")
    print(f"   Embedding Dimension (M): {embed_dim}")
    print(f"   Temperature: {temperature}")
    print(f"   Iterations: {n_iter}")
    print(f"   Device: {device}")
    print("="*60)

    if use_multiple_targets:
        total_targets = num_classes * targets_per_class
    else:
        total_targets = num_classes

    x = torch.randn(total_targets, embed_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=1e-3)
    criterion = uniform_loss(t=temperature)

    best_loss = float("inf")
    best_x = None

    for i in range(n_iter):
        x_norm = F.normalize(x, dim=1)
        loss_u = criterion(x_norm)

        if i % 1000 == 0:
            print(i, loss_u.item())

        if loss_u.item() < best_loss:
            best_loss = loss_u.item()
            best_x = x_norm.detach().cpu()

        optimizer.zero_grad()
        loss_u.backward()
        optimizer.step()

    if use_multiple_targets:
        optimal_target = best_x.view(num_classes, targets_per_class, embed_dim)
    else:
        optimal_target = best_x.view(num_classes, embed_dim)

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f'optimal_target_{num_classes}_{embed_dim}.npy'
    )

    np.save(save_path, optimal_target.numpy())

    print(f"✅ Target generation complete")
    print(f"   Final loss: {best_loss:.6f}")
    print(f"   Shape: {optimal_target.shape}")
    print(f"   Saved to: {save_path}")

    if visualize:
        print("\n📊 Generating visualizations...")
        actual_targets_per_class = targets_per_class
        mode_label = 'multiple' if use_multiple_targets else 'single'
        visualize_targets(optimal_target.numpy(), num_classes, actual_targets_per_class, save_dir, method='pca', mode_label=mode_label)
        visualize_targets(optimal_target.numpy(), num_classes, actual_targets_per_class, save_dir, method='tsne', mode_label=mode_label)

    return optimal_target, save_path


def visualize_targets(targets, num_classes, targets_per_class, save_dir, method='pca', mode_label='single'):

    targets_flat = targets.reshape(-1, targets.shape[-1])

    if method == 'pca':
        reducer = PCA(n_components=2)
        targets_2d = reducer.fit_transform(targets_flat)
        title = f'Target Distribution (PCA) - {mode_label.capitalize()} Target per Class'
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(targets_flat)-1))
        targets_2d = reducer.fit_transform(targets_flat)
        title = f'Target Distribution (t-SNE) - {mode_label.capitalize()} Target per Class'

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for cls_idx in range(num_classes):
        start_idx = cls_idx * targets_per_class
        end_idx = start_idx + targets_per_class

        cls_targets_2d = targets_2d[start_idx:end_idx]

        plt.scatter(cls_targets_2d[:, 0], cls_targets_2d[:, 1],
                c=[colors[cls_idx]], label=f'Class {cls_idx}',
                s=100, alpha=0.7, edgecolors='black')

        # 클래스 중심 표시
        centroid = cls_targets_2d.mean(axis=0)
        plt.scatter(centroid[0], centroid[1],
                c=[colors[cls_idx]], marker='*', s=500,
                edgecolors='black', linewidths=2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    viz_path = os.path.join(save_dir, f'target_visualization_{mode_label}_{method}.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   Visualization saved to: {viz_path}")
    return viz_path


if __name__ == "__main__":
    generate_optimal_target(
        num_classes=3,
        targets_per_class=0,
        embed_dim=128,
        temperature=0.07,
        n_iter=10000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_multiple_targets=False # 클래스별 target을 추가로 부여할 것인지 말 것인지 결정함.
    )