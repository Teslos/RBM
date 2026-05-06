"""
Classical RBM vs. Nosé-Hoover RBM — side-by-side comparison
=============================================================
Trains both models on MNIST with identical hyperparameters and
plots reconstruction error, hidden activation, temperature (NH only),
and learned weight filters.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from rbm_nose_hoover import RBM_NoseHoover

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42


# ──────────────────────────────────────────────────────────────
# 1.  Classical RBM (fixed T = 1)
# ──────────────────────────────────────────────────────────────

class RBM(nn.Module):
    """Standard RBM trained with CD-k at fixed temperature T=1."""

    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_visible))
        self.c = nn.Parameter(torch.zeros(n_hidden))
        self.h_activation_history = []

    def prob_h_given_v(self, v):
        return torch.sigmoid(v @ self.W + self.c)

    def prob_v_given_h(self, h):
        return torch.sigmoid(h @ self.W.t() + self.b)

    def sample_h(self, v):
        p = self.prob_h_given_v(v)
        return p, torch.bernoulli(p)

    def sample_v(self, h):
        p = self.prob_v_given_h(h)
        return p, torch.bernoulli(p)

    def contrastive_divergence(self, v0: torch.Tensor, k: int = 1, lr: float = 0.01) -> float:
        with torch.no_grad():
            p_h0, h = self.sample_h(v0)
            for _ in range(k):
                p_v, v = self.sample_v(h)
                p_hk, h = self.sample_h(v)

            batch_size = v0.size(0)
            pos = v0.t() @ p_h0 / batch_size
            neg = v.t()  @ p_hk / batch_size

            self.W.data += lr * (pos - neg)
            self.b.data += lr * (v0 - v).mean(dim=0)
            self.c.data += lr * (p_h0 - p_hk).mean(dim=0)

        self.h_activation_history.append(p_h0.mean().item())
        return ((v0 - p_v) ** 2).mean().item()


# ──────────────────────────────────────────────────────────────
# 2.  Shared training loop
# ──────────────────────────────────────────────────────────────

def train(model, dataloader, n_epochs, lr, k, device):
    model.to(device)
    epoch_errors = []
    for epoch in range(n_epochs):
        errs = []
        for batch, _ in dataloader:
            v0 = (batch.view(batch.size(0), -1).to(device) > 0.5).float()
            errs.append(model.contrastive_divergence(v0, k=k, lr=lr))
        epoch_errors.append(float(np.mean(errs)))
    return epoch_errors


# ──────────────────────────────────────────────────────────────
# 3.  Plotting
# ──────────────────────────────────────────────────────────────

def plot_filters(W: torch.Tensor, n_show: int, title: str, ax):
    """Plot the first n_show weight filters as 28×28 patches."""
    W = W.detach().cpu().numpy()
    cols = 8
    rows = (n_show + cols - 1) // cols
    grid = np.zeros((rows * 28, cols * 28))
    for idx in range(min(n_show, W.shape[1])):
        r, c = divmod(idx, cols)
        patch = W[:, idx].reshape(28, 28)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        grid[r*28:(r+1)*28, c*28:(c+1)*28] = patch
    ax.imshow(grid, cmap="gray", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def plot_comparison(classical, nh, errors_c, errors_nh, h_target):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Classical RBM vs. Nosé-Hoover RBM", fontsize=14, fontweight="bold")

    # ── Row 1: metrics ────────────────────────────────────────
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.plot(errors_c,  label="Classical", color="steelblue")
    ax1.plot(errors_nh, label="Nosé-Hoover", color="firebrick")
    ax1.set_title("Reconstruction Error")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.legend()

    ax2 = fig.add_subplot(2, 4, 2)
    # Downsample history to per-epoch points for readability
    def epoch_means(hist, n_epochs):
        chunk = max(1, len(hist) // n_epochs)
        return [float(np.mean(hist[i*chunk:(i+1)*chunk])) for i in range(n_epochs)]

    n_epochs = len(errors_c)
    ax2.plot(epoch_means(classical.h_activation_history, n_epochs),
             label="Classical", color="steelblue")
    ax2.plot(epoch_means(nh.h_activation_history, n_epochs),
             label="Nosé-Hoover", color="firebrick")
    ax2.axhline(h_target, linestyle="--", color="gray", label=f"target={h_target}")
    ax2.set_title("Avg Hidden Activation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("⟨h⟩")
    ax2.legend()

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.plot(epoch_means(nh.temperature_history, n_epochs),
             color="firebrick", label="Nosé-Hoover T")
    ax3.axhline(nh.T0, linestyle="--", color="gray", label=f"T0={nh.T0}")
    ax3.axhline(1.0, linestyle=":", color="steelblue", label="Classical T=1")
    ax3.set_title("Effective Temperature")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("T")
    ax3.legend()

    ax4 = fig.add_subplot(2, 4, 4)
    ax4.plot(epoch_means(nh.temperature_history, n_epochs),  # xi proxy
             color="darkorange")
    xi_per_epoch = epoch_means(
        [np.log(t / nh.T0) for t in nh.temperature_history], n_epochs
    )
    ax4.clear()
    ax4.plot(xi_per_epoch, color="darkorange")
    ax4.axhline(0, linestyle="--", color="gray")
    ax4.set_title("Thermostat Variable ξ")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("ξ")

    # ── Row 2: weight filters ─────────────────────────────────
    ax5 = fig.add_subplot(2, 2, 3)
    plot_filters(classical.W, n_show=32, title="Classical RBM — learned filters", ax=ax5)

    ax6 = fig.add_subplot(2, 2, 4)
    plot_filters(nh.W, n_show=32, title="Nosé-Hoover RBM — learned filters", ax=ax6)

    plt.tight_layout()
    out = OUTPUT_DIR / "rbm_comparison.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved to {out}")


# ──────────────────────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    N_VISIBLE  = 784
    N_HIDDEN   = 256
    N_EPOCHS   = 10
    LR         = 0.01
    K          = 1
    BATCH_SIZE = 64
    H_TARGET   = 0.3

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Both models see the same data in the same order
    torch.manual_seed(SEED)
    loader_c = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
    torch.manual_seed(SEED)
    loader_nh = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                           generator=torch.Generator().manual_seed(SEED))

    # Identical weight initialisation
    torch.manual_seed(SEED)
    classical = RBM(N_VISIBLE, N_HIDDEN)

    torch.manual_seed(SEED)
    nh = RBM_NoseHoover(N_VISIBLE, N_HIDDEN, T0=1.0, h_target=H_TARGET, Q=10.0, alpha=0.01)

    print("\nTraining Classical RBM")
    errors_c = train(classical, loader_c, N_EPOCHS, LR, K, device)
    for i, e in enumerate(errors_c, 1):
        print(f"  Epoch {i:2d} | Error {e:.4f} | h_avg {np.mean(classical.h_activation_history[(i-1)*938:i*938]):.3f}")

    print("\nTraining Nose-Hoover RBM")
    errors_nh = train(nh, loader_nh, N_EPOCHS, LR, K, device)
    for i, e in enumerate(errors_nh, 1):
        print(f"  Epoch {i:2d} | Error {e:.4f} | h_avg {np.mean(nh.h_activation_history[(i-1)*938:i*938]):.3f} | T {nh.T0*np.exp(np.mean([np.log(t/nh.T0) for t in nh.temperature_history[(i-1)*938:i*938]])):.4f}")

    print("\nFinal Summary")
    print(f"  Classical   final error: {errors_c[-1]:.4f}  |  h_avg: {np.mean(classical.h_activation_history[-938:]):.3f}")
    print(f"  Nose-Hoover final error: {errors_nh[-1]:.4f}  |  h_avg: {np.mean(nh.h_activation_history[-938:]):.3f}  |  final T: {nh.temperature:.4f}")

    plot_comparison(classical, nh, errors_c, errors_nh, H_TARGET)
