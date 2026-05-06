"""
Does NH find a better schedule than you'd pick by hand?
========================================================
We define a grid of plausible hand-crafted T(t) schedules and train an
RBM with each one on MNIST for 10 epochs.  NH runs alongside and produces
its own schedule automatically.

The question: does NH land near the best manual schedule, or does it find
something a human wouldn't have chosen?

Schedules tested:
  Constant   : T=1.0, T=0.7, T=0.5, T=0.3
  Linear     : T: 1.0->T_end  for T_end in {0.8, 0.6, 0.4, 0.2}
  Exponential: T(e) = 1.0 * r^e  for r in {0.98, 0.95, 0.92, 0.88}
  Cosine     : T(e) = T_min + 0.5*(1-T_min)*(1+cos(pi*e/N))
               for T_min in {0.7, 0.5, 0.3}
  Step       : T drops by half at epoch 5
               starting from T0 in {1.0, 0.8}
  NH         : adaptive (run last, placed on all plots for comparison)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rbm_nose_hoover import RBM_NoseHoover

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED       = 42
N_VISIBLE  = 784
N_HIDDEN   = 256
N_EPOCHS   = 10
LR         = 0.01
K          = 1
BATCH_SIZE = 64
H_TARGET   = 0.3


# ──────────────────────────────────────────────────────────────
# Schedule definitions  (each returns T given epoch index 0..N-1)
# ──────────────────────────────────────────────────────────────

def make_schedules(n_epochs):
    schedules = {}

    # Constant
    for T in [1.0, 0.7, 0.5, 0.3]:
        schedules[f"const T={T}"] = [T] * n_epochs

    # Linear  T: 1.0 -> T_end
    for T_end in [0.8, 0.6, 0.4, 0.2]:
        schedules[f"linear ->{T_end}"] = [
            1.0 - (1.0 - T_end) * e / (n_epochs - 1)
            for e in range(n_epochs)
        ]

    # Exponential  T(e) = r^e
    for r in [0.98, 0.95, 0.92, 0.88]:
        schedules[f"exp r={r}"] = [r ** e for e in range(n_epochs)]

    # Cosine  T_min + 0.5*(1-T_min)*(1+cos(pi*e/(N-1)))
    for T_min in [0.7, 0.5, 0.3]:
        schedules[f"cosine min={T_min}"] = [
            T_min + 0.5 * (1.0 - T_min) * (1 + np.cos(np.pi * e / (n_epochs - 1)))
            for e in range(n_epochs)
        ]

    # Step: drop by half at midpoint
    mid = n_epochs // 2
    for T0 in [1.0, 0.8]:
        schedules[f"step T0={T0}"] = [T0 if e < mid else T0 * 0.5
                                      for e in range(n_epochs)]

    return schedules


# ──────────────────────────────────────────────────────────────
# Minimal RBM
# ──────────────────────────────────────────────────────────────

class RBM:
    def __init__(self, device):
        g = torch.Generator().manual_seed(SEED)
        self.W = (torch.randn(N_VISIBLE, N_HIDDEN, generator=g) * 0.01).to(device)
        self.b = torch.zeros(N_VISIBLE).to(device)
        self.c = torch.zeros(N_HIDDEN).to(device)
        self.device      = device
        self.temperature = 1.0

    def _ph(self, v):
        return torch.sigmoid((v @ self.W + self.c) / self.temperature)

    def _pv(self, h):
        return torch.sigmoid((h @ self.W.t() + self.b) / self.temperature)

    def cd(self, v0):
        with torch.no_grad():
            p_h0 = self._ph(v0)
            h    = torch.bernoulli(p_h0)
            for _ in range(K):
                p_v  = self._pv(h)
                v    = torch.bernoulli(p_v)
                p_hk = self._ph(v)
                h    = torch.bernoulli(p_hk)
            bs = v0.size(0)
            self.W += LR * (v0.t() @ p_h0 / bs - v.t() @ p_hk / bs)
            self.b += LR * (v0 - v).mean(0)
            self.c += LR * (p_h0 - p_hk).mean(0)
        return ((v0 - p_v) ** 2).mean().item()


def binarise(batch, device):
    return (batch.view(batch.size(0), -1).to(device) > 0.5).float()


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def make_loader():
    tf   = transforms.Compose([transforms.ToTensor()])
    data = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                      generator=torch.Generator().manual_seed(SEED))


def train_scheduled(schedule_per_epoch, device):
    """Train with a per-epoch temperature schedule, return per-epoch errors."""
    model  = RBM(device)
    errors = []
    for ep, T in enumerate(schedule_per_epoch):
        model.temperature = T
        errs = [model.cd(binarise(b, device)) for b, _ in make_loader()]
        errors.append(float(np.mean(errs)))
    return errors


def train_nh(device):
    model = RBM_NoseHoover(
        N_VISIBLE, N_HIDDEN, T0=1.0, h_target=H_TARGET, Q=10.0, alpha=0.01
    )
    model.to(device)
    errors = []
    for _ in range(N_EPOCHS):
        errs = [model.contrastive_divergence(binarise(b, device), k=K, lr=LR)
                for b, _ in make_loader()]
        errors.append(float(np.mean(errs)))

    # Extract per-epoch mean temperature
    n_batches   = len(errors[0:1])          # just to get batch count
    n_per_epoch = len(model.temperature_history) // N_EPOCHS
    T_per_epoch = [
        float(np.mean(model.temperature_history[i*n_per_epoch:(i+1)*n_per_epoch]))
        for i in range(N_EPOCHS)
    ]
    return errors, T_per_epoch


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot(results, schedules, nh_errors, nh_T_per_epoch):
    epochs = list(range(1, N_EPOCHS + 1))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        "NH vs hand-crafted schedules on MNIST (10 epochs)",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: T(t) curves ──────────────────────────────────
    ax = axes[0]
    cmap   = cm.get_cmap("tab20", len(schedules))
    groups = {}
    for i, (name, T_curve) in enumerate(schedules.items()):
        group = name.split()[0]
        lw    = 1.2
        alpha = 0.55
        ax.plot(epochs, T_curve, color=cmap(i), lw=lw, alpha=alpha, label=name)
        groups.setdefault(group, []).append(i)

    ax.plot(epochs, nh_T_per_epoch, color="black", lw=2.5, zorder=10,
            label="Nose-Hoover (auto)", linestyle="-")
    ax.set_title("Temperature schedules")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("T")
    ax.legend(fontsize=5.5, ncol=2)

    # ── Panel 2: learning curves (error vs epoch) ─────────────
    ax = axes[1]
    for i, (name, errs) in enumerate(results.items()):
        ax.plot(epochs, errs, color=cmap(i), lw=1.0, alpha=0.5)
    ax.plot(epochs, nh_errors, color="black", lw=2.5, zorder=10,
            label="Nose-Hoover (auto)")
    ax.set_title("Reconstruction error per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)

    # ── Panel 3: final-error bar chart sorted best->worst ─────
    ax = axes[2]
    final = {name: errs[-1] for name, errs in results.items()}
    final["Nose-Hoover"] = nh_errors[-1]

    sorted_names = sorted(final, key=final.get)
    sorted_vals  = [final[n] for n in sorted_names]
    colors       = ["black" if n == "Nose-Hoover" else "steelblue"
                    for n in sorted_names]
    bars = ax.barh(sorted_names, sorted_vals, color=colors, alpha=0.75)
    ax.set_title("Final error (epoch 10), sorted")
    ax.set_xlabel("MSE")
    ax.invert_yaxis()

    # annotate NH rank
    nh_rank = sorted_names.index("Nose-Hoover") + 1
    ax.set_title(
        f"Final error — NH ranks {nh_rank}/{len(sorted_names)}"
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "rbm_schedule_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out}")

    return sorted_names, sorted_vals, nh_rank


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    schedules = make_schedules(N_EPOCHS)
    print(f"\n{len(schedules)} schedules to evaluate + NH\n")

    results = {}
    for name, T_curve in schedules.items():
        errs = train_scheduled(T_curve, device)
        results[name] = errs
        print(f"  {name:25s}  final error {errs[-1]:.4f}  "
              f"T: {T_curve[0]:.2f}->{T_curve[-1]:.2f}")

    print("\nRunning Nose-Hoover...")
    nh_errors, nh_T_per_epoch = train_nh(device)
    print(f"  {'Nose-Hoover':25s}  final error {nh_errors[-1]:.4f}  "
          f"T: {nh_T_per_epoch[0]:.2f}->{nh_T_per_epoch[-1]:.2f}")

    sorted_names, sorted_vals, nh_rank = plot(
        results, schedules, nh_errors, nh_T_per_epoch
    )

    print(f"\nNH final error: {nh_errors[-1]:.4f}")
    print(f"Best schedule:  {sorted_names[0]} ({sorted_vals[0]:.4f})")
    print(f"Worst schedule: {sorted_names[-1]} ({sorted_vals[-1]:.4f})")
    print(f"NH rank: {nh_rank} / {len(sorted_names)}")

    gap_to_best  = (nh_errors[-1] - sorted_vals[0])  / sorted_vals[0]  * 100
    gap_to_worst = (sorted_vals[-1] - nh_errors[-1]) / sorted_vals[-1] * 100
    print(f"Gap to best:  {gap_to_best:+.1f}%")
    print(f"Gap to worst: {gap_to_worst:+.1f}%")
