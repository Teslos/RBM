"""
Schedule grid on non-stationary MNIST -> FashionMNIST switch
=============================================================
Phase 1 (epochs 1-10):  MNIST
Phase 2 (epochs 11-20): FashionMNIST  (no warning)

Fixed schedules are designed for 20 epochs total and cannot react to
the distribution change.  NH adapts its T in response to the shift in
h_avg after the switch.

Key question: does NH outperform the best fixed schedule in phase 2?
If so, the adaptive feedback earns its keep specifically in the
non-stationary setting.
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

SEED           = 42
N_VISIBLE      = 784
N_HIDDEN       = 256
N_EPOCHS_PHASE = 10
N_EPOCHS       = 2 * N_EPOCHS_PHASE
LR             = 0.01
K              = 1
BATCH_SIZE     = 64
H_TARGET       = 0.3


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────

def make_loaders():
    tf      = transforms.Compose([transforms.ToTensor()])
    mnist   = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    fashion = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)

    def loader(ds):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
    return loader(mnist), loader(fashion)


# ──────────────────────────────────────────────────────────────
# Schedule definitions over 20 epochs
# ──────────────────────────────────────────────────────────────

def make_schedules(n_epochs):
    sched = {}

    for T in [1.0, 0.7, 0.5, 0.3]:
        sched[f"const T={T}"] = [T] * n_epochs

    for T_end in [0.8, 0.6, 0.4, 0.2]:
        sched[f"linear ->{T_end}"] = [
            1.0 - (1.0 - T_end) * e / (n_epochs - 1) for e in range(n_epochs)
        ]

    for r in [0.98, 0.95, 0.92, 0.88]:
        sched[f"exp r={r}"] = [r ** e for e in range(n_epochs)]

    for T_min in [0.7, 0.5, 0.3]:
        sched[f"cosine min={T_min}"] = [
            T_min + 0.5 * (1.0 - T_min) * (1 + np.cos(np.pi * e / (n_epochs - 1)))
            for e in range(n_epochs)
        ]

    mid = n_epochs // 2
    for T0 in [1.0, 0.8]:
        sched[f"step T0={T0}"] = [T0 if e < mid else T0 * 0.5 for e in range(n_epochs)]

    return sched


# ──────────────────────────────────────────────────────────────
# Minimal RBM with injectable temperature
# ──────────────────────────────────────────────────────────────

class RBM:
    def __init__(self, device):
        g = torch.Generator().manual_seed(SEED)
        self.W = (torch.randn(N_VISIBLE, N_HIDDEN, generator=g) * 0.01).to(device)
        self.b = torch.zeros(N_VISIBLE).to(device)
        self.c = torch.zeros(N_HIDDEN).to(device)
        self.device      = device
        self.temperature = 1.0

    def _ph(self, v): return torch.sigmoid((v @ self.W + self.c) / self.temperature)
    def _pv(self, h): return torch.sigmoid((h @ self.W.t() + self.b) / self.temperature)

    def cd(self, v0):
        with torch.no_grad():
            p_h0 = self._ph(v0)
            h = torch.bernoulli(p_h0)
            for _ in range(K):
                p_v  = self._pv(h);  v   = torch.bernoulli(p_v)
                p_hk = self._ph(v);  h   = torch.bernoulli(p_hk)
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

def train_scheduled(schedule_per_epoch, loader1, loader2, device):
    model  = RBM(device)
    errors = []
    loaders = [loader1] * N_EPOCHS_PHASE + [loader2] * N_EPOCHS_PHASE
    for ep, (T, ldr) in enumerate(zip(schedule_per_epoch, loaders)):
        model.temperature = T
        errs = [model.cd(binarise(b, device)) for b, _ in ldr]
        errors.append(float(np.mean(errs)))
    return errors


def train_nh(loader1, loader2, device):
    model = RBM_NoseHoover(
        N_VISIBLE, N_HIDDEN, T0=1.0, h_target=H_TARGET, Q=10.0, alpha=0.01
    )
    model.to(device)
    errors = []

    for ldr in [loader1] * N_EPOCHS_PHASE + [loader2] * N_EPOCHS_PHASE:
        errs = [model.contrastive_divergence(binarise(b, device), k=K, lr=LR)
                for b, _ in ldr]
        errors.append(float(np.mean(errs)))

    n_per_epoch = len(model.temperature_history) // N_EPOCHS
    T_per_epoch = [
        float(np.mean(model.temperature_history[i*n_per_epoch:(i+1)*n_per_epoch]))
        for i in range(N_EPOCHS)
    ]
    return errors, T_per_epoch, model


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot(results, schedules, nh_errors, nh_T_per_epoch):
    epochs = list(range(1, N_EPOCHS + 1))
    switch = N_EPOCHS_PHASE + 0.5
    cmap   = plt.colormaps.get_cmap("tab20")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(
        "MNIST -> FashionMNIST switch: NH vs hand-crafted schedules",
        fontsize=13, fontweight="bold"
    )
    kw_sw = dict(color="black", linestyle=":", lw=1.5)

    # Panel 1: T curves
    ax = axes[0]
    for i, (name, T_curve) in enumerate(schedules.items()):
        ax.plot(epochs, T_curve, color=cmap(i / len(schedules)),
                lw=1.0, alpha=0.45, label=name)
    ax.plot(epochs, nh_T_per_epoch, color="red", lw=2.5, zorder=10,
            label="Nose-Hoover (auto)")
    ax.axvline(switch, **kw_sw, label="switch")
    ax.set_title("Temperature schedules")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("T")
    ax.legend(fontsize=5, ncol=2)

    # Panel 2: full learning curves
    ax = axes[1]
    for i, (name, errs) in enumerate(results.items()):
        ax.plot(epochs, errs, color=cmap(i / len(schedules)), lw=0.9, alpha=0.45)
    ax.plot(epochs, nh_errors, color="red", lw=2.5, zorder=10, label="Nose-Hoover")
    ax.axvline(switch, **kw_sw)
    ax.set_title("Reconstruction error (full)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)

    # Panel 3: phase-2 learning curves (zoomed)
    ax = axes[2]
    ph2_epochs = list(range(N_EPOCHS_PHASE + 1, N_EPOCHS + 1))
    for i, (name, errs) in enumerate(results.items()):
        ax.plot(ph2_epochs, errs[N_EPOCHS_PHASE:],
                color=cmap(i / len(schedules)), lw=0.9, alpha=0.45, label=name)
    ax.plot(ph2_epochs, nh_errors[N_EPOCHS_PHASE:],
            color="red", lw=2.5, zorder=10, label="Nose-Hoover")
    ax.set_title("Phase-2 error (FashionMNIST, zoomed)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=5, ncol=2)

    # Panel 4: phase-2 mean error bar chart (recovery speed)
    ax = axes[3]
    ph2_mean = {name: float(np.mean(errs[N_EPOCHS_PHASE:]))
                for name, errs in results.items()}
    ph2_mean["Nose-Hoover"] = float(np.mean(nh_errors[N_EPOCHS_PHASE:]))

    sorted_names = sorted(ph2_mean, key=ph2_mean.get)
    sorted_vals  = [ph2_mean[n] for n in sorted_names]
    bar_colors   = ["red" if n == "Nose-Hoover" else "steelblue" for n in sorted_names]

    ax.barh(sorted_names, sorted_vals, color=bar_colors, alpha=0.75)
    nh_rank = sorted_names.index("Nose-Hoover") + 1
    ax.set_title(f"Phase-2 mean error — NH ranks {nh_rank}/{len(sorted_names)}")
    ax.set_xlabel("Mean MSE (epochs 11-20)")
    ax.invert_yaxis()

    plt.tight_layout()
    out = OUTPUT_DIR / "rbm_schedule_grid_nonstationary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out}")

    return sorted_names, sorted_vals, nh_rank, ph2_mean


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    loader1, loader2 = make_loaders()
    schedules        = make_schedules(N_EPOCHS)
    print(f"\n{len(schedules)} schedules to evaluate + NH\n")

    results = {}
    for name, T_curve in schedules.items():
        errs = train_scheduled(T_curve, loader1, loader2, device)
        results[name] = errs
        ph2_mean = float(np.mean(errs[N_EPOCHS_PHASE:]))
        print(f"  {name:25s}  ph2 mean {ph2_mean:.4f}  "
              f"T: {T_curve[0]:.2f}->{T_curve[N_EPOCHS_PHASE]:.2f}->{T_curve[-1]:.2f}")

    print("\nRunning Nose-Hoover...")
    nh_errors, nh_T_per_epoch, nh_model = train_nh(loader1, loader2, device)
    nh_ph2_mean = float(np.mean(nh_errors[N_EPOCHS_PHASE:]))
    print(f"  {'Nose-Hoover':25s}  ph2 mean {nh_ph2_mean:.4f}  "
          f"T: {nh_T_per_epoch[0]:.2f}->{nh_T_per_epoch[N_EPOCHS_PHASE]:.2f}"
          f"->{nh_T_per_epoch[-1]:.2f}")

    sorted_names, sorted_vals, nh_rank, ph2_mean = plot(
        results, schedules, nh_errors, nh_T_per_epoch
    )

    print(f"\nPhase-2 mean error summary")
    print(f"  NH rank:        {nh_rank} / {len(sorted_names)}")
    print(f"  Best schedule:  {sorted_names[0]}  ({sorted_vals[0]:.4f})")
    print(f"  NH:             {nh_ph2_mean:.4f}")
    print(f"  Worst schedule: {sorted_names[-1]}  ({sorted_vals[-1]:.4f})")

    gap_to_best = (nh_ph2_mean - sorted_vals[0]) / sorted_vals[0] * 100
    print(f"  Gap to best:  {gap_to_best:+.1f}%")

    # Show T trajectory around the switch
    print(f"\nNH temperature around switch:")
    for i in range(N_EPOCHS_PHASE - 2, N_EPOCHS_PHASE + 3):
        phase = "MNIST  " if i < N_EPOCHS_PHASE else "Fashion"
        print(f"  epoch {i+1:2d} [{phase}]  T={nh_T_per_epoch[i]:.4f}")
