"""
Non-stationary experiment: sequential class split
==================================================
Phase 1 (epochs 1-10):  train on MNIST digits 0-4
Phase 2 (epochs 11-20): switch to digits 5-9 without warning

Three models, identical init, identical data order:
  Classical   -- T=1 throughout
  Scheduled   -- replays the exact per-batch T(t) that NH produced
  Nose-Hoover -- adaptive feedback each batch

Key question: after the class switch, does NH recover faster than
Scheduled?  If so, the adaptive feedback is doing real work that a
fixed schedule cannot replicate.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from rbm_nose_hoover import RBM_NoseHoover

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED            = 42
N_VISIBLE       = 784
N_HIDDEN        = 256
N_EPOCHS_PHASE  = 10       # epochs per phase (20 total)
LR              = 0.01
K               = 1
BATCH_SIZE      = 64
H_TARGET        = 0.3


# ──────────────────────────────────────────────────────────────
# Data: two subsets by class
# ──────────────────────────────────────────────────────────────

def make_loaders():
    transform  = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="./data", train=True,
                                download=True, transform=transform)

    idx_0_4 = [i for i, (_, y) in enumerate(train_data) if y < 5]
    idx_5_9 = [i for i, (_, y) in enumerate(train_data) if y >= 5]

    def loader(indices):
        return DataLoader(
            Subset(train_data, indices),
            batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )

    print(f"Phase-1 samples (0-4): {len(idx_0_4)}")
    print(f"Phase-2 samples (5-9): {len(idx_5_9)}")
    return loader(idx_0_4), loader(idx_5_9)


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
        self.h_activation_history = []

    def _ph(self, v):
        return torch.sigmoid((v @ self.W + self.c) / self.temperature)

    def _pv(self, h):
        return torch.sigmoid((h @ self.W.t() + self.b) / self.temperature)

    def cd(self, v0):
        with torch.no_grad():
            p_h0 = self._ph(v0)
            h    = torch.bernoulli(p_h0)
            for _ in range(K):
                p_v = self._pv(h)
                v   = torch.bernoulli(p_v)
                p_hk = self._ph(v)
                h    = torch.bernoulli(p_hk)
            bs  = v0.size(0)
            self.W += LR * (v0.t() @ p_h0 / bs - v.t() @ p_hk / bs)
            self.b += LR * (v0 - v).mean(0)
            self.c += LR * (p_h0 - p_hk).mean(0)
        self.h_activation_history.append(p_h0.mean().item())
        return ((v0 - p_v) ** 2).mean().item()


# ──────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────

def binarise(batch, device):
    return (batch.view(batch.size(0), -1).to(device) > 0.5).float()


def train_epoch(model, loader, device, label="", epoch=0):
    errs = []
    for batch, _ in loader:
        errs.append(model.cd(binarise(batch, device)))
    avg = float(np.mean(errs))
    t   = model.temperature if hasattr(model, "temperature") else 1.0
    h   = model.h_activation_history[-1]
    print(f"  [{label}] epoch {epoch:2d} | error {avg:.4f} | T {t:.4f} | h_avg {h:.3f}")
    return avg


def train_nh_epoch(model, loader, device, label="", epoch=0):
    errs = []
    for batch, _ in loader:
        v0 = binarise(batch, device)
        errs.append(model.contrastive_divergence(v0, k=K, lr=LR))
    avg = float(np.mean(errs))
    print(f"  [{label}] epoch {epoch:2d} | error {avg:.4f} | T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")
    return avg


# ──────────────────────────────────────────────────────────────
# Pass 1 — NH: adaptive, records T schedule
# ──────────────────────────────────────────────────────────────

def run_nh(loader1, loader2, device):
    model = RBM_NoseHoover(
        N_VISIBLE, N_HIDDEN, T0=1.0, h_target=H_TARGET, Q=10.0, alpha=0.01
    )
    model.to(device)
    errors = []

    print("\nNose-Hoover")
    for ep in range(1, N_EPOCHS_PHASE + 1):
        errors.append(train_nh_epoch(model, loader1, device, "NH ph1", ep))
    for ep in range(N_EPOCHS_PHASE + 1, 2 * N_EPOCHS_PHASE + 1):
        errors.append(train_nh_epoch(model, loader2, device, "NH ph2", ep))

    return model, errors, list(model.temperature_history)


# ──────────────────────────────────────────────────────────────
# Pass 2 — Scheduled: fixed NH schedule, no feedback
# ──────────────────────────────────────────────────────────────

def run_scheduled(loader1, loader2, schedule, device):
    model = RBM(device)
    errors, step = [], 0

    print("\nScheduled (NH T replayed, no feedback)")
    for ep in range(1, N_EPOCHS_PHASE + 1):
        for batch, _ in loader1:
            model.temperature = schedule[step] if step < len(schedule) else schedule[-1]
            model.cd(binarise(batch, device))
            step += 1
        errors.append(float(np.mean(model.h_activation_history[-len(loader1):])))
        # rebuild per-epoch error by re-using h history as proxy isn't great;
        # track errors separately
    # redo cleanly
    model  = RBM(device)
    errors = []
    step   = 0

    for ep in range(1, N_EPOCHS_PHASE + 1):
        errs = []
        for batch, _ in loader1:
            model.temperature = schedule[step] if step < len(schedule) else schedule[-1]
            errs.append(model.cd(binarise(batch, device)))
            step += 1
        errors.append(float(np.mean(errs)))
        print(f"  [Sched ph1] epoch {ep:2d} | error {errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")

    for ep in range(N_EPOCHS_PHASE + 1, 2 * N_EPOCHS_PHASE + 1):
        errs = []
        for batch, _ in loader2:
            model.temperature = schedule[step] if step < len(schedule) else schedule[-1]
            errs.append(model.cd(binarise(batch, device)))
            step += 1
        errors.append(float(np.mean(errs)))
        print(f"  [Sched ph2] epoch {ep:2d} | error {errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")

    return model, errors


# ──────────────────────────────────────────────────────────────
# Pass 3 — Classical: T=1 throughout
# ──────────────────────────────────────────────────────────────

def run_classical(loader1, loader2, device):
    model  = RBM(device)
    errors = []

    print("\nClassical (T=1)")
    for ep in range(1, N_EPOCHS_PHASE + 1):
        errors.append(train_epoch(model, loader1, device, "Cls ph1", ep))
    for ep in range(N_EPOCHS_PHASE + 1, 2 * N_EPOCHS_PHASE + 1):
        errors.append(train_epoch(model, loader2, device, "Cls ph2", ep))

    return model, errors


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def epoch_means(hist, n_epochs):
    chunk = max(1, len(hist) // n_epochs)
    return [float(np.mean(hist[i*chunk:(i+1)*chunk])) for i in range(n_epochs)]


def plot(errors_c, errors_s, errors_nh, nh_model, sched_model, class_model):
    n_total = 2 * N_EPOCHS_PHASE
    epochs  = list(range(1, n_total + 1))
    switch  = N_EPOCHS_PHASE + 0.5    # vertical line position

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        "Non-stationary split: digits 0-4  ->  digits 5-9  (switch at epoch 10)",
        fontsize=12, fontweight="bold"
    )

    kw_switch = dict(color="black", linestyle=":", linewidth=1.5, label="class switch")

    # Reconstruction error
    ax = axes[0]
    ax.plot(epochs, errors_c,  label="Classical (T=1)",     color="steelblue")
    ax.plot(epochs, errors_s,  label="Scheduled (NH T(t))", color="darkorange", linestyle="--")
    ax.plot(epochs, errors_nh, label="Nose-Hoover",          color="firebrick")
    ax.axvline(switch, **kw_switch)
    ax.set_title("Reconstruction Error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)

    # Hidden activation
    ax = axes[1]
    ax.plot(epochs, epoch_means(class_model.h_activation_history, n_total),
            label="Classical",  color="steelblue")
    ax.plot(epochs, epoch_means(sched_model.h_activation_history, n_total),
            label="Scheduled",  color="darkorange", linestyle="--")
    ax.plot(epochs, epoch_means(nh_model.h_activation_history, n_total),
            label="Nose-Hoover", color="firebrick")
    ax.axhline(H_TARGET, linestyle=":", color="gray", label=f"target={H_TARGET}")
    ax.axvline(switch, **kw_switch)
    ax.set_title("Avg Hidden Activation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("<h>")
    ax.legend(fontsize=8)

    # Temperature
    ax = axes[2]
    ax.plot(epochs, epoch_means(nh_model.temperature_history, n_total),
            label="NH / Scheduled T", color="firebrick")
    ax.axhline(1.0, linestyle="--", color="steelblue", label="Classical T=1")
    ax.axvline(switch, **kw_switch)
    ax.set_title("Temperature")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("T")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "rbm_nonstationary.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved to {out}")


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────

def summarise(errors_c, errors_s, errors_nh):
    n = N_EPOCHS_PHASE
    print("\nPhase-1 final error (last epoch before switch)")
    print(f"  Classical   {errors_c[n-1]:.4f}")
    print(f"  Scheduled   {errors_s[n-1]:.4f}")
    print(f"  Nose-Hoover {errors_nh[n-1]:.4f}")

    print("\nPhase-2 first epoch error (epoch right after switch)")
    print(f"  Classical   {errors_c[n]:.4f}")
    print(f"  Scheduled   {errors_s[n]:.4f}")
    print(f"  Nose-Hoover {errors_nh[n]:.4f}")

    print("\nPhase-2 final error (last epoch)")
    print(f"  Classical   {errors_c[-1]:.4f}")
    print(f"  Scheduled   {errors_s[-1]:.4f}")
    print(f"  Nose-Hoover {errors_nh[-1]:.4f}")

    # Recovery: area under error curve in phase 2 (lower = faster recovery)
    auc_c  = float(np.mean(errors_c[n:]))
    auc_s  = float(np.mean(errors_s[n:]))
    auc_nh = float(np.mean(errors_nh[n:]))
    print("\nPhase-2 mean error (proxy for recovery speed, lower = better)")
    print(f"  Classical   {auc_c:.4f}")
    print(f"  Scheduled   {auc_s:.4f}")
    print(f"  Nose-Hoover {auc_nh:.4f}")

    gap = (auc_s - auc_nh) / auc_s * 100
    print(f"\n  Feedback gain over Scheduled in phase 2: {gap:+.1f}%")
    if abs(gap) < 0.5:
        print("  -> NH and Scheduled are equivalent: feedback still adds nothing.")
    elif gap > 0:
        print("  -> NH recovers faster: adaptive feedback earns its keep here.")
    else:
        print("  -> Scheduled beats NH: feedback is actively harmful (gain too high).")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    loader1, loader2 = make_loaders()

    print("\nPass 1 -- Nose-Hoover")
    nh_model, errors_nh, schedule = run_nh(loader1, loader2, device)

    print("\nPass 2 -- Scheduled (NH T replayed)")
    sched_model, errors_s = run_scheduled(loader1, loader2, schedule, device)

    print("\nPass 3 -- Classical")
    class_model, errors_c = run_classical(loader1, loader2, device)

    summarise(errors_c, errors_s, errors_nh)
    plot(errors_c, errors_s, errors_nh, nh_model, sched_model, class_model)
