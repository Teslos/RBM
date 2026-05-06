"""
Non-stationary experiment: MNIST -> FashionMNIST hard switch
=============================================================
Phase 1 (epochs 1-10):  train on MNIST
Phase 2 (epochs 11-20): switch to FashionMNIST without warning

Both datasets are 28x28 binary images but with meaningfully different
activation statistics (digit strokes vs clothing textures), so the
distribution shift is much sharper than the digit-split experiment.

Three models, identical init, identical data order:
  Classical   -- T=1 throughout
  Scheduled   -- replays the exact per-batch T(t) that NH produced
  Nose-Hoover -- adaptive feedback each batch

Key question: does NH's T(t) curve diverge from the pre-computed
schedule after the switch, and does that divergence help?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
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
LR             = 0.01
K              = 1
BATCH_SIZE     = 64
H_TARGET       = 0.3


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────

def make_loaders():
    tf = transforms.Compose([transforms.ToTensor()])

    mnist   = datasets.MNIST(root="./data",        train=True, download=True, transform=tf)
    fashion = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)

    def loader(ds):
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )

    print(f"Phase-1 (MNIST):        {len(mnist)} samples")
    print(f"Phase-2 (FashionMNIST): {len(fashion)} samples")
    return loader(mnist), loader(fashion)


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
                p_v  = self._pv(h)
                v    = torch.bernoulli(p_v)
                p_hk = self._ph(v)
                h    = torch.bernoulli(p_hk)
            bs = v0.size(0)
            self.W += LR * (v0.t() @ p_h0 / bs - v.t() @ p_hk / bs)
            self.b += LR * (v0 - v).mean(0)
            self.c += LR * (p_h0 - p_hk).mean(0)
        self.h_activation_history.append(p_h0.mean().item())
        return ((v0 - p_v) ** 2).mean().item()


def binarise(batch, device):
    return (batch.view(batch.size(0), -1).to(device) > 0.5).float()


# ──────────────────────────────────────────────────────────────
# Pass 1 — NH
# ──────────────────────────────────────────────────────────────

def run_nh(loader1, loader2, device):
    model = RBM_NoseHoover(
        N_VISIBLE, N_HIDDEN, T0=1.0, h_target=H_TARGET, Q=10.0, alpha=0.01
    )
    model.to(device)
    errors = []

    print("\nNose-Hoover")
    for ep in range(1, N_EPOCHS_PHASE + 1):
        errs = []
        for batch, _ in loader1:
            errs.append(model.contrastive_divergence(binarise(batch, device), k=K, lr=LR))
        errors.append(float(np.mean(errs)))
        print(f"  [NH  MNIST ] epoch {ep:2d} | error {errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")

    for ep in range(N_EPOCHS_PHASE + 1, 2 * N_EPOCHS_PHASE + 1):
        errs = []
        for batch, _ in loader2:
            errs.append(model.contrastive_divergence(binarise(batch, device), k=K, lr=LR))
        errors.append(float(np.mean(errs)))
        print(f"  [NH  Fashion] epoch {ep:2d} | error {errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")

    return model, errors, list(model.temperature_history)


# ──────────────────────────────────────────────────────────────
# Pass 2 — Scheduled
# ──────────────────────────────────────────────────────────────

def run_scheduled(loader1, loader2, schedule, device):
    model  = RBM(device)
    errors = []
    step   = 0

    print("\nScheduled (NH T replayed)")
    for ep in range(1, N_EPOCHS_PHASE + 1):
        errs = []
        for batch, _ in loader1:
            model.temperature = schedule[step] if step < len(schedule) else schedule[-1]
            errs.append(model.cd(binarise(batch, device)))
            step += 1
        errors.append(float(np.mean(errs)))
        print(f"  [Sch MNIST ] epoch {ep:2d} | error {errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")

    for ep in range(N_EPOCHS_PHASE + 1, 2 * N_EPOCHS_PHASE + 1):
        errs = []
        for batch, _ in loader2:
            model.temperature = schedule[step] if step < len(schedule) else schedule[-1]
            errs.append(model.cd(binarise(batch, device)))
            step += 1
        errors.append(float(np.mean(errs)))
        print(f"  [Sch Fashion] epoch {ep:2d} | error {errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")

    return model, errors


# ──────────────────────────────────────────────────────────────
# Pass 3 — Classical
# ──────────────────────────────────────────────────────────────

def run_classical(loader1, loader2, device):
    model  = RBM(device)
    errors = []

    print("\nClassical (T=1)")
    for ep in range(1, N_EPOCHS_PHASE + 1):
        errs = []
        for batch, _ in loader1:
            errs.append(model.cd(binarise(batch, device)))
        errors.append(float(np.mean(errs)))
        print(f"  [Cls MNIST ] epoch {ep:2d} | error {errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")

    for ep in range(N_EPOCHS_PHASE + 1, 2 * N_EPOCHS_PHASE + 1):
        errs = []
        for batch, _ in loader2:
            errs.append(model.cd(binarise(batch, device)))
        errors.append(float(np.mean(errs)))
        print(f"  [Cls Fashion] epoch {ep:2d} | error {errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")

    return model, errors


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def epoch_means(hist, n):
    chunk = max(1, len(hist) // n)
    return [float(np.mean(hist[i*chunk:(i+1)*chunk])) for i in range(n)]


def plot(errors_c, errors_s, errors_nh, nh_model, sched_model, class_model, schedule):
    n_total = 2 * N_EPOCHS_PHASE
    epochs  = list(range(1, n_total + 1))
    switch  = N_EPOCHS_PHASE + 0.5

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(
        "MNIST -> FashionMNIST hard switch (epoch 10)",
        fontsize=12, fontweight="bold"
    )
    kw = dict(color="black", linestyle=":", linewidth=1.5, label="switch")

    # Error
    ax = axes[0]
    ax.plot(epochs, errors_c,  label="Classical (T=1)",     color="steelblue")
    ax.plot(epochs, errors_s,  label="Scheduled (NH T(t))", color="darkorange", linestyle="--")
    ax.plot(epochs, errors_nh, label="Nose-Hoover",          color="firebrick")
    ax.axvline(switch, **kw)
    ax.set_title("Reconstruction Error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)

    # h_avg
    ax = axes[1]
    ax.plot(epochs, epoch_means(class_model.h_activation_history, n_total),
            label="Classical",  color="steelblue")
    ax.plot(epochs, epoch_means(sched_model.h_activation_history, n_total),
            label="Scheduled",  color="darkorange", linestyle="--")
    ax.plot(epochs, epoch_means(nh_model.h_activation_history, n_total),
            label="Nose-Hoover", color="firebrick")
    ax.axhline(H_TARGET, linestyle=":", color="gray", label=f"target={H_TARGET}")
    ax.axvline(switch, **kw)
    ax.set_title("Avg Hidden Activation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("<h>")
    ax.legend(fontsize=8)

    # Temperature — NH vs Scheduled (should diverge after switch if feedback matters)
    ax = axes[2]
    nh_T   = epoch_means(nh_model.temperature_history, n_total)
    # Scheduled T: reconstruct per-epoch mean from schedule
    batches_per_epoch = len(schedule) // n_total
    sched_T = [
        float(np.mean(schedule[i*batches_per_epoch:(i+1)*batches_per_epoch]))
        for i in range(n_total)
    ]
    ax.plot(epochs, nh_T,    label="NH T (adaptive)",      color="firebrick")
    ax.plot(epochs, sched_T, label="Scheduled T (replayed)", color="darkorange",
            linestyle="--", alpha=0.7)
    ax.axhline(1.0, linestyle="--", color="steelblue", label="Classical T=1", alpha=0.5)
    ax.axvline(switch, **kw)
    ax.set_title("Temperature (do they diverge?)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("T")
    ax.legend(fontsize=8)

    # T divergence after switch
    ax = axes[3]
    divergence = [abs(nh_T[i] - sched_T[i]) for i in range(n_total)]
    ax.bar(epochs[:N_EPOCHS_PHASE],  divergence[:N_EPOCHS_PHASE],
           color="steelblue", alpha=0.6, label="Phase 1 (MNIST)")
    ax.bar(epochs[N_EPOCHS_PHASE:], divergence[N_EPOCHS_PHASE:],
           color="firebrick", alpha=0.6, label="Phase 2 (Fashion)")
    ax.axvline(switch, **kw)
    ax.set_title("|NH T - Scheduled T| per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|delta T|")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "rbm_mnist_to_fashion.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved to {out}")


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────

def summarise(errors_c, errors_s, errors_nh, nh_model, schedule):
    n = N_EPOCHS_PHASE
    n_total = 2 * n

    print("\nPhase-1 final error")
    print(f"  Classical   {errors_c[n-1]:.4f}")
    print(f"  Scheduled   {errors_s[n-1]:.4f}")
    print(f"  Nose-Hoover {errors_nh[n-1]:.4f}")

    print("\nPhase-2 first-epoch error (immediately after switch)")
    print(f"  Classical   {errors_c[n]:.4f}")
    print(f"  Scheduled   {errors_s[n]:.4f}")
    print(f"  Nose-Hoover {errors_nh[n]:.4f}")

    print("\nPhase-2 final error")
    print(f"  Classical   {errors_c[-1]:.4f}")
    print(f"  Scheduled   {errors_s[-1]:.4f}")
    print(f"  Nose-Hoover {errors_nh[-1]:.4f}")

    auc_c  = float(np.mean(errors_c[n:]))
    auc_s  = float(np.mean(errors_s[n:]))
    auc_nh = float(np.mean(errors_nh[n:]))
    print("\nPhase-2 mean error (recovery speed, lower = better)")
    print(f"  Classical   {auc_c:.4f}")
    print(f"  Scheduled   {auc_s:.4f}")
    print(f"  Nose-Hoover {auc_nh:.4f}")

    gap = (auc_s - auc_nh) / auc_s * 100
    print(f"\n  Feedback gain over Scheduled in phase 2: {gap:+.1f}%")

    # Check T divergence after switch
    batches_per_epoch = len(schedule) // n_total
    nh_T    = epoch_means(nh_model.temperature_history, n_total)
    sched_T = [
        float(np.mean(schedule[i*batches_per_epoch:(i+1)*batches_per_epoch]))
        for i in range(n_total)
    ]
    mean_div_ph1 = float(np.mean([abs(nh_T[i]-sched_T[i]) for i in range(n)]))
    mean_div_ph2 = float(np.mean([abs(nh_T[i]-sched_T[i]) for i in range(n, n_total)]))
    print(f"\n  Mean |NH T - Scheduled T|  phase 1: {mean_div_ph1:.5f}")
    print(f"  Mean |NH T - Scheduled T|  phase 2: {mean_div_ph2:.5f}")
    print(f"  T divergence ratio (ph2/ph1): {mean_div_ph2/(mean_div_ph1+1e-9):.1f}x")

    if abs(gap) < 0.5:
        verdict = "NH and Scheduled still equivalent -- feedback adds nothing."
    elif gap > 0:
        verdict = "NH recovers faster -- adaptive feedback earns its keep."
    else:
        verdict = "Scheduled beats NH -- gain is too aggressive post-switch."
    print(f"\n  Verdict: {verdict}")


def epoch_means(hist, n):
    chunk = max(1, len(hist) // n)
    return [float(np.mean(hist[i*chunk:(i+1)*chunk])) for i in range(n)]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    loader1, loader2 = make_loaders()

    nh_model, errors_nh, schedule = run_nh(loader1, loader2, device)
    sched_model, errors_s         = run_scheduled(loader1, loader2, schedule, device)
    class_model, errors_c         = run_classical(loader1, loader2, device)

    summarise(errors_c, errors_s, errors_nh, nh_model, schedule)
    plot(errors_c, errors_s, errors_nh, nh_model, sched_model, class_model, schedule)
