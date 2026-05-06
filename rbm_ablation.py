"""
Annealing ablation: NH thermostat vs fixed schedule vs T=1
==========================================================
Three models, identical weight init, identical data order:

  1. Classical    -- T fixed at 1.0 throughout
  2. Scheduled    -- T follows the exact per-batch schedule NH produced
  3. Nose-Hoover  -- T adapts via feedback each batch

If Scheduled matches NH, the adaptive feedback adds nothing beyond annealing.
If NH beats Scheduled, the feedback loop earns its keep.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
# RBM with injectable temperature
# ──────────────────────────────────────────────────────────────

class RBM:
    """
    Minimal RBM whose temperature can be set externally each batch.
    temperature=None means T=1 (classical).
    """
    def __init__(self, n_visible, n_hidden, device):
        g = torch.Generator().manual_seed(SEED)
        self.W = (torch.randn(n_visible, n_hidden, generator=g) * 0.01).to(device)
        self.b = torch.zeros(n_visible).to(device)
        self.c = torch.zeros(n_hidden).to(device)
        self.device = device
        self.h_activation_history = []
        self.temperature = 1.0

    def _prob_h(self, v):
        return torch.sigmoid((v @ self.W + self.c) / self.temperature)

    def _prob_v(self, h):
        return torch.sigmoid((h @ self.W.t() + self.b) / self.temperature)

    def cd(self, v0, lr=LR, k=K):
        with torch.no_grad():
            p_h0 = self._prob_h(v0)
            h = torch.bernoulli(p_h0)
            for _ in range(k):
                p_v = self._prob_v(h)
                v   = torch.bernoulli(p_v)
                p_hk = self._prob_h(v)
                h    = torch.bernoulli(p_hk)

            bs  = v0.size(0)
            pos = v0.t() @ p_h0 / bs
            neg = v.t()  @ p_hk / bs

            self.W += lr * (pos - neg)
            self.b += lr * (v0 - v).mean(0)
            self.c += lr * (p_h0 - p_hk).mean(0)

        self.h_activation_history.append(p_h0.mean().item())
        return ((v0 - p_v) ** 2).mean().item()


# ──────────────────────────────────────────────────────────────
# Pass 1 -- NH: record per-batch temperature schedule
# ──────────────────────────────────────────────────────────────

def run_nh(loader, device):
    torch.manual_seed(SEED)
    model = RBM_NoseHoover(
        N_VISIBLE, N_HIDDEN, T0=1.0, h_target=H_TARGET, Q=10.0, alpha=0.01
    )
    model.to(device)

    epoch_errors = []
    for epoch in range(N_EPOCHS):
        errs = []
        for batch, _ in loader:
            v0 = (batch.view(batch.size(0), -1).to(device) > 0.5).float()
            errs.append(model.contrastive_divergence(v0, k=K, lr=LR))
        epoch_errors.append(float(np.mean(errs)))
        print(f"  [NH]        epoch {epoch+1:2d} | error {epoch_errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")

    # temperature_history has one entry per batch
    return model, epoch_errors, list(model.temperature_history)


# ──────────────────────────────────────────────────────────────
# Pass 2 -- Scheduled: replay NH's T(t) batch by batch
# ──────────────────────────────────────────────────────────────

def run_scheduled(loader, schedule, device):
    model = RBM(N_VISIBLE, N_HIDDEN, device)
    epoch_errors = []
    step = 0
    for epoch in range(N_EPOCHS):
        errs = []
        for batch, _ in loader:
            v0 = (batch.view(batch.size(0), -1).to(device) > 0.5).float()
            model.temperature = schedule[step] if step < len(schedule) else schedule[-1]
            errs.append(model.cd(v0))
            step += 1
        epoch_errors.append(float(np.mean(errs)))
        print(f"  [Scheduled] epoch {epoch+1:2d} | error {epoch_errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")
    return model, epoch_errors


# ──────────────────────────────────────────────────────────────
# Pass 3 -- Classical: T = 1 throughout
# ──────────────────────────────────────────────────────────────

def run_classical(loader, device):
    model = RBM(N_VISIBLE, N_HIDDEN, device)
    epoch_errors = []
    for epoch in range(N_EPOCHS):
        errs = []
        for batch, _ in loader:
            v0 = (batch.view(batch.size(0), -1).to(device) > 0.5).float()
            errs.append(model.cd(v0))
        epoch_errors.append(float(np.mean(errs)))
        print(f"  [Classical] epoch {epoch+1:2d} | error {epoch_errors[-1]:.4f} | "
              f"T {model.temperature:.4f} | h_avg {model.h_activation_history[-1]:.3f}")
    return model, epoch_errors


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def epoch_means(hist, n_epochs):
    chunk = max(1, len(hist) // n_epochs)
    return [float(np.mean(hist[i*chunk:(i+1)*chunk])) for i in range(n_epochs)]


def plot(errors_c, errors_s, errors_nh, nh_model, sched_model, class_model):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Annealing Ablation: does adaptive feedback matter?", fontsize=13)

    epochs = range(1, N_EPOCHS + 1)

    # Reconstruction error
    axes[0].plot(epochs, errors_c,  label="Classical (T=1)",    color="steelblue")
    axes[0].plot(epochs, errors_s,  label="Scheduled (NH T(t))", color="darkorange", linestyle="--")
    axes[0].plot(epochs, errors_nh, label="Nose-Hoover",         color="firebrick")
    axes[0].set_title("Reconstruction Error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    # Hidden activation
    n = N_EPOCHS
    axes[1].plot(epochs, epoch_means(class_model.h_activation_history, n),
                 label="Classical",  color="steelblue")
    axes[1].plot(epochs, epoch_means(sched_model.h_activation_history, n),
                 label="Scheduled",  color="darkorange", linestyle="--")
    axes[1].plot(epochs, epoch_means(nh_model.h_activation_history, n),
                 label="Nose-Hoover", color="firebrick")
    axes[1].axhline(H_TARGET, linestyle=":", color="gray", label=f"target={H_TARGET}")
    axes[1].set_title("Avg Hidden Activation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("<h>")
    axes[1].legend()

    # Temperature trajectories
    axes[2].plot(epoch_means(nh_model.temperature_history, n),
                 label="Nose-Hoover T", color="firebrick")
    axes[2].plot(epoch_means(sched_model.h_activation_history, n),  # placeholder x-axis
                 [nh_model.temperature_history[0]] * n,
                 label="Scheduled T (same)", color="darkorange", linestyle="--", alpha=0.5)
    axes[2].clear()
    axes[2].plot(epochs, epoch_means(nh_model.temperature_history, n),
                 label="NH / Scheduled T", color="firebrick")
    axes[2].axhline(1.0, linestyle="--", color="steelblue", label="Classical T=1")
    axes[2].set_title("Temperature Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("T")
    axes[2].legend()

    plt.tight_layout()
    out = OUTPUT_DIR / "rbm_ablation.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved to {out}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    transform  = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    def make_loader():
        return DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )

    print("Pass 1 -- Nose-Hoover (record T schedule)")
    nh_model, errors_nh, schedule = run_nh(make_loader(), device)

    print("\nPass 2 -- Scheduled (replay NH T schedule, no feedback)")
    sched_model, errors_s = run_scheduled(make_loader(), schedule, device)

    print("\nPass 3 -- Classical (T=1 fixed)")
    class_model, errors_c = run_classical(make_loader(), device)

    print("\nFinal errors")
    print(f"  Classical   {errors_c[-1]:.4f}")
    print(f"  Scheduled   {errors_s[-1]:.4f}")
    print(f"  Nose-Hoover {errors_nh[-1]:.4f}")

    gap_sched_vs_nh = (errors_s[-1] - errors_nh[-1]) / errors_s[-1] * 100
    gap_class_vs_sched = (errors_c[-1] - errors_s[-1]) / errors_c[-1] * 100
    print(f"\n  Annealing gain (Classical->Scheduled): {gap_class_vs_sched:+.1f}%")
    print(f"  Feedback  gain (Scheduled->NH):         {gap_sched_vs_nh:+.1f}%")

    plot(errors_c, errors_s, errors_nh, nh_model, sched_model, class_model)
