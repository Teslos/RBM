"""
Reconstruction-error-gradient thermostat for RBMs
==================================================
Replaces the h_avg feedback signal with the smoothed gradient of
reconstruction error.  No h_target needed.

Update rule:
    ema_t  = alpha_ema * ema_{t-1} + (1-alpha_ema) * error_t
    delta  = ema_t - ema_{t-1}          # negative during normal learning
    xi    += gain * delta               # xi falls as model improves → T cools
    T      = T0 * exp(xi)

Behaviour:
  Normal learning : error falls → delta < 0 → xi falls → T cools (auto-annealing)
  Distribution switch: error spikes → delta > 0 → xi rises → T raises (exploration)
  After adaptation: error falls again → xi falls → T cools again

No h_target.  The annealing rate is proportional to how fast learning
is happening; the exploration response is proportional to how large
the disruption is.

Compared against: const T=0.3, const T=1.0, NH (h_avg), and Grad thermostat
on the MNIST->Fashion->MNIST->Fashion multi-switch benchmark (5 seeds).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rbm_nose_hoover import RBM_NoseHoover

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

N_VISIBLE      = 784
N_HIDDEN       = 256
N_EPOCHS_PHASE = 10
N_PHASES       = 4
N_EPOCHS       = N_EPOCHS_PHASE * N_PHASES
LR             = 0.01
K              = 1
BATCH_SIZE     = 64
SEEDS          = [42, 7, 13, 99, 256]
PHASE_NAMES    = ["MNIST", "Fashion", "MNIST", "Fashion"]
PHASE_DATASETS = [0, 1, 0, 1]


# ──────────────────────────────────────────────────────────────
# Gradient thermostat RBM
# ──────────────────────────────────────────────────────────────

class RBM_GradThermostat(nn.Module):
    """
    RBM whose temperature is controlled by the log-error gradient,
    updated once per epoch.  No h_target needed.

    Update rule (called once per epoch with the epoch's mean error):
        delta = log(epoch_error) - log(prev_epoch_error)
        xi   += gain * delta
        T     = T0 * exp(xi)

    Using log-error makes the signal scale-invariant:
      - A 10% error drop always gives delta ≈ -0.10, regardless of
        whether the absolute error is 0.07 or 0.02.
      - A distribution switch that doubles the error gives delta ≈ +0.69,
        a bounded, interpretable signal.

    Parameters
    ----------
    T0       : baseline temperature
    gain     : scale of xi response per epoch (default 1.0)
    xi_clamp : symmetric clamp on xi
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        T0: float = 1.0,
        gain: float = 1.0,
        xi_clamp: float = 2.0,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_visible))
        self.c = nn.Parameter(torch.zeros(n_hidden))

        self.T0       = T0
        self.gain     = gain
        self.xi_clamp = xi_clamp

        self.xi             = 0.0
        self.prev_log_error = None   # set on first epoch_step call

        self.temperature_history = []
        self.delta_history       = []

    @property
    def temperature(self) -> float:
        return self.T0 * np.exp(self.xi)

    def _ph(self, v):
        return torch.sigmoid((v @ self.W + self.c) / self.temperature)

    def _pv(self, h):
        return torch.sigmoid((h @ self.W.t() + self.b) / self.temperature)

    def contrastive_divergence(self, v0: torch.Tensor, k: int = 1, lr: float = 0.01) -> float:
        """CD-k step — temperature is fixed for the whole epoch."""
        with torch.no_grad():
            p_h0 = self._ph(v0)
            h = torch.bernoulli(p_h0)
            for _ in range(k):
                p_v  = self._pv(h);  v   = torch.bernoulli(p_v)
                p_hk = self._ph(v);  h   = torch.bernoulli(p_hk)
            bs = v0.size(0)
            self.W.data += lr * (v0.t() @ p_h0 / bs - v.t() @ p_hk / bs)
            self.b.data += lr * (v0 - v).mean(0)
            self.c.data += lr * (p_h0 - p_hk).mean(0)
        return ((v0 - p_v) ** 2).mean().item()

    def epoch_step(self, epoch_error: float):
        """Call once per epoch with the epoch's mean reconstruction error."""
        log_err = np.log(epoch_error + 1e-9)

        if self.prev_log_error is None:
            # First epoch: no previous value, just record
            self.prev_log_error = log_err
            self.temperature_history.append(self.temperature)
            self.delta_history.append(0.0)
            return

        delta = log_err - self.prev_log_error   # negative = improving
        self.xi += self.gain * delta
        self.xi  = float(np.clip(self.xi, -self.xi_clamp, self.xi_clamp))
        self.prev_log_error = log_err

        self.temperature_history.append(self.temperature)
        self.delta_history.append(delta)


# ──────────────────────────────────────────────────────────────
# Minimal fixed-T RBM
# ──────────────────────────────────────────────────────────────

class RBM:
    def __init__(self, device, seed):
        g = torch.Generator().manual_seed(seed)
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


# ──────────────────────────────────────────────────────────────
# Data / helpers
# ──────────────────────────────────────────────────────────────

def make_loaders(seed):
    tf      = transforms.Compose([transforms.ToTensor()])
    mnist   = datasets.MNIST(root="./data",        train=True, download=True, transform=tf)
    fashion = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)
    def loader(ds):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    return [loader(mnist), loader(fashion)]


def binarise(batch, device):
    return (batch.view(batch.size(0), -1).to(device) > 0.5).float()


def epoch_means(hist, n_epochs):
    chunk = max(1, len(hist) // n_epochs)
    return [float(np.mean(hist[i*chunk:(i+1)*chunk])) for i in range(n_epochs)]


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_fixed(T_value, loaders, device, seed):
    model = RBM(device, seed)
    model.temperature = T_value
    errors = []
    for ds_idx in PHASE_DATASETS:
        for _ in range(N_EPOCHS_PHASE):
            errs = [model.cd(binarise(b, device)) for b, _ in loaders[ds_idx]]
            errors.append(float(np.mean(errs)))
    return errors


def train_nh(loaders, device, seed):
    model = RBM_NoseHoover(N_VISIBLE, N_HIDDEN, T0=1.0, h_target=0.3, Q=10.0, alpha=0.01)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.W.data = torch.randn(N_VISIBLE, N_HIDDEN, generator=g).to(device) * 0.01
        model.b.data = torch.zeros(N_VISIBLE).to(device)
        model.c.data = torch.zeros(N_HIDDEN).to(device)
    model.to(device)
    errors = []
    for ds_idx in PHASE_DATASETS:
        for _ in range(N_EPOCHS_PHASE):
            errs = [model.contrastive_divergence(binarise(b, device), k=K, lr=LR)
                    for b, _ in loaders[ds_idx]]
            errors.append(float(np.mean(errs)))
    n_per_epoch = len(model.temperature_history) // N_EPOCHS
    T_ep = [float(np.mean(model.temperature_history[i*n_per_epoch:(i+1)*n_per_epoch]))
            for i in range(N_EPOCHS)]
    return errors, T_ep


def train_grad(loaders, device, seed, gain=1.0, xi_clamp=2.0):
    model = RBM_GradThermostat(N_VISIBLE, N_HIDDEN, T0=1.0, gain=gain, xi_clamp=xi_clamp)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.W.data = torch.randn(N_VISIBLE, N_HIDDEN, generator=g).to(device) * 0.01
        model.b.data = torch.zeros(N_VISIBLE).to(device)
        model.c.data = torch.zeros(N_HIDDEN).to(device)
    model.to(device)
    errors = []
    for ds_idx in PHASE_DATASETS:
        for _ in range(N_EPOCHS_PHASE):
            errs = [model.contrastive_divergence(binarise(b, device), k=K, lr=LR)
                    for b, _ in loaders[ds_idx]]
            avg_error = float(np.mean(errs))
            errors.append(avg_error)
            model.epoch_step(avg_error)
    return errors, list(model.temperature_history)


# ──────────────────────────────────────────────────────────────
# Multi-seed run
# ──────────────────────────────────────────────────────────────

def run_all(device):
    results   = {"const T=0.3": [], "const T=1.0": [],
                 "NH (h_avg)": [],  "Grad thermostat": []}
    T_curves  = {"NH (h_avg)": [], "Grad thermostat": []}

    for i, seed in enumerate(SEEDS):
        print(f"\nSeed {seed} ({i+1}/{len(SEEDS)})")
        loaders = make_loaders(seed)

        e = train_fixed(0.3, loaders, device, seed)
        results["const T=0.3"].append(e)
        ph = [np.mean(e[p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE]) for p in range(N_PHASES)]
        print(f"  const T=0.3       phases: " + "  ".join(f"{x:.4f}" for x in ph))

        e = train_fixed(1.0, loaders, device, seed)
        results["const T=1.0"].append(e)
        ph = [np.mean(e[p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE]) for p in range(N_PHASES)]
        print(f"  const T=1.0       phases: " + "  ".join(f"{x:.4f}" for x in ph))

        e, T_ep = train_nh(loaders, device, seed)
        results["NH (h_avg)"].append(e)
        T_curves["NH (h_avg)"].append(T_ep)
        ph = [np.mean(e[p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE]) for p in range(N_PHASES)]
        print(f"  NH (h_avg)        phases: " + "  ".join(f"{x:.4f}" for x in ph)
              + f"  T_end={T_ep[-1]:.3f}")

        e, T_ep = train_grad(loaders, device, seed)
        results["Grad thermostat"].append(e)
        T_curves["Grad thermostat"].append(T_ep)
        ph = [np.mean(e[p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE]) for p in range(N_PHASES)]
        print(f"  Grad thermostat   phases: " + "  ".join(f"{x:.4f}" for x in ph)
              + f"  T_end={T_ep[-1]:.3f}")

    return results, T_curves


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────

def summarise(results):
    names = list(results.keys())
    print("\nPer-phase mean error (mean ± std across seeds)")
    header = f"{'Schedule':20s}" + "".join(
        f"  Ph{p+1} {PHASE_NAMES[p]:7s}" for p in range(N_PHASES)) + "  Overall"
    print(header)
    print("-" * len(header))

    overall = {}
    for name in names:
        arr = np.array(results[name])
        means, stds = [], []
        for p in range(N_PHASES):
            chunk = arr[:, p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE].mean(axis=1)
            means.append(chunk.mean()); stds.append(chunk.std())
        ov = np.mean(means)
        overall[name] = ov
        row = f"{name:20s}" + "".join(f"  {m:.4f}±{s:.4f}" for m, s in zip(means, stds))
        print(row + f"  {ov:.4f}")

    print("\nOverall ranking:")
    for rank, name in enumerate(sorted(overall, key=overall.get), 1):
        tag = " <-- GRAD" if name == "Grad thermostat" else \
              " <-- NH"   if name == "NH (h_avg)" else ""
        print(f"  {rank}. {name:20s}  {overall[name]:.4f}{tag}")


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot(results, T_curves):
    epochs = np.arange(1, N_EPOCHS + 1)
    boundaries = [N_EPOCHS_PHASE * (p+1) + 0.5 for p in range(N_PHASES - 1)]
    kw_sw = dict(color="black", linestyle=":", lw=1.2, alpha=0.5)

    COLORS = {"const T=0.3":    "steelblue",
              "const T=1.0":    "lightsteelblue",
              "NH (h_avg)":     "darkorange",
              "Grad thermostat":"firebrick"}
    LW     = {"const T=0.3":    1.2, "const T=1.0": 1.2,
              "NH (h_avg)":     1.8, "Grad thermostat": 2.5}

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        "Error-gradient thermostat vs NH vs fixed T  "
        f"(MNIST->Fashion x2, {len(SEEDS)} seeds)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: learning curves
    ax = axes[0]
    for name, runs in results.items():
        arr  = np.array(runs)
        mean = arr.mean(0);  std = arr.std(0)
        ax.plot(epochs, mean, label=name, color=COLORS[name], lw=LW[name])
        ax.fill_between(epochs, mean-std, mean+std, color=COLORS[name], alpha=0.12)
    for xv in boundaries: ax.axvline(xv, **kw_sw)
    for p, pn in enumerate(PHASE_NAMES):
        ax.text(p*N_EPOCHS_PHASE + N_EPOCHS_PHASE/2, 0.002, pn,
                ha="center", va="bottom", fontsize=7, color="gray")
    ax.set_title("Reconstruction error (mean ± 1σ)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend(fontsize=8)

    # Panel 2: temperature trajectories
    ax = axes[1]
    for name, T_runs in T_curves.items():
        arr  = np.array(T_runs)
        mean = arr.mean(0);  std = arr.std(0)
        ax.plot(epochs, mean, label=name, color=COLORS[name], lw=LW[name])
        ax.fill_between(epochs, mean-std, mean+std, color=COLORS[name], alpha=0.15)
    ax.axhline(0.3, linestyle="--", color="steelblue", lw=1, alpha=0.6, label="T=0.3 ref")
    ax.axhline(1.0, linestyle=":",  color="gray",      lw=1, alpha=0.5, label="T=1.0 ref")
    for xv in boundaries: ax.axvline(xv, **kw_sw)
    ax.set_title("Temperature trajectories")
    ax.set_xlabel("Epoch"); ax.set_ylabel("T"); ax.legend(fontsize=8)

    # Panel 3: per-phase bar chart
    ax = axes[2]
    names = list(results.keys())
    x     = np.arange(N_PHASES)
    w     = 0.18
    offs  = np.linspace(-(len(names)-1)/2, (len(names)-1)/2, len(names)) * w
    for i, name in enumerate(names):
        arr = np.array(results[name])
        means = [arr[:, p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE].mean(axis=1).mean()
                 for p in range(N_PHASES)]
        stds  = [arr[:, p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE].mean(axis=1).std()
                 for p in range(N_PHASES)]
        ax.bar(x + offs[i], means, w*0.9, yerr=stds, label=name,
               color=COLORS[name], alpha=0.8, error_kw=dict(capsize=2, elinewidth=1))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ph{p+1}\n{PHASE_NAMES[p]}" for p in range(N_PHASES)])
    ax.set_title("Per-phase mean error ± 1σ")
    ax.set_ylabel("Mean MSE"); ax.legend(fontsize=7)

    plt.tight_layout()
    out = OUTPUT_DIR / "rbm_grad_thermostat.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Phases: {' -> '.join(PHASE_NAMES)}")
    print(f"Seeds:  {SEEDS}")

    results, T_curves = run_all(device)
    summarise(results)
    plot(results, T_curves)
