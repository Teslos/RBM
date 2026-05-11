"""
Multi-switch experiment: MNIST -> Fashion -> MNIST -> Fashion
=============================================================
4 phases x 10 epochs = 40 epochs total.
5 random seeds for statistical robustness.

Schedules compared:
  const T=0.3  -- oracle low temperature
  const T=0.5  -- moderate constant
  const T=1.0  -- classical baseline
  linear       -- 1.0 -> 0.3 over 40 epochs (can only descend)
  exp r=0.94   -- reaches ~0.1 by epoch 40 (aggressive cooling)
  Nose-Hoover  -- adaptive, Q=10, alpha=0.01

Key question: is there a fixed T that stays competitive across all
4 phases, or does NH's ability to reverse direction give it an
advantage that no monotone schedule can match?
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

N_VISIBLE      = 784
N_HIDDEN       = 256
N_EPOCHS_PHASE = 10
N_PHASES       = 4
N_EPOCHS       = N_EPOCHS_PHASE * N_PHASES   # 40
LR             = 0.01
K              = 1
BATCH_SIZE     = 64
H_TARGET       = 0.3
SEEDS          = [42, 7, 13, 99, 256]

PHASE_NAMES = ["MNIST", "Fashion", "MNIST", "Fashion"]
PHASE_DATASETS = [0, 1, 0, 1]   # index into datasets list


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────

def make_loaders(seed):
    tf      = transforms.Compose([transforms.ToTensor()])
    mnist   = datasets.MNIST(root="./data",        train=True, download=True, transform=tf)
    fashion = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)

    def loader(ds):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    return [loader(mnist), loader(fashion)]


# ──────────────────────────────────────────────────────────────
# Minimal RBM with injectable temperature
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


def binarise(batch, device):
    return (batch.view(batch.size(0), -1).to(device) > 0.5).float()


# ──────────────────────────────────────────────────────────────
# Schedule definitions (per epoch, length = N_EPOCHS)
# ──────────────────────────────────────────────────────────────

def make_schedules():
    n = N_EPOCHS
    return {
        "const T=0.3":  [0.3]  * n,
        "const T=0.5":  [0.5]  * n,
        "const T=1.0":  [1.0]  * n,
        "linear 1->0.3": [1.0 - 0.7 * e / (n - 1) for e in range(n)],
        "exp r=0.94":   [0.94 ** e for e in range(n)],
    }


# ──────────────────────────────────────────────────────────────
# Train one model (fixed schedule or NH) for all phases / seeds
# ──────────────────────────────────────────────────────────────

def train_fixed(schedule, loaders, device, seed):
    """Returns list of per-epoch errors (length N_EPOCHS)."""
    model  = RBM(device, seed)
    errors = []
    for phase_idx, ds_idx in enumerate(PHASE_DATASETS):
        T = schedule[phase_idx * N_EPOCHS_PHASE]  # epoch T at phase start
        for ep in range(N_EPOCHS_PHASE):
            model.temperature = schedule[phase_idx * N_EPOCHS_PHASE + ep]
            errs = [model.cd(binarise(b, device)) for b, _ in loaders[ds_idx]]
            errors.append(float(np.mean(errs)))
    return errors


def train_nh(loaders, device, seed):
    """Returns per-epoch errors and per-epoch mean temperature."""
    model = RBM_NoseHoover(
        N_VISIBLE, N_HIDDEN, T0=1.0, h_target=H_TARGET, Q=10.0, alpha=0.01
    )
    # Match fixed-schedule weight init via same seed
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.W.data = torch.randn(N_VISIBLE, N_HIDDEN, generator=g).to(device) * 0.01
        model.b.data = torch.zeros(N_VISIBLE).to(device)
        model.c.data = torch.zeros(N_HIDDEN).to(device)
    model.to(device)

    errors = []
    for phase_idx, ds_idx in enumerate(PHASE_DATASETS):
        for ep in range(N_EPOCHS_PHASE):
            errs = [model.contrastive_divergence(binarise(b, device), k=K, lr=LR)
                    for b, _ in loaders[ds_idx]]
            errors.append(float(np.mean(errs)))

    n_per_epoch = len(model.temperature_history) // N_EPOCHS
    T_per_epoch = [
        float(np.mean(model.temperature_history[i*n_per_epoch:(i+1)*n_per_epoch]))
        for i in range(N_EPOCHS)
    ]
    return errors, T_per_epoch


# ──────────────────────────────────────────────────────────────
# Main loop over seeds
# ──────────────────────────────────────────────────────────────

def run_all(device):
    schedules = make_schedules()
    schedule_names = list(schedules.keys()) + ["Nose-Hoover"]

    # all_errors[name][seed] = list of N_EPOCHS errors
    all_errors = {n: [] for n in schedule_names}
    # nh T trajectories per seed
    nh_T_all   = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\nSeed {seed} ({seed_idx+1}/{len(SEEDS)})")
        loaders = make_loaders(seed)

        for name, sched in schedules.items():
            errs = train_fixed(sched, loaders, device, seed)
            all_errors[name].append(errs)
            ph_means = [np.mean(errs[p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE])
                        for p in range(N_PHASES)]
            print(f"  {name:18s}  phases: "
                  + "  ".join(f"{m:.4f}" for m in ph_means))

        errs, T_ep = train_nh(loaders, device, seed)
        all_errors["Nose-Hoover"].append(errs)
        nh_T_all.append(T_ep)
        ph_means = [np.mean(errs[p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE])
                    for p in range(N_PHASES)]
        print(f"  {'Nose-Hoover':18s}  phases: "
              + "  ".join(f"{m:.4f}" for m in ph_means)
              + f"  T_end={T_ep[-1]:.3f}")

    return all_errors, nh_T_all, schedules


# ──────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────

def phase_stats(all_errors, name):
    """Per-phase mean ± std across seeds."""
    seeds_errors = all_errors[name]   # list of lists (seed x epoch)
    arr = np.array(seeds_errors)      # (n_seeds, N_EPOCHS)
    means, stds = [], []
    for p in range(N_PHASES):
        chunk = arr[:, p*N_EPOCHS_PHASE:(p+1)*N_EPOCHS_PHASE]
        ph_mean_per_seed = chunk.mean(axis=1)   # (n_seeds,)
        means.append(ph_mean_per_seed.mean())
        stds.append(ph_mean_per_seed.std())
    return np.array(means), np.array(stds)


def epoch_curve(all_errors, name):
    """Mean ± std per epoch across seeds."""
    arr = np.array(all_errors[name])
    return arr.mean(axis=0), arr.std(axis=0)


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot(all_errors, nh_T_all, schedules):
    schedule_names = list(schedules.keys()) + ["Nose-Hoover"]
    epochs = np.arange(1, N_EPOCHS + 1)
    phase_boundaries = [N_EPOCHS_PHASE * (p+1) + 0.5 for p in range(N_PHASES - 1)]

    COLORS = {
        "const T=0.3":   "steelblue",
        "const T=0.5":   "royalblue",
        "const T=1.0":   "lightsteelblue",
        "linear 1->0.3": "darkorange",
        "exp r=0.94":    "goldenrod",
        "Nose-Hoover":   "firebrick",
    }
    STYLES = {
        "const T=0.3":   "-",
        "const T=0.5":   "--",
        "const T=1.0":   ":",
        "linear 1->0.3": "-.",
        "exp r=0.94":    (0, (3,1,1,1)),
        "Nose-Hoover":   "-",
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f"Multi-switch: MNIST->Fashion->MNIST->Fashion  "
        f"({len(SEEDS)} seeds, {N_EPOCHS_PHASE} epochs/phase)",
        fontsize=12, fontweight="bold"
    )
    kw_phase = dict(color="black", linestyle=":", lw=1.2, alpha=0.6)

    # Panel 1: learning curves (mean ± std)
    ax = axes[0]
    for name in schedule_names:
        mean, std = epoch_curve(all_errors, name)
        lw  = 2.5 if name == "Nose-Hoover" else 1.3
        ax.plot(epochs, mean, label=name, color=COLORS[name],
                linestyle=STYLES[name], lw=lw, zorder=3 if name=="Nose-Hoover" else 2)
        ax.fill_between(epochs, mean-std, mean+std,
                        color=COLORS[name], alpha=0.12)
    for xv in phase_boundaries:
        ax.axvline(xv, **kw_phase)
    for p, pname in enumerate(PHASE_NAMES):
        ax.text(p*N_EPOCHS_PHASE + N_EPOCHS_PHASE/2, ax.get_ylim()[1]*0.98,
                pname, ha="center", va="top", fontsize=7, color="gray")
    ax.set_title("Reconstruction error (mean ± 1σ)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=7)

    # Panel 2: per-phase mean error with error bars
    ax = axes[1]
    x = np.arange(N_PHASES)
    width = 0.13
    offsets = np.linspace(-(len(schedule_names)-1)/2,
                          (len(schedule_names)-1)/2, len(schedule_names)) * width
    for i, name in enumerate(schedule_names):
        means, stds = phase_stats(all_errors, name)
        lw  = 1.5 if name == "Nose-Hoover" else 0.8
        ax.bar(x + offsets[i], means, width*0.9, yerr=stds,
               label=name, color=COLORS[name], alpha=0.8,
               error_kw=dict(elinewidth=lw, capsize=2))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ph{p+1}\n{PHASE_NAMES[p]}" for p in range(N_PHASES)])
    ax.set_title("Per-phase mean error ± 1σ")
    ax.set_ylabel("Mean MSE")
    ax.legend(fontsize=7)

    # Panel 3: NH temperature trajectory (mean ± std across seeds)
    ax = axes[2]
    T_arr = np.array(nh_T_all)   # (n_seeds, N_EPOCHS)
    T_mean = T_arr.mean(axis=0)
    T_std  = T_arr.std(axis=0)
    ax.plot(epochs, T_mean, color="firebrick", lw=2)
    ax.fill_between(epochs, T_mean-T_std, T_mean+T_std,
                    color="firebrick", alpha=0.2)
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5, label="T0=1.0")
    for xv in phase_boundaries:
        ax.axvline(xv, **kw_phase)
    for p, pname in enumerate(PHASE_NAMES):
        ax.text(p*N_EPOCHS_PHASE + N_EPOCHS_PHASE/2,
                ax.get_ylim()[1] if len(ax.get_ylim()) > 0 else 1.0,
                pname, ha="center", va="top", fontsize=7, color="gray")
    ax.set_title("NH temperature (mean ± 1σ across seeds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("T")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "rbm_multiswitch.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out}")


# ──────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────

def summarise(all_errors, nh_T_all):
    schedule_names = list(make_schedules().keys()) + ["Nose-Hoover"]
    print("\nPer-phase mean error (mean ± std across seeds)")
    header = f"{'Schedule':20s}" + "".join(
        f"  Ph{p+1} {PHASE_NAMES[p]:7s}" for p in range(N_PHASES)
    ) + "  Overall"
    print(header)
    print("-" * len(header))

    overall = {}
    for name in schedule_names:
        means, stds = phase_stats(all_errors, name)
        row = f"{name:20s}"
        for m, s in zip(means, stds):
            row += f"  {m:.4f}±{s:.4f}"
        ov = np.mean(means)
        overall[name] = ov
        row += f"  {ov:.4f}"
        print(row)

    sorted_names = sorted(overall, key=overall.get)
    nh_rank = sorted_names.index("Nose-Hoover") + 1
    print(f"\nOverall ranking (by mean phase error):")
    for rank, name in enumerate(sorted_names, 1):
        marker = " <-- NH" if name == "Nose-Hoover" else ""
        print(f"  {rank}. {name:20s}  {overall[name]:.4f}{marker}")

    print(f"\nNH overall rank: {nh_rank}/{len(sorted_names)}")

    # Check if NH's T reverses direction at any switch
    T_arr  = np.array(nh_T_all).mean(axis=0)
    print("\nNH mean T at phase boundaries:")
    for p in range(N_PHASES):
        t_start = T_arr[p * N_EPOCHS_PHASE]
        t_end   = T_arr[min((p+1)*N_EPOCHS_PHASE - 1, N_EPOCHS-1)]
        direction = "UP" if t_end > t_start else "down"
        print(f"  Phase {p+1} ({PHASE_NAMES[p]:7s}): "
              f"{t_start:.4f} -> {t_end:.4f}  [{direction}]")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Phases: {' -> '.join(PHASE_NAMES)}")
    print(f"Seeds:  {SEEDS}\n")

    all_errors, nh_T_all, schedules = run_all(device)
    summarise(all_errors, nh_T_all)
    plot(all_errors, nh_T_all, schedules)

    out = OUTPUT_DIR / "rbm_multiswitch_errors.npz"
    np.savez(str(out),
             **{k.replace(" ", "_").replace("=", "").replace(".", ""):
                np.array(v) for k, v in all_errors.items()},
             nh_T=np.array(nh_T_all))
    print(f"Raw arrays saved to {out}")
