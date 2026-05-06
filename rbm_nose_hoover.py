"""
Restricted Boltzmann Machine (RBM) with Nosé-Hoover Thermostat
===============================================================
The Nosé-Hoover thermostat dynamically controls the sampling temperature
by tracking the average hidden unit activation and adjusting T to keep
it near a target value. This removes the need for manual temperature
scheduling and improves mixing during Gibbs sampling.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 1.  RBM with Nosé-Hoover Thermostat
# ──────────────────────────────────────────────────────────────

class RBM_NoseHoover(nn.Module):
    """
    Restricted Boltzmann Machine with a Nosé-Hoover thermostat.

    The thermostat introduces an auxiliary variable xi (ξ) that evolves
    based on the difference between the current average hidden activation
    and a target activation. The effective temperature is:

        T(t) = T0 * exp(xi(t))

    Parameters
    ----------
    n_visible   : number of visible units
    n_hidden    : number of hidden units
    T0          : baseline temperature  (default 1.0)
    h_target    : target average hidden activation (default 0.3)
    Q           : thermostat inertia — larger Q = slower response
    alpha       : thermostat learning rate
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        T0: float = 1.0,
        h_target: float = 0.3,
        Q: float = 0.1,
        alpha: float = 0.01,
    ):
        super().__init__()

        # ── Model parameters ──────────────────────────────────
        self.W = nn.Parameter(
            torch.randn(n_visible, n_hidden) * 0.01
        )
        self.b = nn.Parameter(torch.zeros(n_visible))   # visible bias
        self.c = nn.Parameter(torch.zeros(n_hidden))    # hidden bias

        # ── Thermostat state ──────────────────────────────────
        self.T0       = T0
        self.h_target = h_target
        self.Q        = Q
        self.alpha    = alpha

        # xi starts at 0 → T = T0 * exp(0) = T0
        self.xi = 0.0

        # Track temperature and hidden activations for diagnostics
        self.temperature_history    = []
        self.h_activation_history   = []

    # ── Temperature ───────────────────────────────────────────
    @property
    def temperature(self) -> float:
        """Current effective temperature."""
        return self.T0 * np.exp(self.xi)

    # ── Conditional probabilities ─────────────────────────────
    def prob_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        """P(h_j = 1 | v) with current temperature."""
        logit = v @ self.W + self.c          # (batch, n_hidden)
        return torch.sigmoid(logit / self.temperature)

    def prob_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        """P(v_i = 1 | h) with current temperature."""
        logit = h @ self.W.t() + self.b      # (batch, n_visible)
        return torch.sigmoid(logit / self.temperature)

    # ── Sampling ──────────────────────────────────────────────
    def sample_h(self, v: torch.Tensor):
        """Sample hidden units given visible units."""
        p_h = self.prob_h_given_v(v)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h: torch.Tensor):
        """Sample visible units given hidden units."""
        p_v = self.prob_v_given_h(h)
        return p_v, torch.bernoulli(p_v)

    # ── Gibbs sampling ────────────────────────────────────────
    def gibbs_chain(self, v0: torch.Tensor, k: int = 1):
        """
        Run k steps of block Gibbs sampling.  k must be >= 1.

        Returns
        -------
        p_h0  : P(h | v0)         — positive phase hidden probs
        h0    : sampled h from v0
        p_vk  : P(v | hk)         — negative phase visible probs
        vk    : sampled v after k steps
        p_hk  : P(h | vk)         — negative phase hidden probs
        hk    : sampled h from vk
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        p_h0, h0 = self.sample_h(v0)

        v, h = v0, h0
        for _ in range(k):
            p_v, v = self.sample_v(h)
            p_h, h = self.sample_h(v)

        return p_h0, h0, p_v, v, p_h, h

    # ── Energy ────────────────────────────────────────────────
    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        E(v, h) = -b^T v - c^T h - v^T W h
        Returns shape (batch,)
        """
        bv  = (v * self.b).sum(dim=1)
        ch  = (h * self.c).sum(dim=1)
        vWh = (v @ self.W * h).sum(dim=1)
        return -(bv + ch + vWh)

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        F(v) = -b^T v - T * sum_j log(1 + exp((c_j + W_j^T v) / T))
        Returns shape (batch,)
        """
        bv       = (v * self.b).sum(dim=1)
        wx_b     = (v @ self.W + self.c) / self.temperature
        log_term = self.temperature * torch.nn.functional.softplus(wx_b).sum(dim=1)
        return -(bv + log_term)

    # ── Nosé-Hoover thermostat update ─────────────────────────
    def update_thermostat(self, h_avg: float):
        """
        Update the thermostat variable xi based on the deviation
        of the average hidden activation from the target.

        Discretised first-order feedback (inspired by Nosé-Hoover):
            xi += (alpha/Q) * (h_avg - h_target)

        Note: alpha and Q only appear as their ratio — they are
        effectively one parameter.  xi is clamped to [-5, 5] to
        keep temperature in [T0*e^-5, T0*e^5] ≈ [T0/148, T0*148].

        h_avg is the mean of P(h=1|v) over the full batch×hidden
        tensor, so h_target should match that same global mean.
        """
        delta = h_avg - self.h_target
        # Negative sign: h_avg > h_target → lower T → sparser sigmoid
        self.xi -= self.alpha * delta / self.Q
        self.xi = float(np.clip(self.xi, -2.0, 2.0))

        # Log diagnostics
        self.temperature_history.append(self.temperature)
        self.h_activation_history.append(h_avg)

    # ── Contrastive Divergence update ─────────────────────────
    def contrastive_divergence(
        self,
        v0: torch.Tensor,
        k: int = 1,
        lr: float = 0.01,
    ) -> float:
        """
        One step of CD-k with the Nosé-Hoover thermostat.

        Weight update:
            ΔW  = lr * (⟨vh⟩_data − ⟨vh⟩_model)
            Δb  = lr * (v0 − vk)
            Δc  = lr * (p_h0 − p_hk)

        Returns the reconstruction error for monitoring.

        Note: W, b, c are nn.Parameter but are updated manually via
        .data — do NOT pass model.parameters() to a torch optimizer,
        as .grad is never populated here.
        """
        with torch.no_grad():
            # ── Positive and negative phases ──────────────────
            p_h0, h0, p_vk, vk, p_hk, hk = self.gibbs_chain(v0, k=k)

            batch_size = v0.size(0)

            # Positive statistics: ⟨v h⟩_data
            pos = v0.t() @ p_h0 / batch_size      # (n_visible, n_hidden)

            # Negative statistics: ⟨v h⟩_model
            neg = vk.t() @ p_hk / batch_size      # (n_visible, n_hidden)

            # ── Weight and bias updates ────────────────────────
            self.W.data += lr * (pos - neg)
            self.b.data += lr * (v0 - vk).mean(dim=0)
            self.c.data += lr * (p_h0 - p_hk).mean(dim=0)

        # ── Thermostat update ─────────────────────────────────
        h_avg = p_h0.mean().item()
        self.update_thermostat(h_avg)

        # ── Reconstruction error ──────────────────────────────
        recon_error = ((v0 - p_vk) ** 2).mean().item()
        return recon_error

    # ── Generate samples ──────────────────────────────────────
    def generate(
        self,
        n_samples: int = 16,
        n_gibbs: int = 1000,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate samples by running a long Gibbs chain from noise.

        Uses the current self.temperature (whatever xi settled at after
        training).  To generate at the baseline temperature, reset xi
        to 0.0 before calling.
        """
        print(f"Generating at T={self.temperature:.4f} (xi={self.xi:.4f})")
        n_visible = self.b.size(0)
        v = torch.bernoulli(torch.full((n_samples, n_visible), 0.5)).to(device)
        with torch.no_grad():
            for _ in range(n_gibbs):
                _, h = self.sample_h(v)
                _, v = self.sample_v(h)
        return v


# ──────────────────────────────────────────────────────────────
# 2.  Training Loop
# ──────────────────────────────────────────────────────────────

def train_rbm(
    model: RBM_NoseHoover,
    dataloader: DataLoader,
    n_epochs: int = 100,
    lr: float = 0.01,
    k: int = 1,
    device: str = "cpu",
) -> list:
    """
    Train the RBM using CD-k with Nosé-Hoover temperature control.

    Returns a list of per-epoch average reconstruction errors.
    """
    model.to(device)
    epoch_errors = []

    for epoch in range(n_epochs):
        batch_errors = []

        for batch, _ in dataloader:
            # Flatten and binarise the input
            v0 = batch.view(batch.size(0), -1).to(device)
            v0 = (v0 > 0.5).float()

            error = model.contrastive_divergence(v0, k=k, lr=lr)
            batch_errors.append(error)

        avg_error = np.mean(batch_errors)
        epoch_errors.append(avg_error)

        print(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Recon Error: {avg_error:.4f} | "
            f"T: {model.temperature:.4f} | "
            f"xi: {model.xi:.4f} | "
            f"h_avg: {model.h_activation_history[-1]:.3f}"
        )

    return epoch_errors


# ──────────────────────────────────────────────────────────────
# 3.  Diagnostics / Plotting
# ──────────────────────────────────────────────────────────────

def plot_diagnostics(model: RBM_NoseHoover, errors: list):
    """Plot training diagnostics: error, temperature, and h activation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Reconstruction error
    axes[0].plot(errors, color="steelblue")
    axes[0].set_title("Reconstruction Error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")

    # Temperature trajectory
    axes[1].plot(model.temperature_history, color="firebrick", alpha=0.7)
    axes[1].axhline(model.T0, linestyle="--", color="gray", label=f"T0={model.T0}")
    axes[1].set_title("Temperature (Nosé-Hoover)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("T")
    axes[1].legend()

    # Hidden activation
    axes[2].plot(model.h_activation_history, color="darkorange", alpha=0.7)
    axes[2].axhline(
        model.h_target, linestyle="--", color="gray",
        label=f"target={model.h_target}"
    )
    axes[2].set_title("Avg Hidden Activation")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("⟨h⟩")
    axes[2].legend()

    plt.tight_layout()
    out = OUTPUT_DIR / "rbm_diagnostics.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Diagnostics saved to {out}")


# ──────────────────────────────────────────────────────────────
# 4.  Example usage on MNIST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load MNIST ────────────────────────────────────────────
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # ── Create model ──────────────────────────────────────────
    model = RBM_NoseHoover(
        n_visible=784,      # 28x28 pixels
        n_hidden=256,
        T0=1.0,             # baseline temperature
        h_target=0.3,       # target average hidden activation
        Q=10.0,             # thermostat inertia — effective gain = alpha/Q = 0.001
        alpha=0.01,         # thermostat learning rate
    )

    # ── Train ─────────────────────────────────────────────────
    errors = train_rbm(
        model,
        loader,
        n_epochs=100,
        lr=0.01,
        k=1,
        device=device,
    )

    # ── Plot diagnostics ──────────────────────────────────────
    plot_diagnostics(model, errors)

    # ── Save model ────────────────────────────────────────────
    out = OUTPUT_DIR / "rbm_nose_hoover.pt"
    torch.save(model.state_dict(), out)
    print(f"Model saved to {out}")
