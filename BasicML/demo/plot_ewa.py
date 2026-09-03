# AI generated (refactored/authored with Claude Code)
"""Exponentially Weighted Averages (EWA) -- an interactive demo.

Run directly: python BasicML/demo/plot_ewa.py

EWA is the running-average trick that Momentum / RMSProp / Adam are built on:

    v[0] = 0
    v[t] = beta * v[t-1] + (1 - beta) * theta[t]

`v[t]` behaves like the average of the last ~ 1 / (1 - beta) samples, with
older samples decaying geometrically (weight of theta[t-k] is (1-beta)*beta^k).

The figure has three panels:
  1. Noisy signal + EWA curves for several beta values. Larger beta -> smoother
     but more lag (the curve trails the true trend).
  2. Bias correction. v[0]=0 drags the first samples down; dividing by
     (1 - beta^t) fixes the cold start and fades out as t grows.
  3. The geometric weights each past sample receives, per beta.

Keys:  b -> toggle bias correction on panel 1     q -> quit
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy             as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG --------------------------------------------------------------------
SEED       = 7
N_DAYS     = 200
BETAS      = (0.5, 0.9, 0.98)          # ~2, ~10, ~50 sample window
NOISE_STD  = 6.0
FIG_SIZE   = (14, 10)


def ewa(theta: np.ndarray, beta: float, bias_correction: bool = False) -> np.ndarray:
    """Exponentially weighted average of the 1-D sequence `theta`.

    Returns an array `v` of the same length, where v[t] uses theta[0..t].
    With `bias_correction`, v[t] is divided by (1 - beta ** (t + 1)).
    """
    v       = np.zeros_like(theta, dtype=float)
    running = 0.0
    for t, x in enumerate(theta):
        running = beta * running + (1.0 - beta) * x
        v[t]    = running / (1.0 - beta ** (t + 1)) if bias_correction else running
    return v


def make_signal(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A slow seasonal trend (the 'truth') plus Gaussian measurement noise."""
    days  = np.arange(N_DAYS)
    trend = 15.0 + 12.0 * np.sin(2.0 * np.pi * days / N_DAYS) + 0.03 * days
    noisy = trend + rng.normal(0.0, NOISE_STD, size=N_DAYS)
    return trend, noisy


def main() -> None:
    rng          = np.random.default_rng(SEED)
    trend, noisy = make_signal(rng)
    days         = np.arange(N_DAYS)

    state = {"bias_correction": False}

    fig, (ax_sig, ax_bias, ax_w) = plt.subplots(3, 1, figsize=FIG_SIZE)
    fig.suptitle("Exponentially Weighted Averages", fontsize=14, fontweight="bold")

    def draw_signal() -> None:
        ax_sig.clear()
        ax_sig.plot(days, noisy, ".", color="0.7", label="noisy samples (theta)")
        ax_sig.plot(days, trend, "k--", lw=1.5, label="true trend")
        for beta in BETAS:
            v      = ewa(noisy, beta, state["bias_correction"])
            window = 1.0 / (1.0 - beta)
            ax_sig.plot(days, v, lw=2.0, label=f"EWA beta={beta} (~{window:.0f} samples)")
        title = "1. Noisy signal vs EWA"
        if state["bias_correction"]:
            title += "   [bias-corrected]"
        ax_sig.set_title(title + "   -- press 'b' to toggle bias correction")
        ax_sig.set_xlabel("day")
        ax_sig.set_ylabel("temperature")
        ax_sig.legend(loc="upper left", fontsize=8)
        fig.canvas.draw_idle()

    # Panel 2: bias correction, fixed beta.
    beta_bc = 0.98
    raw     = ewa(noisy, beta_bc, bias_correction=False)
    fixed   = ewa(noisy, beta_bc, bias_correction=True)
    ax_bias.plot(days, noisy, ".", color="0.8", label="noisy samples")
    ax_bias.plot(days, trend, "k--", lw=1.2, label="true trend")
    ax_bias.plot(days, raw,   lw=2.0, label=f"EWA beta={beta_bc}, v[0]=0  (cold start)")
    ax_bias.plot(days, fixed, lw=2.0, label=f"EWA beta={beta_bc}, / (1 - beta^t)")
    ax_bias.set_xlim(0, 80)
    ax_bias.set_title("2. Bias correction removes the cold-start dip (first ~50 steps)")
    ax_bias.set_xlabel("day")
    ax_bias.set_ylabel("temperature")
    ax_bias.legend(loc="lower right", fontsize=8)

    # Panel 3: geometric weight each past sample gets.
    lag = np.arange(60)
    for beta in BETAS:
        ax_w.plot(lag, (1.0 - beta) * beta ** lag, "o-", ms=3, label=f"beta={beta}")
    ax_w.set_title("3. Weight of sample theta[t-k]:  (1 - beta) * beta^k")
    ax_w.set_xlabel("age of the sample  (k steps in the past)")
    ax_w.set_ylabel("weight")
    ax_w.legend(fontsize=8)

    def on_key(event) -> None:
        if event.key == "b":
            state["bias_correction"] = not state["bias_correction"]
            draw_signal()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_signal()
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # Console sanity check: constant input -> bias-corrected EWA is exactly that constant.
    const = np.full(10, 42.0)
    print("constant-input check (beta=0.9):")
    print("  raw EWA           :", ewa(const, 0.9)[:5])
    print("  bias-corrected EWA:", ewa(const, 0.9, bias_correction=True)[:5])

    out = os.environ.get("EWA_SAVE")
    if out:
        fig.savefig(out, dpi=110)
        print("saved:", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
