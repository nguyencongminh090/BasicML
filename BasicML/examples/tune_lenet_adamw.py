# AI generated (refactored/authored with Claude Code)
"""Hyperparameter search for ``train_lenet_adamw.py``'s LeNet+AdamW model.

Run directly: python BasicML/examples/tune_lenet_adamw.py

Searches ``lr`` / ``weight_decay`` / the two conv kernel sizes on a cheap
proxy budget (fewer epochs, smaller train/test split than the full run) by
building each candidate as a ``dataclasses.replace()`` of one base
``LeNetConfig`` and calling straight into ``train_lenet_adamw.build_model``/
``train`` -- the model and training loop are defined exactly once, in that
file. MNIST is fetched once and every trial reuses the same in-memory
split, so cost scales with the number of trials, not with
re-downloading/re-splitting.

Two search modes (``SEARCH_MODE``):
    "random" (default) -- ``N_RANDOM_TRIALS`` candidates, each field drawn
        independently from ``SEARCH_SPACE``. With 4 fields to search,
        an exhaustive grid needs 3*3*2*2 = 36 trials to cover every corner;
        random search gets comparable coverage in far fewer trials and
        doesn't force every dimension's resolution up when only one or two
        actually matter (Bergstra & Bengio, 2012) -- the general reason to
        prefer it over grid search once you're past 2 tunable dimensions.
    "grid" -- the full cartesian product of ``SEARCH_SPACE``, useful when
        you want an exhaustive sweep and can afford all 36 trials.

This script only searches -- it does NOT retrain the winner at full
budget. It writes the winning ``LeNetConfig`` to ``OUTPUT_CONFIG_PATH`` via
``lenet_config.save_config``; run ``train_lenet_adamw.py --config
<path>`` separately to actually train it end to end. Keeping "search" and
"train" as two separate invocations means a long full-budget run is always
an explicit, deliberate step, never an automatic side effect of tuning.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import numpy as np
from dataclasses import dataclass, replace

from examples.lenet_config      import LeNetConfig, DEFAULT_CONFIG, save_config
from examples.train_lenet_adamw import load_mnist_split, build_model, train

# --- SEARCH BUDGET -----------------------------------------------------------
# Cheap proxy for ranking candidates: fewer epochs, smaller train/test split
# than the full run in train_lenet_adamw.py. A relative ranking of
# hyperparameters is stable well before accuracy saturates, so this doesn't
# need the full budget to be useful.
TUNE_EPOCHS  = 4
TUNE_N_TRAIN = 4000
TUNE_N_TEST  = 1000

SEARCH_MODE      = "random"   # "random" or "grid"
N_RANDOM_TRIALS  = 12
RANDOM_SEED      = 1

SEARCH_SPACE = {
    "lr"          : (0.0005, 0.001, 0.002),
    "weight_decay": (0.0, 0.01, 0.05),
    "kernel1"     : (3, 5),
    "kernel2"     : (3, 5),
}

OUTPUT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lenet_best.cfg")
# ---------------------------------------------------------------------------

BASE_CONFIG = replace(DEFAULT_CONFIG, epochs=TUNE_EPOCHS, n_train=TUNE_N_TRAIN, n_test=TUNE_N_TEST)


@dataclass
class TrialResult:
    """Outcome of one hyperparameter candidate.

    Attributes:
        config: The ``LeNetConfig`` this trial trained with.
        test_loss: Held-out test loss after the last epoch.
        test_acc: Held-out test accuracy after the last epoch.
    """
    config   : LeNetConfig
    test_loss: float
    test_acc : float


def search_grid(base: LeNetConfig) -> list[LeNetConfig]:
    """Build the full cartesian product of ``SEARCH_SPACE``.

    Args:
        base: Config every candidate is a ``replace()`` of.

    Returns:
        One ``LeNetConfig`` per grid point (``36`` for the default space).
    """
    names, choices = zip(*SEARCH_SPACE.items())
    return [replace(base, **dict(zip(names, combo))) for combo in itertools.product(*choices)]


def search_random(base: LeNetConfig, n_trials: int, seed: int) -> list[LeNetConfig]:
    """Draw ``n_trials`` candidates, each field sampled independently from ``SEARCH_SPACE``.

    Args:
        base: Config every candidate is a ``replace()`` of.
        n_trials: Number of candidates to draw.
        seed: Seed for the sampling RNG, for reproducible searches.

    Returns:
        ``n_trials`` sampled ``LeNetConfig`` instances (duplicates are
        possible since sampling is with replacement across trials).
    """
    rng = np.random.RandomState(seed)
    candidates = []
    for _ in range(n_trials):
        sampled = {name: choices[rng.randint(len(choices))] for name, choices in SEARCH_SPACE.items()}
        candidates.append(replace(base, **sampled))   # type: ignore[arg-type]
    return candidates


def run_trial(cfg: LeNetConfig, X_train, y_train_onehot, X_test, y_test_onehot) -> TrialResult:
    """Train one candidate config and record its final held-out metrics.

    Args:
        cfg: Candidate configuration.
        X_train: Training images, shared across every trial in the search.
        y_train_onehot: One-hot training labels.
        X_test: Held-out test images, shared across every trial.
        y_test_onehot: One-hot test labels.

    Returns:
        The candidate's :class:`TrialResult`.
    """
    layers = build_model(cfg)
    logs   = train(cfg, layers, X_train, y_train_onehot, X_test, y_test_onehot, verbose=False)
    return TrialResult(cfg, logs[-1].test_loss, logs[-1].test_acc)


def tune() -> list[TrialResult]:
    """Fetch MNIST once, then train every candidate (grid or random) on the shared split.

    Returns:
        Trial results sorted by descending ``test_acc``.
    """
    X_train, _, y_train_onehot, X_test, _, y_test_onehot = load_mnist_split(BASE_CONFIG)

    if SEARCH_MODE == "grid":
        candidates = search_grid(BASE_CONFIG)
    elif SEARCH_MODE == "random":
        candidates = search_random(BASE_CONFIG, N_RANDOM_TRIALS, RANDOM_SEED)
    else:
        raise ValueError(f"Unknown SEARCH_MODE {SEARCH_MODE!r}, expected 'grid' or 'random'")

    results = []
    print(f"Searching {len(candidates)} candidates ({SEARCH_MODE}) -- "
         f"{TUNE_EPOCHS} epochs, {TUNE_N_TRAIN} train / {TUNE_N_TEST} test images each...")
    for i, cfg in enumerate(candidates, start=1):
        result = run_trial(cfg, X_train, y_train_onehot, X_test, y_test_onehot)
        results.append(result)
        print(f"[{i:>2}/{len(candidates)}] lr={cfg.lr:<7} weight_decay={cfg.weight_decay:<5} "
             f"kernel1={cfg.kernel1} kernel2={cfg.kernel2} "
             f"-> test_loss={result.test_loss:.4f}  test_acc={result.test_acc:.4f}")

    return sorted(results, key=lambda r: r.test_acc, reverse=True)


def print_ranking(results: list[TrialResult]) -> None:
    """Print every trial ranked best-to-worst by test accuracy.

    Args:
        results: Trial results, as returned by :func:`tune`.
    """
    print("\nRanking (best first):")
    for rank, result in enumerate(results, start=1):
        cfg = result.config
        print(f"  {rank:>2}. lr={cfg.lr:<7} weight_decay={cfg.weight_decay:<5} "
             f"kernel1={cfg.kernel1} kernel2={cfg.kernel2} "
             f"test_acc={result.test_acc:.4f}  test_loss={result.test_loss:.4f}")


def main() -> None:
    """Search the AdamW/kernel-size hyperparameter space and save the winner to a .cfg file."""
    results = tune()
    print_ranking(results)

    best = results[0].config
    winner = replace(DEFAULT_CONFIG, lr=best.lr, weight_decay=best.weight_decay,
                     kernel1=best.kernel1, kernel2=best.kernel2)
    save_config(winner, OUTPUT_CONFIG_PATH)
    print(f"\nBest candidate (test_acc={results[0].test_acc:.4f} on the "
         f"{TUNE_N_TRAIN}/{TUNE_N_TEST}-image search budget): "
         f"lr={best.lr}  weight_decay={best.weight_decay}  kernel1={best.kernel1}  kernel2={best.kernel2}")
    print(f"Saved full-budget config to {OUTPUT_CONFIG_PATH}")
    print(f"Train it: python BasicML/examples/train_lenet_adamw.py --config {OUTPUT_CONFIG_PATH}")


if __name__ == "__main__":
    main()
