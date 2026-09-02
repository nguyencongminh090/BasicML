# AI generated (refactored/authored with Claude Code)
"""Visualize the VANISHING GRADIENT on a deep MLP, live as it trains.

Run directly: python BasicML/demo/plot_dynamic_vanishing_gradient.py

Two networks of identical depth and width are compared; they differ only in
hidden activation and weight init:
  - left : Sigmoid + Xavier  -> the backprop signal shrinks toward the input
                                layers (VANISHING)
  - right: ReLU + He         -> healthy gradient flow (control)

Each column shows:
  1. A sparse neuron graph (top-k strongest edges per layer pair).
  2. A row of heatmaps for a sample of Linear layers, showing `w` or `dL/dw`
     (toggle with the `g` / `w` keys); each subplot title reports its norm.
  3. BCE-loss (train) and validation-accuracy curves for both networks.
  4. The backprop signal by depth at the current epoch (log scale):
     RMS(delta = dL/dz) as a solid line and RMS(dL/dw) dotted. delta shrinking
     toward the input layers is the vanishing signature. Alongside: mean f'(z)
     per layer and its cross-layer product (the attenuation factor).

There is no regularization in this demo, so every gradient shown is the plain
backprop gradient.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from basicml.nn.sequential   import Sequential
from basicml.nn.linear       import Linear
from basicml.nn.activation   import ReLU, Sigmoid, Tanh, Activation
from basicml.nn.loss         import BinaryCrossEntropy
from basicml.optim.momentum  import Momentum

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ----------------------------------------------------------------
SEED           = 7
DEPTH          = 12                    # number of hidden layers -> a deep net
WIDTH          = 24
EPOCHS         = 800
LEARN_RATE     = 0.10
MOMENTUM       = 0.9
RECORD_EVERY   = 6                     # snapshot every N epochs

MAX_NODES      = 9                     # neurons drawn per layer in the graph
TOPK_EDGES     = 22                    # strongest edges drawn per layer pair
N_HEATMAPS     = 6                     # Linear layers sampled for the heatmap row
HEATMAP_MODE   = "grad"               # "weight" | "grad"; toggled with the g / w keys
FRAME_INTERVAL = 40                    # milliseconds between frames
FIG_SIZE       = (18, 13)

MODELS = (
    {"name": "Sigmoid + Xavier (deep)  ->  VANISHING", "short": "sigmoid+xavier",
     "act": "sigmoid", "init": "xavier"},
    {"name": "ReLU + He (deep)  ->  healthy",          "short": "relu+he",
     "act": "relu", "init": "he"},
)
COLORS   = ("#c0392b", "#2c7fb8")      # left (vanishing) / right (healthy)
_ACT_CLS = {"sigmoid": Sigmoid, "relu": ReLU, "tanh": Tanh}
# ---------------------------------------------------------------------------


def load_data() -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a binary-classification dataset, split it, and standardize the features.

    Prefers sklearn's breast-cancer dataset and falls back to a synthetic
    two-moons set so the demo runs without sklearn. Features are standardized
    using train-split statistics only.

    Returns:
        A tuple ``(source, Xtr, ytr, Xval, yval)``: ``source`` is a short dataset
        name, ``Xtr`` / ``Xval`` are ``(N, F)`` float arrays and ``ytr`` /
        ``yval`` are ``(N, 1)`` float label arrays.
    """
    try:
        from sklearn.datasets import load_breast_cancer
        X_raw, y_raw = load_breast_cancer(return_X_y=True)
        X      = np.asarray(X_raw, dtype=np.float64)
        y      = np.asarray(y_raw, dtype=np.float64).reshape(-1, 1)
        source = "sklearn.load_breast_cancer (30 features)"
    except Exception:
        from basicml.datasets import make_moons
        X, y   = make_moons(n_samples=1500, noise=0.30, random_state=SEED)
        X      = np.asarray(X, dtype=np.float64)
        y      = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        source = "basicml.make_moons (fallback, 2 features)"

    rng       = np.random.RandomState(SEED)
    perm      = rng.permutation(len(X))
    X, y      = X[perm], y[perm]
    n_tr      = int(0.7 * len(X))
    Xtr, Xval = X[:n_tr], X[n_tr:]
    ytr, yval = y[:n_tr], y[n_tr:]

    mu, sd    = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr       = (Xtr - mu) / sd
    Xval      = (Xval - mu) / sd
    return source, Xtr, ytr, Xval, yval


def build_model(n_features: int, act: str, init_type: str) -> Sequential:
    """Build a deep MLP: ``[n_features] -> DEPTH x WIDTH (act) -> 1 (Sigmoid)``.

    Args:
        n_features: Number of input features.
        act: Hidden activation, one of ``"sigmoid"``, ``"relu"``, ``"tanh"``.
        init_type: Weight init for the hidden layers, e.g. ``"xavier"`` or
            ``"he"``. The output layer always uses Xavier init.

    Returns:
        The assembled ``Sequential`` model.
    """
    act_cls          = _ACT_CLS[act]
    sizes: list[int] = [n_features, *([WIDTH] * DEPTH)]
    layers: list     = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], init_type=init_type))
        layers.append(act_cls())
    layers.append(Linear(sizes[-1], 1, init_type="xavier"))
    layers.append(Sigmoid())
    return Sequential(*layers)


def linear_layers(model: Sequential) -> list[Linear]:
    """Return the model's ``Linear`` layers in forward order.

    Args:
        model: A ``Sequential`` whose ``layers`` list is scanned.

    Returns:
        The ``Linear`` instances only.
    """
    return [layer for layer in model.layers if isinstance(layer, Linear)]


def hidden_activations(model: Sequential) -> list[Activation]:
    """Return the hidden activation layers, excluding the output Sigmoid.

    Args:
        model: A ``Sequential`` built by :func:`build_model`.

    Returns:
        The activation layers in forward order, minus the final one.
    """
    return [layer for layer in model.layers if isinstance(layer, Activation)][:-1]


def weight_grad(layer: Linear) -> np.ndarray:
    """Return a Linear layer's weight gradient, asserting it has been populated.

    Args:
        layer: The ``Linear`` layer to read ``w.grad`` from.

    Returns:
        The ``(in_features, out_features)`` gradient array.

    Raises:
        RuntimeError: If ``w.grad`` is None (no backward pass since the last
            ``zero_grad()``).
    """
    g = layer.w.grad
    if g is None:
        raise RuntimeError("weight gradient is None; run the backward pass first")
    return g


def act_out(layer: Activation) -> np.ndarray | None:
    """Return an activation layer's cached output, or None if it has not run.

    Args:
        layer: Any ``Activation`` (its ``out`` attribute is read if present).

    Returns:
        The cached forward output array, or None.
    """
    return getattr(layer, "out", None)


def act_derivative_mean(layer: Activation) -> float:
    """Estimate mean f'(z) for an activation layer from its cached output.

    Uses the closed form of each activation's derivative in terms of its output
    ``o``: sigmoid ``o(1-o)``, tanh ``1-o**2``, ReLU ``1[o > 0]``.

    Args:
        layer: The activation layer; must have run a forward pass.

    Returns:
        The mean derivative over all elements, or 0.0 if the type is unknown or
        no output is cached.
    """
    o = act_out(layer)
    if o is None:
        return 0.0
    if isinstance(layer, Sigmoid):
        d = o * (1.0 - o)
    elif isinstance(layer, Tanh):
        d = 1.0 - o ** 2
    elif isinstance(layer, ReLU):
        d = (o > 0.0).astype(np.float64)
    else:
        return 0.0
    return float(np.mean(d))


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary accuracy as a percentage, thresholding predictions at 0.5.

    Args:
        y_pred: Predicted probabilities, shape ``(N, 1)``.
        y_true: Ground-truth labels (0/1), shape ``(N, 1)``.

    Returns:
        Accuracy in the range ``[0, 100]``.
    """
    return float(np.mean((y_pred >= 0.5).astype(int) == y_true.astype(int))) * 100.0


def backward_capture(model: Sequential, grad_out: np.ndarray) -> list[np.ndarray]:
    """Run the backward pass and return delta = dL/dz at each Linear layer's input.

    Behaves exactly like ``Sequential.backward`` (weights still accumulate into
    ``w.grad``), but also captures the upstream gradient entering every Linear
    layer. That gradient, ``delta_l = dL/dz_l``, is the pure backprop signal;
    ``||delta_l||`` shrinking toward the input layers is the direct signature of
    a vanishing gradient. It is cleaner than ``||dL/dw||`` because
    ``dL/dw_l = a_{l-1}^T @ delta_l`` and the forward activations ``a`` partly
    mask the decay.

    Args:
        model: The ``Sequential`` to back-propagate through.
        grad_out: dL/d(model output), shape ``(N, 1)``.

    Returns:
        One delta array per Linear layer, ordered input layer to output layer.
    """
    grad   = grad_out
    deltas: list[np.ndarray] = []
    for layer in reversed(model.layers):
        if isinstance(layer, Linear):
            deltas.append(np.asarray(grad))  # grad entering this Linear == dL/dz_l
        grad = layer.backward(grad)
    deltas.reverse()
    return deltas


def train_and_record(
    model: Sequential,
    Xtr  : np.ndarray, ytr: np.ndarray,
    Xval : np.ndarray, yval: np.ndarray,
) -> dict:
    """Full-batch train a model and snapshot its state for the animation.

    Every ``RECORD_EVERY`` epochs it records: weight matrices, weight-gradient
    matrices, per-layer RMS of dL/dw, per-layer RMS of delta (dL/dz),
    validation accuracy, and per-hidden-layer activation statistics
    (mean|activation| and mean f'(z)).

    Args:
        model: The MLP to train; mutated in place.
        Xtr, ytr: Training features ``(N, F)`` and labels ``(N, 1)``.
        Xval, yval: Validation features and labels, same shapes.

    Returns:
        A dict of history lists keyed by ``"epochs"``, ``"loss"``, ``"vacc"``,
        ``"weights"``, ``"grad"``, ``"gnorm"``, ``"dnorm"``, ``"act_mean"`` and
        ``"act_deriv"``. Each entry has one item per recorded epoch.
    """
    criterion = BinaryCrossEntropy()
    optimizer = Momentum(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)
    linears   = linear_layers(model)
    hid_acts  = hidden_activations(model)

    hist: dict = {k: [] for k in
                  ("epochs", "loss", "vacc", "weights", "grad", "gnorm", "dnorm",
                   "act_mean", "act_deriv")}

    for epoch in range(EPOCHS):
        model.train()
        y_pred = model(Xtr)
        loss   = criterion(y_pred, ytr)
        deltas = backward_capture(model, criterion.backward())

        record = epoch % RECORD_EVERY == 0 or epoch == EPOCHS - 1
        if record:
            hist["grad"].append([weight_grad(l).copy() for l in linears])
            # RMS (not Frobenius) so layers of different size compare fairly.
            hist["gnorm"].append([float(np.sqrt(np.mean(weight_grad(l) ** 2)))
                                  for l in linears])
            hist["dnorm"].append([float(np.sqrt(np.mean(d ** 2))) for d in deltas])

        optimizer.step()
        optimizer.zero_grad()

        if record:
            model.eval()
            val_pred = model(Xval)
            _        = model(Xtr)  # last forward leaves activation.out on the train set
            hist["epochs"].append(epoch)
            hist["loss"].append(float(loss))
            hist["vacc"].append(accuracy(val_pred, yval))
            hist["weights"].append([l.w.data.copy() for l in linears])
            hist["act_mean"].append([float(np.mean(np.abs(o))) if (o := act_out(a)) is not None
                                     else 0.0 for a in hid_acts])
            hist["act_deriv"].append([act_derivative_mean(a) for a in hid_acts])

    print(f"  done: loss {hist['loss'][-1]:.4f} | val acc {hist['vacc'][-1]:.1f}%")
    return hist


def node_layout(sizes: list[int]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Compute which neurons to draw and their screen coordinates per layer.

    Args:
        sizes: Neuron count of each layer to draw, input to output.

    Returns:
        ``(disp, ys, xs)``: ``disp[j]`` are the neuron indices shown for layer
        ``j`` (sub-sampled to at most ``MAX_NODES``), ``ys[j]`` their vertical
        positions in ``[0, 1]``, and ``xs`` the horizontal position of each layer.
    """
    disp: list[np.ndarray] = []
    ys  : list[np.ndarray] = []
    for n in sizes:
        k = min(n, MAX_NODES)
        disp.append(np.linspace(0, n - 1, k).round().astype(int))
        ys.append(np.linspace(0.06, 0.94, k))
    xs = np.arange(len(sizes), dtype=np.float64)
    return disp, ys, xs


def draw_graph(ax, weights: list[np.ndarray],
               disp: list[np.ndarray], ys: list[np.ndarray], xs: np.ndarray,
               wmax: float) -> None:
    """Redraw the sparse neuron graph for one model on ``ax``.

    Args:
        ax: The matplotlib Axes to clear and draw on.
        weights: Per-layer weight matrices.
        disp, ys, xs: Layout from :func:`node_layout`.
        wmax: Global max |weight| used to normalize edge width and opacity.
    """
    ax.clear()
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for j, W in enumerate(weights):
        sub = W[np.ix_(disp[j], disp[j + 1])]
        mag = np.abs(sub).ravel()
        if mag.size == 0:
            continue
        # Keep only the TOPK_EDGES largest-magnitude edges of this layer pair.
        k   = min(TOPK_EDGES, mag.size)
        thr = np.partition(mag, -k)[-k]
        for a, _ in enumerate(disp[j]):
            for b, _ in enumerate(disp[j + 1]):
                w = sub[a, b]
                if abs(w) < thr:
                    continue
                ax.plot([xs[j], xs[j + 1]], [ys[j][a], ys[j + 1][b]],
                        color=("#c0392b" if w > 0 else "#2c3e50"),
                        linewidth=0.4 + 2.5 * abs(w) / wmax,
                        alpha=min(1.0, 0.12 + abs(w) / wmax),
                        zorder=1)

    for j, xj in enumerate(xs):
        ax.scatter(np.full(len(ys[j]), xj), ys[j], s=55, color="white",
                   edgecolors="#34495e", linewidths=1.1, zorder=2)


def animate(source: str, sizes: list[int], hists: tuple[dict, dict]) -> FuncAnimation:
    """Build and show the side-by-side vanishing-gradient animation.

    Args:
        source: Dataset name for the figure title.
        sizes: ``[n_features, WIDTH, ..., WIDTH, 1]`` (length ``DEPTH + 2``).
        hists: The two history dicts from :func:`train_and_record`, in the same
            order as :data:`MODELS`.

    Returns:
        The ``FuncAnimation`` (kept alive by the caller / ``plt.show``).
    """
    n_lin        = len(sizes) - 1  # number of Linear layers == DEPTH + 1
    n_hid        = DEPTH
    disp, ys, xs = node_layout(sizes)

    hm_idx = sorted(set(np.linspace(0, n_lin - 1, N_HEATMAPS).round().astype(int).tolist()))

    wmax = max(float(np.abs(W).max())
               for h in hists for snap in h["weights"] for W in snap)
    allg = [max(v, 1e-13) for h in hists for key in ("gnorm", "dnorm")
            for row in h[key] for v in row]
    gn_lo, gn_hi = min(allg) * 0.3, max(allg) * 3.0
    alld = [d for h in hists for row in h["act_deriv"] for d in row if d > 0]
    ad_lo, ad_hi = (min(alld) * 0.5, max(alld) * 2.0) if alld else (1e-3, 1.0)

    n_frames = min(len(hists[0]["epochs"]), len(hists[1]["epochs"]))
    max_loss = max(hists[0]["loss"][0], hists[1]["loss"][0]) * 1.05

    fig = plt.figure(figsize=FIG_SIZE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - Vanishing Gradient")
    outer = fig.add_gridspec(4, 2, height_ratios=[2.8, 1.3, 1.7, 1.9],
                             hspace=0.5, wspace=0.13)

    ax_graph = [fig.add_subplot(outer[0, c]) for c in range(2)]
    hm_ax    = [
        [fig.add_subplot(outer[1, c].subgridspec(1, len(hm_idx), wspace=0.2)[0, i])
         for i in range(len(hm_idx))]
        for c in range(2)
    ]
    ax_loss = fig.add_subplot(outer[2, 0])
    ax_acc  = fig.add_subplot(outer[2, 1])
    ax_grad = fig.add_subplot(outer[3, 0])
    ax_der  = fig.add_subplot(outer[3, 1])

    names  = [m["name"] for m in MODELS]
    shorts = [m["short"] for m in MODELS]

    # Heatmap view state; toggled between weights and gradients by the g / w keys.
    view = {"mode": HEATMAP_MODE if HEATMAP_MODE in ("weight", "grad") else "grad"}

    def on_key(event) -> None:
        """Switch the heatmap view on `g` (gradient) / `w` (weight) key presses."""
        if event.key == "g":
            view["mode"] = "grad"
        elif event.key == "w":
            view["mode"] = "weight"
    fig.canvas.mpl_connect("key_press_event", on_key)

    loss_lines = [ax_loss.plot([], [], color=COLORS[m], lw=2, label=shorts[m])[0]
                  for m in range(2)]
    ax_loss.set_xlim(0, EPOCHS)
    ax_loss.set_ylim(0, max_loss)
    ax_loss.set_title("BCE loss (train)")
    ax_loss.set_xlabel("epoch")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend(loc="upper right", fontsize=8)

    acc_lines = [ax_acc.plot([], [], color=COLORS[m], lw=2, label=shorts[m])[0]
                 for m in range(2)]
    ax_acc.set_xlim(0, EPOCHS)
    ax_acc.set_ylim(40, 101)
    ax_acc.set_title("Validation accuracy (%)")
    ax_acc.set_xlabel("epoch")
    ax_acc.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend(loc="lower right", fontsize=8)

    def update(frame: int):
        """Redraw every panel for the recorded epoch at index ``frame``."""
        for c, h in enumerate(hists):
            draw_graph(ax_graph[c], h["weights"][frame], disp, ys, xs, wmax)
            ax_graph[c].set_title(names[c], fontsize=11, color=COLORS[c])

            src  = h["grad"] if view["mode"] == "grad" else h["weights"]
            unit = "\\partial L/\\partial w" if view["mode"] == "grad" else "w"
            for slot, k in enumerate(hm_idx):
                ax  = hm_ax[c][slot]
                mat = np.asarray(src[frame][k])
                vmx = float(np.abs(mat).max()) or 1e-12
                ax.clear()
                ax.imshow(mat.T, cmap="RdBu_r", vmin=-vmx, vmax=vmx, aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"L{k + 1}\n$\\|{unit}\\|$={float(np.linalg.norm(mat)):.1e}",
                             fontsize=7)

        e    = hists[0]["epochs"][frame]
        for m in range(2):
            ep = hists[m]["epochs"][:frame + 1]
            loss_lines[m].set_data(ep, hists[m]["loss"][:frame + 1])
            acc_lines[m].set_data(ep, hists[m]["vacc"][:frame + 1])

        # Backprop signal by depth: RMS(delta) solid, RMS(dL/dw) dotted.
        ax_grad.clear()
        layers_x = np.arange(1, n_lin + 1)
        for m in range(2):
            dn = np.maximum(hists[m]["dnorm"][frame], 1e-13)
            gn = np.maximum(hists[m]["gnorm"][frame], 1e-13)
            ax_grad.plot(layers_x, dn, "-o", color=COLORS[m], ms=4,
                         label=f"{shorts[m]}  RMS($\\delta$)")
            ax_grad.plot(layers_x, gn, ":", color=COLORS[m], lw=1, alpha=0.6,
                         label=f"{shorts[m]}  RMS($\\partial L/\\partial w$)")
        ax_grad.set_yscale("log")
        ax_grad.set_ylim(gn_lo, gn_hi)
        ax_grad.set_xlabel("layer  (1 = near input,  higher = near output)")
        ax_grad.set_title("Backprop signal by depth  "
                          "($\\delta = \\partial L/\\partial z$, current epoch)")
        ax_grad.grid(True, which="both", linestyle="--", alpha=0.3)
        ax_grad.legend(loc="lower right", fontsize=6, ncol=2)

        # mean f'(z) per layer, with the cross-layer product in the legend.
        ax_der.clear()
        hid_x = np.arange(1, n_hid + 1)
        for m, mk in enumerate(("-o", "--s")):
            d = np.asarray(hists[m]["act_deriv"][frame])
            ax_der.plot(hid_x, np.maximum(d, 1e-13), mk, color=COLORS[m], ms=4,
                        label=f"{shorts[m]}  $\\prod$={np.prod(np.maximum(d, 1e-13)):.1e}")
        ax_der.axhline(0.25, color="grey", lw=0.8, linestyle=":")
        ax_der.text(1, 0.26, "sigmoid' max = 0.25", fontsize=6, color="grey")
        ax_der.set_yscale("log")
        ax_der.set_ylim(ad_lo, ad_hi)
        ax_der.set_xlabel("hidden layer")
        ax_der.set_title("mean $f'(z)$ per layer  (product = gradient attenuation factor)")
        ax_der.grid(True, which="both", linestyle="--", alpha=0.3)
        ax_der.legend(loc="lower left", fontsize=7)

        r = [hists[m]["dnorm"][frame][0] / max(hists[m]["dnorm"][frame][-1], 1e-13)
             for m in range(2)]
        fig.suptitle(
            f"{source}   |   epoch {e:4d}/{EPOCHS}   |   "
            f"val acc  sigmoid {hists[0]['vacc'][frame]:.1f}%  vs  relu {hists[1]['vacc'][frame]:.1f}%"
            f"   |   $\\|\\delta_{{L1}}\\|/\\|\\delta_{{last}}\\|$:  "
            f"sigmoid {r[0]:.1e}  vs  relu {r[1]:.1e}"
            f"   |   heatmap: {view['mode']}  (keys g / w)",
            fontsize=12,
        )
        return ()

    print("Rendering animation...")
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=FRAME_INTERVAL, blit=False, repeat=False)
    plt.show()
    return anim


def main() -> None:
    """Load data, train the Sigmoid and ReLU deep nets, then show the animation."""
    source, Xtr, ytr, Xval, yval = load_data()
    n_features = Xtr.shape[1]
    print(f"Data   : {source}  (train {Xtr.shape}, val {Xval.shape})")
    print(f"Model  : {n_features} -> {DEPTH} x {WIDTH} -> 1   (depth {DEPTH})")

    hists: list[dict] = []
    for cfg in MODELS:
        np.random.seed(SEED)  # same seed per build for reproducibility
        model = build_model(n_features, cfg["act"], cfg["init"])
        print(f"Training [{cfg['act']} + {cfg['init']}] ...")
        hists.append(train_and_record(model, Xtr, ytr, Xval, yval))

    layer_sizes = [n_features, *([WIDTH] * DEPTH), 1]
    animate(source, layer_sizes, (hists[0], hists[1]))


if __name__ == "__main__":
    main()
