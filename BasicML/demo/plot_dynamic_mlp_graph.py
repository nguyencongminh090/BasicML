# AI generated (refactored/authored with Claude Code)
"""Live neuron-graph view of a deep MLP as it trains.

Run directly: python BasicML/demo/plot_dynamic_mlp_graph.py

Two models are trained side by side from an identical weight initialization:
  - left : plain MLP, no Dropout, no regularization
  - right: MLP + Dropout(p) on the hidden layers + L2 weight penalty

Each column shows:
  1. A sparse neuron graph (top-k strongest edges per layer pair); edge colour
     is the weight sign and width/alpha scales with |weight|. Wide layers are
     sub-sampled down to MAX_NODES neurons.
  2. A row of per-Linear-layer heatmaps (RdBu, centred at 0) showing either the
     weights `w` or the gradient `dL/dw`; toggle with the `g` / `w` keys.
  3. BCE-loss (train) and validation-accuracy curves for both models.
  4. Per-layer `||dL/dw||` over epochs on a log scale (to spot gradients shrinking
     toward the input layers), plus mean(|ReLU activation|) and the % of dead
     units per hidden layer.

The gradient shown is the plain backprop gradient: the L2 term is added by the
optimizer at `step()` and never enters `w.grad`, which is what makes `w.grad`
the right quantity for observing vanishing gradients.

Dataset: sklearn `load_breast_cancer` (30 features, binary) when available,
otherwise `basicml.datasets.make_moons` (2 features).
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from basicml.nn.sequential   import Sequential
from basicml.nn.linear       import Linear
from basicml.nn.activation   import ReLU, Sigmoid
from basicml.nn.dropout      import Dropout
from basicml.nn.loss         import BinaryCrossEntropy
from basicml.optim.momentum  import Momentum
from basicml.regularization  import L2, Regularizer

np.set_printoptions(suppress=True, precision=4)

# --- CONFIG ------------------------------------------------------------------
SEED           = 7
HIDDEN         = (64, 64, 64, 32)      # hidden widths -> a deep, over-parameterized net
DROPOUT_P      = 0.30
L2_LAMBDA      = 1e-3

LEARN_RATE     = 0.05
MOMENTUM       = 0.9
EPOCHS         = 1200
RECORD_EVERY   = 8                     # snapshot weights/grads every N epochs

MAX_NODES      = 12                    # neurons drawn per layer in the graph
TOPK_EDGES     = 40                    # strongest edges drawn per layer pair
HEATMAP_MODE   = "weight"              # "weight" | "grad"; toggled with the g / w keys
FRAME_INTERVAL = 40                    # milliseconds between frames
FIG_SIZE       = (18, 13)
# ---------------------------------------------------------------------------


def load_data() -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a binary-classification dataset, split it, and standardize the features.

    Prefers sklearn's breast-cancer dataset and falls back to a synthetic
    two-moons set so the demo runs without sklearn installed. Features are
    standardized using train-split statistics only.

    Returns:
        A tuple ``(source, Xtr, ytr, Xval, yval)`` where ``source`` is a short
        human-readable name of the dataset, ``Xtr`` / ``Xval`` are ``(N, F)``
        float arrays and ``ytr`` / ``yval`` are ``(N, 1)`` float label arrays.
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

    rng          = np.random.RandomState(SEED)
    perm         = rng.permutation(len(X))
    X, y         = X[perm], y[perm]
    n_tr         = int(0.7 * len(X))
    Xtr, Xval    = X[:n_tr], X[n_tr:]
    ytr, yval    = y[:n_tr], y[n_tr:]

    mu, sd       = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr          = (Xtr - mu) / sd
    Xval         = (Xval - mu) / sd
    return source, Xtr, ytr, Xval, yval


def build_model(n_features: int, use_dropout: bool) -> Sequential:
    """Build the demo MLP: ``[n_features] -> HIDDEN (ReLU) -> 1 (Sigmoid)``.

    Args:
        n_features: Number of input features; sets the first layer's fan-in.
        use_dropout: If True, insert ``Dropout(DROPOUT_P)`` after every hidden
            ReLU. The output layer never gets dropout.

    Returns:
        The assembled ``Sequential`` model. Hidden layers use He init, the
        output layer uses Xavier init.
    """
    sizes:  list[int]    = [n_features, *HIDDEN]
    layers: list         = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], init_type="he"))
        layers.append(ReLU())
        if use_dropout:
            layers.append(Dropout(p=DROPOUT_P))
    layers.append(Linear(sizes[-1], 1, init_type="xavier"))
    layers.append(Sigmoid())
    return Sequential(*layers)


def linear_layers(model: Sequential) -> list[Linear]:
    """Return the model's ``Linear`` layers in forward order.

    Args:
        model: A ``Sequential`` whose ``layers`` list is scanned.

    Returns:
        The ``Linear`` instances only, skipping activations and dropout.
    """
    return [layer for layer in model.layers if isinstance(layer, Linear)]


def weight_grad(layer: Linear) -> np.ndarray:
    """Return a Linear layer's weight gradient, asserting it has been populated.

    Args:
        layer: The ``Linear`` layer to read ``w.grad`` from.

    Returns:
        The ``(in_features, out_features)`` gradient array.

    Raises:
        RuntimeError: If ``w.grad`` is None, i.e. ``model.backward()`` has not
            run since the last ``zero_grad()``.
    """
    g = layer.w.grad
    if g is None:
        raise RuntimeError("weight gradient is None; call model.backward() first")
    return g


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary accuracy as a percentage, thresholding predictions at 0.5.

    Args:
        y_pred: Predicted probabilities, shape ``(N, 1)``.
        y_true: Ground-truth labels (0/1), shape ``(N, 1)``.

    Returns:
        Accuracy in the range ``[0, 100]``.
    """
    return float(np.mean((y_pred >= 0.5).astype(int) == y_true.astype(int))) * 100.0


def train_and_record(
    model: Sequential,
    reg  : Regularizer | None,
    Xtr  : np.ndarray, ytr: np.ndarray,
    Xval : np.ndarray, yval: np.ndarray,
) -> dict:
    """Full-batch train a model and snapshot its state for the animation.

    Every ``RECORD_EVERY`` epochs it records: weight matrices, weight-gradient
    matrices, per-layer gradient norms (read before ``zero_grad()``), train/val
    accuracy, the L2 penalty, and per-hidden-layer ReLU activation statistics.

    Args:
        model: The MLP to train; mutated in place.
        reg: L2 (or other) regularizer passed to the optimizer, or None for the
            plain model. Only its gradient contribution is applied at ``step()``;
            it does not enter ``w.grad``.
        Xtr, ytr: Training features ``(N, F)`` and labels ``(N, 1)``.
        Xval, yval: Validation features and labels, same shapes.

    Returns:
        A dict of history lists keyed by ``"epochs"``, ``"loss"``, ``"tracc"``,
        ``"vacc"``, ``"penalty"``, ``"weights"``, ``"grad"``, ``"gnorm"``,
        ``"act_mean"`` and ``"act_dead"``. Each entry has one item per recorded
        epoch.
    """
    criterion = BinaryCrossEntropy()
    optimizer = Momentum(model.parameters(), lr=LEARN_RATE,
                         regularizer=reg, momentum=MOMENTUM)
    linears   = linear_layers(model)
    relus     = [layer for layer in model.layers if isinstance(layer, ReLU)]

    epochs_seen : list[int]              = []
    loss_hist   : list[float]            = []
    tracc_hist  : list[float]            = []
    vacc_hist   : list[float]            = []
    penalty_hist: list[float]            = []
    weight_hist : list[list[np.ndarray]] = []
    grad_hist   : list[list[np.ndarray]] = []
    gnorm_hist  : list[list[float]]      = []
    act_mean_h  : list[list[float]]      = []
    act_dead_h  : list[list[float]]      = []

    for epoch in range(EPOCHS):
        model.train()
        y_pred = model(Xtr)
        loss   = criterion(y_pred, ytr)
        model.backward(criterion.backward())

        record = epoch % RECORD_EVERY == 0 or epoch == EPOCHS - 1
        if record:
            # Read gradients before the optimizer clears them.
            grad_hist.append([weight_grad(l).copy() for l in linears])
            gnorm_hist.append([float(np.linalg.norm(weight_grad(l))) for l in linears])

        optimizer.step()
        optimizer.zero_grad()

        if record:
            model.eval()
            val_pred = model(Xval)
            tr_pred  = model(Xtr)  # last forward leaves relu.out holding train activations
            epochs_seen.append(epoch)
            loss_hist.append(float(loss))
            tracc_hist.append(accuracy(tr_pred, ytr))
            vacc_hist.append(accuracy(val_pred, yval))
            penalty_hist.append(
                float(sum(reg.penalty(l.w) for l in linears)) if reg is not None else 0.0
            )
            weight_hist.append([l.w.data.copy() for l in linears])
            act_mean_h.append([
                float(np.mean(np.abs(r.out))) if r.out is not None else 0.0 for r in relus
            ])
            act_dead_h.append([
                float(np.mean(r.out == 0.0)) * 100.0 if r.out is not None else 0.0
                for r in relus
            ])

    print(f"  done: loss {loss_hist[-1]:.4f} | "
          f"train acc {tracc_hist[-1]:.1f}% | val acc {vacc_hist[-1]:.1f}%")
    return {
        "epochs"  : epochs_seen,
        "loss"    : loss_hist,
        "tracc"   : tracc_hist,
        "vacc"    : vacc_hist,
        "penalty" : penalty_hist,
        "weights" : weight_hist,
        "grad"    : grad_hist,
        "gnorm"   : gnorm_hist,
        "act_mean": act_mean_h,
        "act_dead": act_dead_h,
    }


def linear_sizes(sizes: list[int]) -> list[int]:
    """Return the output sizes of every Linear layer (hidden widths plus the 1-unit head).

    Args:
        sizes: ``[n_features, *HIDDEN]`` — the input size followed by hidden widths.

    Returns:
        ``[*HIDDEN, 1]``: one entry per Linear layer.
    """
    return [*sizes, 1][1:]


def node_layout(sizes: list[int]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Compute which neurons to draw and their screen coordinates per layer.

    Args:
        sizes: Neuron count of each layer to draw, input to output.

    Returns:
        ``(disp, ys, xs)`` where ``disp[j]`` are the neuron indices shown for
        layer ``j`` (sub-sampled to at most ``MAX_NODES``), ``ys[j]`` their
        vertical positions in ``[0, 1]``, and ``xs`` the horizontal position of
        each layer.
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
        weights: Per-layer weight matrices, layer ``j`` shaped
            ``(len(disp[j]) source, len(disp[j+1]) target)`` before sub-sampling.
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
        ax.scatter(np.full(len(ys[j]), xj), ys[j], s=80, color="white",
                   edgecolors="#34495e", linewidths=1.3, zorder=2)


def animate(source: str, sizes: list[int], hist_plain: dict, hist_reg: dict) -> FuncAnimation:
    """Build and show the side-by-side training animation.

    Args:
        source: Dataset name for the figure title.
        sizes: ``[n_features, *HIDDEN]``.
        hist_plain: History dict from :func:`train_and_record` for the plain model.
        hist_reg: History dict for the Dropout + L2 model.

    Returns:
        The ``FuncAnimation`` (kept alive by the caller / ``plt.show``).
    """
    hists        = (hist_plain, hist_reg)
    n_lin        = len(linear_sizes(sizes))
    n_hid        = len(HIDDEN)
    disp, ys, xs = node_layout(sizes + [1])

    wmax = max(float(np.abs(W).max())
               for h in hists for snap in h["weights"] for W in snap)
    gmax = max((float(np.abs(G).max())
                for h in hists for snap in h["grad"] for G in snap), default=1.0) or 1.0

    gnorms       = [g for h in hists for row in h["gnorm"] for g in row if g > 0.0]
    gn_lo, gn_hi = (min(gnorms) * 0.5, max(gnorms) * 2.0) if gnorms else (1e-8, 1.0)
    acts         = [a for h in hists for row in h["act_mean"] for a in row if a > 0.0]
    act_lo, act_hi = (min(acts) * 0.5, max(acts) * 2.0) if acts else (1e-3, 1.0)

    n_frames = min(len(hist_plain["epochs"]), len(hist_reg["epochs"]))
    max_loss = max(hist_plain["loss"][0], hist_reg["loss"][0]) * 1.05

    fig = plt.figure(figsize=FIG_SIZE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("BasicML - Dynamic MLP Graph")
    outer = fig.add_gridspec(4, 2, height_ratios=[3.0, 1.35, 1.7, 1.8],
                             hspace=0.5, wspace=0.12)

    ax_graph = [fig.add_subplot(outer[0, c]) for c in range(2)]
    hm_ax    = [
        [fig.add_subplot(outer[1, c].subgridspec(1, n_lin, wspace=0.15)[0, i])
         for i in range(n_lin)]
        for c in range(2)
    ]
    ax_loss = fig.add_subplot(outer[2, 0])
    ax_acc  = fig.add_subplot(outer[2, 1])
    ax_grad = fig.add_subplot(outer[3, 0])
    ax_act  = fig.add_subplot(outer[3, 1])

    col_titles = [
        "plain MLP (no regularization)",
        f"MLP + Dropout(p={DROPOUT_P}) + L2($\\lambda$={L2_LAMBDA})",
    ]

    # Heatmap view state; toggled between weights and gradients by the g / w keys.
    view = {"mode": HEATMAP_MODE if HEATMAP_MODE in ("weight", "grad") else "weight"}

    def on_key(event) -> None:
        """Switch the heatmap view on `g` (gradient) / `w` (weight) key presses."""
        if event.key == "g":
            view["mode"] = "grad"
        elif event.key == "w":
            view["mode"] = "weight"
    fig.canvas.mpl_connect("key_press_event", on_key)

    loss_p, = ax_loss.plot([], [], color="#7f8c8d", lw=2, label="plain")
    loss_r, = ax_loss.plot([], [], color="#27ae60", lw=2, label="dropout+L2")
    ax_loss.set_xlim(0, EPOCHS)
    ax_loss.set_ylim(0, max_loss)
    ax_loss.set_title("BCE loss (train)")
    ax_loss.set_xlabel("epoch")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend(loc="upper right", fontsize=8)

    acc_p, = ax_acc.plot([], [], color="#7f8c8d", lw=2, label="plain")
    acc_r, = ax_acc.plot([], [], color="#27ae60", lw=2, label="dropout+L2")
    ax_acc.set_xlim(0, EPOCHS)
    ax_acc.set_ylim(40, 101)
    ax_acc.set_title("Validation accuracy (%)")
    ax_acc.set_xlabel("epoch")
    ax_acc.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend(loc="lower right", fontsize=8)

    # Per-Linear-layer gradient norm over epochs: plain = solid, reg = dashed,
    # colour graded by depth.
    depth_c    = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, n_lin))
    grad_lines = []
    for i in range(n_lin):
        gp, = ax_grad.plot([], [], color=depth_c[i], lw=1.8, label=f"L{i + 1}")
        gr, = ax_grad.plot([], [], color=depth_c[i], lw=1.4, linestyle="--")
        grad_lines.append((gp, gr))
    ax_grad.set_yscale("log")
    ax_grad.set_xlim(0, EPOCHS)
    ax_grad.set_ylim(gn_lo, gn_hi)
    ax_grad.set_title("$\\|\\partial L/\\partial w\\|$ per layer  (—— plain,  - - dropout+L2)")
    ax_grad.set_xlabel("epoch")
    ax_grad.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_grad.legend(loc="lower left", fontsize=7, ncol=n_lin)

    def update(frame: int):
        """Redraw every panel for the recorded epoch at index ``frame``."""
        for c, h in enumerate(hists):
            draw_graph(ax_graph[c], h["weights"][frame], disp, ys, xs, wmax)
            ax_graph[c].set_title(col_titles[c], fontsize=11)

            if view["mode"] == "grad":
                mats, vlim, lbl = h["grad"][frame], gmax, "|$\\partial L/\\partial w$|"
            else:
                mats, vlim, lbl = h["weights"][frame], wmax, "weights $w$"
            for i, ax in enumerate(hm_ax[c]):
                ax.clear()
                ax.imshow(mats[i].T, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"L{i + 1}", fontsize=8)
            hm_ax[c][0].set_ylabel(lbl, fontsize=8)

        e    = hist_plain["epochs"][frame]
        ep_p = hist_plain["epochs"][:frame + 1]
        ep_r = hist_reg["epochs"][:frame + 1]
        loss_p.set_data(ep_p, hist_plain["loss"][:frame + 1])
        loss_r.set_data(ep_r, hist_reg["loss"][:frame + 1])
        acc_p.set_data(ep_p, hist_plain["vacc"][:frame + 1])
        acc_r.set_data(ep_r, hist_reg["vacc"][:frame + 1])

        for i, (gp, gr) in enumerate(grad_lines):
            gp.set_data(ep_p, [row[i] for row in hist_plain["gnorm"][:frame + 1]])
            gr.set_data(ep_r, [row[i] for row in hist_reg["gnorm"][:frame + 1]])

        ax_act.clear()
        xh = np.arange(n_hid)
        mp, dp = hist_plain["act_mean"][frame], hist_plain["act_dead"][frame]
        mr, dr = hist_reg["act_mean"][frame],   hist_reg["act_dead"][frame]
        ax_act.bar(xh - 0.19, mp, width=0.38, color="#7f8c8d", label="plain")
        ax_act.bar(xh + 0.19, mr, width=0.38, color="#27ae60", label="dropout+L2")
        ax_act.set_yscale("log")
        ax_act.set_ylim(act_lo, act_hi)
        ax_act.set_xticks(xh)
        ax_act.set_xticklabels([f"L{k + 1}" for k in range(n_hid)])
        ax_act.set_title("mean(|ReLU activation|)  +  % dead units")
        ax_act.legend(loc="upper right", fontsize=7)
        for k in range(n_hid):
            ax_act.text(xh[k] - 0.19, mp[k], f"{dp[k]:.0f}%", ha="center", va="bottom", fontsize=6)
            ax_act.text(xh[k] + 0.19, mr[k], f"{dr[k]:.0f}%", ha="center", va="bottom", fontsize=6)

        fig.suptitle(
            f"{source}   |   epoch {e:4d}/{EPOCHS}   |   "
            f"val acc  plain {hist_plain['vacc'][frame]:.1f}%  vs  "
            f"dropout+L2 {hist_reg['vacc'][frame]:.1f}%   |   "
            f"R($\\theta$) = {hist_reg['penalty'][frame]:.3f}"
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
    """Load data, train the plain and Dropout+L2 models, then show the animation."""
    source, Xtr, ytr, Xval, yval = load_data()
    n_features = Xtr.shape[1]
    sizes      = [n_features, *HIDDEN]
    print(f"Data   : {source}  (train {Xtr.shape}, val {Xval.shape})")
    print(f"Model  : {sizes} -> 1   dropout p={DROPOUT_P}   L2 lambda={L2_LAMBDA}")

    # Seed before each build so both models start from identical weights.
    np.random.seed(SEED)
    model_plain = build_model(n_features, use_dropout=False)
    np.random.seed(SEED)
    model_reg   = build_model(n_features, use_dropout=True)
    np.random.seed(SEED)

    print("Training plain model...")
    hist_plain = train_and_record(model_plain, None, Xtr, ytr, Xval, yval)
    print("Training dropout + L2 model...")
    hist_reg   = train_and_record(model_reg, L2(L2_LAMBDA), Xtr, ytr, Xval, yval)

    animate(source, sizes, hist_plain, hist_reg)


if __name__ == "__main__":
    main()
