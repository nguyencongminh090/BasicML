from typing import Callable, Optional, Union, List
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(
    pred_func: Union[Callable[[np.ndarray], np.ndarray], object],
    X: np.ndarray,
    y: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: Union[str, plt.cm.ScalarMappable] = 'Spectral',
    h: float = 0.01,
    padding: float = 0.5,
    title: Optional[str] = None,
    show: bool = True
) -> plt.Axes:
    """Plot the 2D decision boundary for a binary or multiclass classifier.

    Args:
        pred_func: A prediction function or model callable that maps
                   an (N, 2) NumPy array to predictions / probabilities / class indices.
        X: Feature array of shape (N, 2).
        y: Labels of shape (N, 1) or (N,).
        ax: Optional matplotlib Axes. If None, a new figure is created.
        cmap: Matplotlib colormap name or Colormap instance (default: 'Spectral').
        h: Step size of the meshgrid.
        padding: Padding added around min/max data coordinates.
        title: Title of the subplot / figure.
        show: Whether to call plt.show() when ax is None or show is requested.

    Returns:
        The matplotlib Axes object containing the plot.
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Handle model instances or raw functions
    if callable(pred_func):
        out = pred_func(grid_points)
    elif hasattr(pred_func, 'forward'):
        out = pred_func.forward(grid_points)
    else:
        raise ValueError("pred_func must be a callable or have a forward() method")

    # Unwrap Tensor if needed
    if hasattr(out, 'data'):
        out = out.data

    Z = np.asarray(out)

    # Convert predictions to class labels
    if Z.ndim > 1 and Z.shape[1] > 1:
        # Multiclass output (softmax / logits)
        Z = np.argmax(Z, axis=1)
    elif Z.ndim > 1 and Z.shape[1] == 1:
        # Binary classification probabilities / continuous output
        Z = (Z >= 0.5).astype(int)
    elif Z.ndim == 1:
        if np.issubdtype(Z.dtype, np.floating):
            Z = (Z >= 0.5).astype(int)

    Z = Z.reshape(xx.shape)

    standalone = False
    if ax is None:
        standalone = True
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot decision boundary regions
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
    
    # Plot decision boundary contour line
    ax.contour(xx, yy, Z, levels=[0.5], colors='yellow', linewidths=1.5, alpha=0.9)

    # Plot sample data points
    y_flat = np.asarray(y).ravel()
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y_flat,
        cmap=cmap,
        edgecolors='k',
        s=30,
        alpha=0.9
    )

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    if title:
        ax.set_title(title)

    if standalone and show:
        plt.show()

    return ax


def plot_dataset_2d(
    X: np.ndarray,
    y: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'Spectral',
    title: Optional[str] = "2D Dataset Scatter",
    show: bool = True
) -> plt.Axes:
    """Helper to visualize 2D classification data."""
    standalone = False
    if ax is None:
        standalone = True
        fig, ax = plt.subplots(figsize=(8, 6))

    y_flat = np.asarray(y).ravel()
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y_flat,
        cmap=cmap,
        edgecolors='k',
        s=35,
        alpha=0.9
    )

    if title:
        ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

    if standalone and show:
        plt.show()

    return ax


def plot_loss_curve(
    loss_history: Union[List[float], np.ndarray],
    ax: Optional[plt.Axes] = None,
    title: str = "Training Loss",
    color: str = "royalblue",
    show: bool = True
) -> plt.Axes:
    """Helper to visualize training loss curve."""
    standalone = False
    if ax is None:
        standalone = True
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(loss_history, color=color, linewidth=2, label="Cost")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    if standalone and show:
        plt.show()

    return ax
