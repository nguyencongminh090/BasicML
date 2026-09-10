from typing import Optional, Tuple
import numpy as np


def make_moons(
    n_samples: int = 200,
    noise: Optional[float] = 0.20,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving half circles (moons dataset).

    Args:
        n_samples: Total number of points generated.
        noise: Standard deviation of Gaussian noise added to the data.
        random_state: Seed for reproducible random numbers.

    Returns:
        X: Feature array of shape (n_samples, 2).
        y: Binary label array of shape (n_samples, 1).
    """
    rng = np.random.RandomState(random_state)
    n_samples_out = n_samples // 2
    n_samples_in  = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1.0 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1.0 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack([
        np.append(outer_circ_x, inner_circ_x),
        np.append(outer_circ_y, inner_circ_y)
    ]).T
    y = np.hstack([
        np.zeros(n_samples_out, dtype=int),
        np.ones(n_samples_in, dtype=int)
    ]).reshape(-1, 1)

    if noise is not None and noise > 0:
        X += rng.normal(scale=noise, size=X.shape)

    # Shuffle samples
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def make_circles(
    n_samples: int = 200,
    noise: Optional[float] = 0.10,
    factor: float = 0.5,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate large circle containing a smaller inner circle.

    Args:
        n_samples: Total number of points generated.
        noise: Standard deviation of Gaussian noise added to the data.
        factor: Scale factor between inner and outer circle (0 < factor < 1).
        random_state: Seed for reproducible random numbers.

    Returns:
        X: Feature array of shape (n_samples, 2).
        y: Binary label array of shape (n_samples, 1).
    """
    rng = np.random.RandomState(random_state)
    n_samples_out = n_samples // 2
    n_samples_in  = n_samples - n_samples_out

    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in  = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)

    outer_x = np.cos(linspace_out)
    outer_y = np.sin(linspace_out)
    inner_x = np.cos(linspace_in) * factor
    inner_y = np.sin(linspace_in) * factor

    X = np.vstack([
        np.append(outer_x, inner_x),
        np.append(outer_y, inner_y)
    ]).T
    y = np.hstack([
        np.zeros(n_samples_out, dtype=int),
        np.ones(n_samples_in, dtype=int)
    ]).reshape(-1, 1)

    if noise is not None and noise > 0:
        X += rng.normal(scale=noise, size=X.shape)

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def make_parabola_regression(
    n_samples: int = 200,
    x1_range: Tuple[float, float] = (-3.0, 3.0),
    curvature: float = 1.0,
    feature_noise: float = 0.3,
    weights: Tuple[float, float] = (2.0, -1.0),
    bias: float = 1.0,
    target_noise: float = 0.5,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 2-feature linear regression dataset whose inputs scatter around a parabola.

    ``x1`` is drawn uniformly from ``x1_range``; ``x2`` follows the curve
    ``x2 = curvature * x1**2`` with Gaussian noise added, so the point cloud
    in the ``(x1, x2)`` plane is scattered around a parabola. The regression
    target is then a noisy linear combination of the two features:
    ``y = weights[0] * x1 + weights[1] * x2 + bias + noise``.

    Args:
        n_samples: Total number of points generated.
        x1_range: ``(min, max)`` range that ``x1`` is sampled uniformly from.
        curvature: Coefficient of the ``x1**2`` term defining the parabola.
        feature_noise: Standard deviation of Gaussian noise added to ``x2``
            around the parabola.
        weights: True ``(w1, w2)`` coefficients used to compute the target.
        bias: True bias term used to compute the target.
        target_noise: Standard deviation of Gaussian noise added to the
            target ``y``.
        random_state: Seed for reproducible random numbers.

    Returns:
        X: Feature array of shape (n_samples, 2), columns ``(x1, x2)``.
        y: Target array of shape (n_samples, 1).
    """
    rng = np.random.RandomState(random_state)
    x1 = rng.uniform(*x1_range, size=n_samples)
    x2 = curvature * x1 ** 2 + rng.normal(scale=feature_noise, size=n_samples)

    X = np.stack([x1, x2], axis=1)
    y = (weights[0] * x1 + weights[1] * x2 + bias
         + rng.normal(scale=target_noise, size=n_samples))
    return X, y.reshape(-1, 1)


def make_blob_images(
    n_samples: int = 240,
    size: int = 12,
    blob_sigma: float = 2.0,
    noise: float = 0.15,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a tiny binary image dataset: one Gaussian blob per image.

    Each image is a single-channel ``size x size`` grid holding one bright
    Gaussian bump plus additive noise. Class ``0`` places the bump in the
    top-left quadrant, class ``1`` in the bottom-right quadrant, so the label
    depends on *where* the activation is, not on any single pixel -- a small
    task that a convolution followed by pooling can solve but a bare linear
    model on raw pixels struggles with.

    Args:
        n_samples: Total number of images generated; split as evenly as
            possible between the two classes.
        size: Height and width of each square image in pixels.
        blob_sigma: Standard deviation, in pixels, of the Gaussian bump.
        noise: Standard deviation of the per-pixel Gaussian noise added on top
            of the bump.
        random_state: Seed for reproducible random numbers.

    Returns:
        X: Image array of shape (n_samples, 1, size, size), float64.
        y: Binary label array of shape (n_samples, 1).
    """
    rng      = np.random.RandomState(random_state)
    n_class1 = n_samples // 2
    n_class0 = n_samples - n_class1
    labels   = np.concatenate([np.zeros(n_class0, dtype=int), np.ones(n_class1, dtype=int)])

    rows, cols = np.mgrid[0:size, 0:size]
    quarter    = size / 4.0
    centers    = np.where(labels[:, None] == 0, quarter, 3.0 * quarter)

    images = np.empty((n_samples, 1, size, size), dtype=np.float64)
    for k in range(n_samples):
        cy, cx        = centers[k], centers[k]
        bump          = np.exp(-((rows - cy) ** 2 + (cols - cx) ** 2) / (2.0 * blob_sigma ** 2))
        images[k, 0]  = bump + rng.normal(scale=noise, size=(size, size))

    indices = rng.permutation(n_samples)
    return images[indices], labels[indices].reshape(-1, 1)
