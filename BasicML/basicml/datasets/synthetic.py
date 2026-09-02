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
