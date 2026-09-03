from typing import Iterator, Optional, Tuple
import numpy as np


def iter_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    random_state: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset into consecutive mini-batches for one training epoch.

    Mini-batch gradient descent estimates the full-batch gradient from a small
    random subset of the data each step: cheaper per update than full-batch and
    less noisy than a single sample. Calling this once per epoch yields every
    row exactly once, grouped into slices of at most ``batch_size``.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Target array whose first axis is aligned with ``X`` (shape
            (n_samples, ...)).
        batch_size: Maximum number of samples per yielded batch. Must be >= 1.
        shuffle: If True, permute the rows before slicing so each epoch sees a
            different batch composition.
        drop_last: If True, discard the final batch when it holds fewer than
            ``batch_size`` samples; otherwise yield the shorter remainder.
        random_state: Seed for the shuffle permutation, for reproducible epochs.

    Yields:
        (X_batch, y_batch): Views into permuted copies of ``X`` and ``y``, with
        matching first-axis length (``batch_size``, or fewer for the trailing
        batch when ``drop_last`` is False).

    Raises:
        ValueError: If ``batch_size`` < 1, or if ``X`` and ``y`` disagree on
            the number of samples.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if len(X) != len(y):
        raise ValueError(f"X and y sample count mismatch: {len(X)} vs {len(y)}")

    n_samples = len(X)
    if shuffle:
        rng     = np.random.RandomState(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        if drop_last and len(batch_idx) < batch_size:
            break
        yield X[batch_idx], y[batch_idx]
