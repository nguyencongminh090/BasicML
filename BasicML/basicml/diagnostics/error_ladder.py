from dataclasses import dataclass
from typing      import Callable, Tuple

import numpy as np

from basicml.nn.module import Module

Split   = Tuple[np.ndarray, np.ndarray]
ErrorFn = Callable[[Module, np.ndarray, np.ndarray], float]


def classification_error(model: Module, X: np.ndarray, y: np.ndarray,
                         threshold: float = 0.5) -> float:
    """Fraction of misclassified samples for a binary classifier.

    Runs ``model`` in eval mode over ``X`` and compares thresholded
    probabilities against ``y``. This is ``1 - accuracy`` -- the "error"
    quantity the bias/variance ladder is built from.

    Args:
        model: Any ``Module`` whose forward pass returns per-sample scores of
            shape (n_samples, 1) in ``[0, 1]`` (e.g. a network ending in
            ``Sigmoid``).
        X: Feature array of shape (n_samples, n_features).
        y: Ground-truth label array of shape (n_samples, 1), entries in
            ``{0, 1}``.
        threshold: Decision threshold; a score ``>= threshold`` predicts class 1.

    Returns:
        The misclassification rate in ``[0, 1]``.
    """
    model.eval()
    scores = np.asarray(model(X)).reshape(-1)
    preds  = (scores >= threshold).astype(np.float64)
    truth  = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(np.mean(preds != truth))


@dataclass(frozen=True)
class ErrorLadder:
    """The five-rung error ladder from Ng's bias/variance recipe.

    Rungs (each a scalar error, lower is better), ordered so the model sees
    progressively less familiar data:

    - ``human``: human-level / Bayes-error proxy -- the best achievable error.
    - ``train``: error on the training set.
    - ``train_dev``: error on held-out data drawn from the *training*
      distribution (same distribution as ``train``, never optimised against).
    - ``dev``: error on the dev/validation set (target distribution).
    - ``test``: error on the test set (target distribution, touched once).

    The gap between consecutive rungs isolates one problem:

    - ``avoidable_bias`` = ``train - human``     -> underfitting
    - ``variance``       = ``train_dev - train`` -> overfitting the training rows
    - ``data_mismatch``  = ``dev - train_dev``   -> train/target distributions differ
    - ``dev_overfit``    = ``test - dev``        -> choices overfit the dev set
    """

    human: float
    train: float
    train_dev: float
    dev: float
    test: float

    @property
    def avoidable_bias(self) -> float:
        """``train - human``: the reducible part of the training error."""
        return self.train - self.human

    @property
    def variance(self) -> float:
        """``train_dev - train``: generalisation gap within the train distribution."""
        return self.train_dev - self.train

    @property
    def data_mismatch(self) -> float:
        """``dev - train_dev``: error from a train/target distribution shift."""
        return self.dev - self.train_dev

    @property
    def dev_overfit(self) -> float:
        """``test - dev``: how much model/hyper-parameter choices overfit dev."""
        return self.test - self.dev

    def gaps(self) -> dict[str, float]:
        """Return the four named gaps keyed by cause.

        Returns:
            Mapping ``{"avoidable bias", "variance", "data mismatch",
            "dev overfitting"} -> gap size``. Values can be negative when a
            later rung happens to score better than an earlier one (noise on
            small splits).
        """
        return {
            "avoidable bias":  self.avoidable_bias,
            "variance":        self.variance,
            "data mismatch":   self.data_mismatch,
            "dev overfitting": self.dev_overfit,
        }

    def diagnosis(self, tolerance: float = 1e-9) -> str:
        """Name the single largest gap as the dominant problem to fix next.

        Args:
            tolerance: A gap this size or smaller is treated as noise. When the
                largest gap does not exceed ``tolerance`` the ladder is reported
                as flat rather than pointing at a rung. Set this to roughly the
                sampling error of your smallest split (e.g. ``1 / n_dev``).

        Returns:
            A short verdict, e.g.
            ``"variance (0.0810) dominates -> regularise or get more training data"``,
            or a "ladder is flat" message when no gap exceeds ``tolerance``.
        """
        cause, size = max(self.gaps().items(), key=lambda item: item[1])
        if size <= tolerance:
            return f"ladder is flat -- every gap <= {tolerance:.4f}"
        advice = {
            "avoidable bias":  "increase model capacity / train longer",
            "variance":        "regularise or get more training data",
            "data mismatch":   "make training data resemble the target distribution",
            "dev overfitting": "enlarge the dev set / tune fewer choices against it",
        }[cause]
        return f"{cause} ({size:.4f}) dominates -> {advice}"

    def summary(self, tolerance: float = 1e-9) -> str:
        """Render the ladder, its gaps, and the diagnosis as printable lines.

        Args:
            tolerance: Forwarded to :meth:`diagnosis` -- gaps at or below this
                size are treated as noise.
        """
        rungs = [
            ("human",     self.human),
            ("train",     self.train),
            ("train-dev", self.train_dev),
            ("dev",       self.dev),
            ("test",      self.test),
        ]
        lines  = ["error ladder", "-" * 40]
        lines += [f"  {name:<16} {value:.4f}" for name, value in rungs]
        lines += ["gaps", "-" * 40]
        lines += [f"  {name:<16} {value:+.4f}" for name, value in self.gaps().items()]
        lines += ["-" * 40, self.diagnosis(tolerance)]
        return "\n".join(lines)


def error_ladder(
    model: Module,
    train: Split,
    train_dev: Split,
    dev: Split,
    test: Split,
    human_error: float = 0.0,
    error_fn: ErrorFn = classification_error,
) -> ErrorLadder:
    """Evaluate a trained model on every split and assemble an ``ErrorLadder``.

    The model is only read here -- call this once training has finished. Each
    split is scored with ``error_fn`` (which is expected to put the model in
    eval mode).

    Args:
        model: The trained ``Module`` to evaluate.
        train: ``(X, y)`` for the training set the model was fitted on.
        train_dev: ``(X, y)`` held out from the training distribution and never
            trained on -- this rung is what separates variance from data
            mismatch.
        dev: ``(X, y)`` dev/validation set from the target distribution.
        test: ``(X, y)`` test set from the target distribution.
        human_error: Human-level / Bayes-error proxy for the bottom rung
            (default ``0.0`` assumes a near-perfect achievable error).
        error_fn: ``(model, X, y) -> float`` scoring function; defaults to
            :func:`classification_error` (misclassification rate).

    Returns:
        An :class:`ErrorLadder` with all five rungs populated.
    """
    return ErrorLadder(
        human     = human_error,
        train     = error_fn(model, *train),
        train_dev = error_fn(model, *train_dev),
        dev       = error_fn(model, *dev),
        test      = error_fn(model, *test),
    )
