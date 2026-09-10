from typing        import Optional
from numpy.typing  import ArrayLike
import numpy as np

from basicml.metrics import Metric


def format_metric_line(
    epoch: int,
    total: int,
    values: dict[str, float],
    prefix: str = "",
) -> str:
    """Format one PyTorch-style progress line for an epoch.

    Produces a line such as::

        epoch  12/400 | loss: 0.0412 | mae: 0.1523 | r2: 0.9812

    Args:
        epoch: 1-based index of the epoch just finished.
        total: Total number of epochs; used to right-pad ``epoch`` so the
            counter column stays a fixed width across the whole run.
        values: Ordered mapping of column name to scalar value. Insertion order
            is preserved in the output, so pass ``loss`` first.
        prefix: Optional tag placed before the counter (e.g. ``"val"``); an
            empty string adds nothing.

    Returns:
        The formatted line, without a trailing newline.
    """
    width   = len(str(total))
    counter = f"epoch {epoch:>{width}}/{total}"
    if prefix:
        counter = f"{prefix} {counter}"
    cells = " | ".join(f"{name}: {value:.4f}" for name, value in values.items())
    return f"{counter} | {cells}" if cells else counter


class ProgressPrinter:
    """Accumulate metrics over an epoch and print a progress line per epoch.

    The core library owns the stateful :class:`~basicml.metrics.Metric` objects;
    this helper only drives them and renders the result, so a training loop
    reports loss plus any metrics the caller cares about without the loop
    growing formatting code. Typical use::

        printer = ProgressPrinter({
            "mae": MeanAbsoluteError(),
            "r2":  R2Score(),
        })
        for epoch in range(1, EPOCHS + 1):
            for xb, yb in batches:
                y_pred = model(xb)
                loss   = criterion(y_pred, yb)
                ...
                printer.update(y_pred, yb, loss)
            printer.end_epoch(epoch, EPOCHS)

    Args:
        metrics: Mapping of column name to a scalar-valued ``Metric``. Metrics
            that ``compute()`` to an array (e.g. per-class precision) are not
            supported.
        every: Only print on epochs where ``epoch % every == 0``, plus the final
            epoch. Defaults to every epoch.
        loss_key: Column name used for the running mean loss. Defaults to
            ``"loss"``.
    """

    def __init__(self, metrics: dict[str, Metric], every: int = 1,
                 loss_key: str = "loss"):
        self.metrics  = metrics
        self.every    = every
        self.loss_key = loss_key
        self.reset()

    def reset(self) -> None:
        """Zero the running loss and reset every tracked metric."""
        self._loss_total   = 0.0
        self._loss_batches = 0
        for metric in self.metrics.values():
            metric.reset()

    def update(self, y_pred: ArrayLike, y_true: ArrayLike, loss: float) -> None:
        """Feed one batch's predictions, targets, and scalar loss.

        Args:
            y_pred: Model output for the batch.
            y_true: Ground-truth targets for the batch.
            loss: The batch's scalar loss value.
        """
        self._loss_total   += loss
        self._loss_batches += 1
        for metric in self.metrics.values():
            metric.update(y_pred, y_true)

    def values(self) -> dict[str, float]:
        """Return the current epoch's mean loss and computed metric values."""
        mean_loss = self._loss_total / self._loss_batches if self._loss_batches else 0.0
        line: dict[str, float] = {self.loss_key: mean_loss}
        for name, metric in self.metrics.items():
            result = metric.compute()
            if not isinstance(result, (int, float, np.floating, np.integer)):
                raise TypeError(
                    f"metric {name!r} is not scalar-valued "
                    f"(compute() returned {type(result).__name__})"
                )
            line[name] = float(result)
        return line

    def end_epoch(self, epoch: int, total: int, prefix: str = "") -> Optional[str]:
        """Print the progress line for ``epoch`` (subject to ``every``) and reset.

        Args:
            epoch: 1-based index of the epoch just finished.
            total: Total number of epochs.
            prefix: Optional tag forwarded to :func:`format_metric_line`.

        Returns:
            The printed line, or ``None`` when ``every`` suppressed this epoch.
        """
        show = epoch % self.every == 0 or epoch == total
        line = format_metric_line(epoch, total, self.values(), prefix) if show else None
        if line is not None:
            print(line)
        self.reset()
        return line
