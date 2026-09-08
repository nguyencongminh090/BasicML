from abc          import ABC, abstractmethod
from typing       import Optional
from numpy.typing  import ArrayLike
import numpy as np

MetricValue = float | np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]


class Metric(ABC):
    @abstractmethod
    def update(self, y_pred: ArrayLike, y_true: ArrayLike) -> None:
        ...

    @abstractmethod
    def compute(self) -> MetricValue:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    def __call__(self, y_pred: ArrayLike, y_true: ArrayLike) -> MetricValue:
        self.update(y_pred, y_true)
        return self.compute()


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out   = np.zeros(np.broadcast(numerator, denominator).shape, dtype=float),
        where = denominator != 0,
    )


class ConfusionMatrix(Metric):
    def __init__(self, num_classes: int = 2, threshold: float = 0.5):
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.threshold   = threshold
        self.reset()

    def reset(self) -> None:
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self._count = 0

    def _true_labels(self, y_true: ArrayLike) -> np.ndarray:
        a = np.asarray(y_true)
        if a.ndim > 1 and a.shape[-1] == self.num_classes and self.num_classes > 2:
            return np.argmax(a, axis=-1).reshape(-1)
        return np.rint(a.reshape(-1).astype(float)).astype(np.int64)

    def _pred_labels(self, y_pred: ArrayLike) -> np.ndarray:
        a = np.asarray(y_pred, dtype=float)
        if a.ndim > 1 and a.shape[-1] == self.num_classes and self.num_classes > 2:
            return np.argmax(a, axis=-1).reshape(-1)
        a = a.reshape(-1)
        if self.num_classes == 2:
            return (a >= self.threshold).astype(np.int64)
        return np.rint(a).astype(np.int64)

    def update(self, y_pred: ArrayLike, y_true: ArrayLike) -> None:
        t = self._true_labels(y_true)
        p = self._pred_labels(y_pred)
        if t.shape != p.shape:
            raise ValueError("y_pred and y_true describe a different number of samples")
        np.add.at(self.matrix, (t, p), 1)
        self._count += t.size

    def compute(self) -> np.ndarray:
        if self._count == 0:
            raise RuntimeError("compute called before update")
        return self.matrix.copy()

    def stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._count == 0:
            raise RuntimeError("compute called before update")
        m       = self.matrix.astype(float)
        tp      = np.diag(m).copy()
        fp      = m.sum(axis=0) - tp
        fn      = m.sum(axis=1) - tp
        support = m.sum(axis=1)
        return tp, fp, fn, support


class _ConfusionDerived(Metric):
    def __init__(
        self,
        num_classes: int                     = 2,
        average    : Optional[str]            = "binary",
        threshold  : float                    = 0.5,
        confusion  : Optional[ConfusionMatrix] = None,
    ):
        valid = ("binary", "macro", "micro", "weighted", None)
        if average not in valid:
            raise ValueError(f"average must be one of {valid}")
        self.confusion = confusion if confusion is not None \
                         else ConfusionMatrix(num_classes, threshold)
        if average == "binary" and self.confusion.num_classes != 2:
            raise ValueError("average='binary' requires a 2-class confusion matrix")
        self.average = average

    def update(self, y_pred: ArrayLike, y_true: ArrayLike) -> None:
        self.confusion.update(y_pred, y_true)

    def reset(self) -> None:
        self.confusion.reset()

    @abstractmethod
    def _per_class(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def _micro(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
        ...

    def compute(self) -> float | np.ndarray:
        tp, fp, fn, support = self.confusion.stats()
        if self.average == "micro":
            return self._micro(tp, fp, fn)
        values = self._per_class(tp, fp, fn)
        if self.average is None:
            return values
        if self.average == "binary":
            return float(values[1])
        if self.average == "macro":
            return float(np.mean(values))
        total = support.sum()
        return float(np.sum(values * support) / total) if total > 0 else 0.0


class Precision(_ConfusionDerived):
    def _per_class(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        return _safe_divide(tp, tp + fp)

    def _micro(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
        denominator = tp.sum() + fp.sum()
        return float(tp.sum() / denominator) if denominator > 0 else 0.0


class Recall(_ConfusionDerived):
    def _per_class(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        return _safe_divide(tp, tp + fn)

    def _micro(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
        denominator = tp.sum() + fn.sum()
        return float(tp.sum() / denominator) if denominator > 0 else 0.0


class F1Score(_ConfusionDerived):
    def _per_class(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        return _safe_divide(2 * tp, 2 * tp + fp + fn)

    def _micro(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
        denominator = 2 * tp.sum() + fp.sum() + fn.sum()
        return float(2 * tp.sum() / denominator) if denominator > 0 else 0.0


class Accuracy(Metric):
    def __init__(
        self,
        num_classes: int                      = 2,
        threshold  : float                     = 0.5,
        confusion  : Optional[ConfusionMatrix] = None,
    ):
        self.confusion = confusion if confusion is not None \
                         else ConfusionMatrix(num_classes, threshold)

    def update(self, y_pred: ArrayLike, y_true: ArrayLike) -> None:
        self.confusion.update(y_pred, y_true)

    def reset(self) -> None:
        self.confusion.reset()

    def compute(self) -> float:
        matrix = self.confusion.compute()
        total  = matrix.sum()
        return float(np.trace(matrix) / total) if total > 0 else 0.0


class _RankingMetric(Metric):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._scores : list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, y_pred: ArrayLike, y_true: ArrayLike) -> None:
        self._scores.append(np.asarray(y_pred, dtype=float).reshape(-1))
        self._targets.append(np.rint(np.asarray(y_true).reshape(-1).astype(float)).astype(np.int64))

    def _pooled(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._scores:
            raise RuntimeError("compute called before update")
        return np.concatenate(self._scores), np.concatenate(self._targets)

    def _thresholded_counts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, targets = self._pooled()
        order    = np.argsort(-scores, kind="mergesort")
        scores   = scores[order]
        targets  = targets[order]
        distinct = np.where(np.diff(scores))[0]
        index    = np.r_[distinct, scores.size - 1]
        tp       = np.cumsum(targets)[index].astype(float)
        fp       = 1 + index - tp
        return tp, fp, scores[index]

    def _roc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tp, fp, thresholds = self._thresholded_counts()
        tp          = np.r_[0.0, tp]
        fp          = np.r_[0.0, fp]
        thresholds  = np.r_[thresholds[0] + 1.0, thresholds]
        total_pos   = tp[-1]
        total_neg   = fp[-1]
        tpr = tp / total_pos if total_pos > 0 else np.zeros_like(tp)
        fpr = fp / total_neg if total_neg > 0 else np.zeros_like(fp)
        return fpr, tpr, thresholds

    def _pr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tp, fp, thresholds = self._thresholded_counts()
        precision = _safe_divide(tp, tp + fp)
        total_pos = tp[-1]
        recall    = tp / total_pos if total_pos > 0 else np.zeros_like(tp)
        precision = np.r_[precision[::-1], 1.0]
        recall    = np.r_[recall[::-1], 0.0]
        return precision, recall, thresholds[::-1]


class RocCurve(_RankingMetric):
    def compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._roc()


class AUROC(_RankingMetric):
    def compute(self) -> float:
        fpr, tpr, _ = self._roc()
        return float(np.trapezoid(tpr, fpr))


class PrecisionRecallCurve(_RankingMetric):
    def compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._pr()


class AveragePrecision(_RankingMetric):
    def compute(self) -> float:
        precision, recall, _ = self._pr()
        ascending_recall     = recall[::-1]
        ascending_precision  = precision[::-1]
        return float(np.sum(np.diff(ascending_recall) * ascending_precision[1:]))


class _RegressionMetric(Metric):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._n              = 0
        self._squared_error  = 0.0
        self._absolute_error = 0.0
        self._sum_true       = 0.0
        self._sum_true_sq    = 0.0

    def update(self, y_pred: ArrayLike, y_true: ArrayLike) -> None:
        pred  = np.asarray(y_pred, dtype=float).reshape(-1)
        true  = np.asarray(y_true, dtype=float).reshape(-1)
        error = pred - true
        self._n              += true.size
        self._squared_error  += float(np.sum(error ** 2))
        self._absolute_error += float(np.sum(np.abs(error)))
        self._sum_true       += float(np.sum(true))
        self._sum_true_sq    += float(np.sum(true ** 2))

    def _guard(self) -> None:
        if self._n == 0:
            raise RuntimeError("compute called before update")


class MeanSquaredError(_RegressionMetric):
    def compute(self) -> float:
        self._guard()
        return self._squared_error / self._n


class RootMeanSquaredError(_RegressionMetric):
    def compute(self) -> float:
        self._guard()
        return float(np.sqrt(self._squared_error / self._n))


class MeanAbsoluteError(_RegressionMetric):
    def compute(self) -> float:
        self._guard()
        return self._absolute_error / self._n


class R2Score(_RegressionMetric):
    def compute(self) -> float:
        self._guard()
        total_sum_squares = self._sum_true_sq - self._sum_true ** 2 / self._n
        return 1.0 - self._squared_error / total_sum_squares if total_sum_squares > 0 else 0.0
