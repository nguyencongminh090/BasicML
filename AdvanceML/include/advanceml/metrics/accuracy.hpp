#pragma once

#include <cstddef>
#include <vector>

#include "advanceml/tensor.hpp"

namespace advanceml::metrics {

/**
 * Accumulates classification accuracy across one or more batches,
 * mirroring `basicml.metrics.Accuracy`'s update/compute/reset protocol
 * (`Metric` in BasicML) so a training loop can report accuracy the same
 * way whether it's running on BasicML or AdvanceML.
 *
 * Accuracy is `correct / total` over every row seen since construction
 * or the last `reset()`, where a row is "correct" if `pred`'s row-wise
 * argmax matches the row-wise argmax of a one-hot/probability `target`
 * (or a raw integer label, via the other `update()` overload).
 */
class Accuracy {
public:
    /**
     * Compares each row's argmax between `pred` and `target` and
     * accumulates the number of matches.
     *
     * @param pred `(N, num_classes)` logits or a probability
     *        distribution (argmax is invariant to any monotonic
     *        transform, so raw logits and post-Softmax output agree).
     * @param target `(N, num_classes)` one-hot or probability
     *        distribution, matching `CrossEntropyLoss`'s target
     *        convention.
     * @throws std::runtime_error if `pred`/`target` aren't 2D or don't
     *         share a shape.
     */
    void update(const Tensor& pred, const Tensor& target);

    /**
     * Compares each row's argmax in `pred` against the corresponding
     * raw integer label and accumulates the number of matches.
     *
     * @param pred `(N, num_classes)` logits or a probability distribution.
     * @param labels One class index per row of `pred`.
     * @throws std::runtime_error if `pred` isn't 2D or its row count
     *         doesn't match `labels.size()`.
     */
    void update(const Tensor& pred, const std::vector<int>& labels);

    /**
     * @returns The fraction of correct predictions across every
     *          `update()` call since construction or the last `reset()`.
     * @throws std::runtime_error if `update()` was never called.
     */
    [[nodiscard]] float compute() const;

    /** Resets the accumulated correct/total counts to zero. */
    void reset() noexcept;

private:
    size_t correct_ = 0;
    size_t total_ = 0;
};

}  // namespace advanceml::metrics
