#include "advanceml/metrics/accuracy.hpp"

#include <stdexcept>

namespace advanceml::metrics {

namespace {

size_t argmax_row(const std::vector<float>& data, size_t row, size_t num_cols) {
    size_t best = 0;
    float best_value = data[row * num_cols];
    for (size_t c = 1; c < num_cols; ++c) {
        const float value = data[row * num_cols + c];
        if (value > best_value) {
            best_value = value;
            best = c;
        }
    }
    return best;
}

}  // namespace

void Accuracy::update(const Tensor& pred, const Tensor& target) {
    if (pred.shape().size() != 2 || target.shape().size() != 2) {
        throw std::runtime_error("Accuracy::update: pred and target must be 2D (N, num_classes)");
    }
    if (pred.shape() != target.shape()) {
        throw std::runtime_error("Accuracy::update: pred and target shapes differ");
    }

    const size_t num_rows = pred.shape()[0];
    const size_t num_cols = pred.shape()[1];
    const std::vector<float>& pred_data = pred.data();
    const std::vector<float>& target_data = target.data();
    for (size_t r = 0; r < num_rows; ++r) {
        if (argmax_row(pred_data, r, num_cols) == argmax_row(target_data, r, num_cols)) {
            ++correct_;
        }
    }
    total_ += num_rows;
}

void Accuracy::update(const Tensor& pred, const std::vector<int>& labels) {
    if (pred.shape().size() != 2) {
        throw std::runtime_error("Accuracy::update: pred must be 2D (N, num_classes)");
    }
    const size_t num_rows = pred.shape()[0];
    if (num_rows != labels.size()) {
        throw std::runtime_error("Accuracy::update: pred and labels describe a different number of samples");
    }

    const size_t num_cols = pred.shape()[1];
    const std::vector<float>& pred_data = pred.data();
    for (size_t r = 0; r < num_rows; ++r) {
        if (static_cast<int>(argmax_row(pred_data, r, num_cols)) == labels[r]) {
            ++correct_;
        }
    }
    total_ += num_rows;
}

float Accuracy::compute() const {
    if (total_ == 0) {
        throw std::runtime_error("Accuracy::compute called before update");
    }
    return static_cast<float>(correct_) / static_cast<float>(total_);
}

void Accuracy::reset() noexcept {
    correct_ = 0;
    total_ = 0;
}

}  // namespace advanceml::metrics
