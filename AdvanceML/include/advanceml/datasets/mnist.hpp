#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "advanceml/tensor.hpp"

namespace advanceml::datasets {

/**
 * One MNIST-format split: dense pixel data plus integer labels.
 *
 * `images` is `(N, rows * cols)` -- one flattened row per example,
 * pixel values scaled from the raw `[0, 255]` byte range to `[0, 1]`.
 * `labels[i]` is the digit (0-9) for row `i` of `images`.
 */
struct MnistDataset {
    Tensor images;
    std::vector<int> labels;
    size_t rows;
    size_t cols;

    /** Constructs from an already-loaded flattened `images` tensor and parallel `labels`. */
    MnistDataset(Tensor images, std::vector<int> labels, size_t rows, size_t cols);

    /**
     * One-hot encodes `labels` into an `(N, num_classes)` Tensor, matching
     * the target shape `CrossEntropyLoss` expects (a one-hot distribution
     * per row).
     *
     * @param num_classes Number of columns in the output (10 for digits).
     * @returns An `(N, num_classes)` Tensor; row `i` is all zeros except a
     *          1 at column `labels[i]`.
     * @throws std::runtime_error if any label is outside `[0, num_classes)`.
     */
    [[nodiscard]] Tensor one_hot_labels(size_t num_classes = 10) const;
};

/**
 * Loads an MNIST-format image/label pair from the IDX file format
 * (http://yann.lecun.com/exdb/mnist/) -- the ubyte files MNIST and
 * Fashion-MNIST are distributed as. Reads the big-endian binary layout
 * directly, with no Python dependency at runtime (unlike BasicML's
 * `sklearn.datasets.fetch_openml`-based fetcher).
 *
 * @param images_path Path to an IDX3 image file (magic number 2051).
 * @param labels_path Path to an IDX1 label file (magic number 2049).
 * @returns The loaded dataset, with pixel values normalized to `[0, 1]`.
 * @throws std::runtime_error if either file cannot be opened, has the
 *         wrong magic number, or the two files disagree on example count.
 */
MnistDataset load_mnist(const std::string& images_path, const std::string& labels_path);

}  // namespace advanceml::datasets
