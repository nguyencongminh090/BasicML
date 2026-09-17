/**
 * @file train_cnn_mnist.cpp
 * @brief Efficient CNN on real MNIST, the AdvanceML counterpart to
 * `BasicML/examples/train_cnn_mnist.py`.
 *
 * Same architecture and training config as the BasicML script -- `(Conv2D ->
 * BatchNorm2D -> ReLU -> MaxPool2D) x2 -> Conv2D -> BatchNorm2D -> ReLU ->
 * GlobalAvgPool2D -> Flatten -> Linear`, trained with a fused softmax +
 * cross-entropy loss on the logits, `AdamW`, a genuine
 * disjoint train/test split of real MNIST -- built on TODO-0038's CNN
 * layers and TODO-0039's MNIST loader/accuracy metric, closing out area 6
 * of the TODO-0034 umbrella. Run `AdvanceML/scripts/prepare_mnist_idx.py`
 * once first to produce the IDX files this reads.
 */
#include "advanceml/datasets/mnist.hpp"
#include "advanceml/metrics/accuracy.hpp"
#include "advanceml/nn/batch_norm2d.hpp"
#include "advanceml/nn/conv2d.hpp"
#include "advanceml/nn/flatten.hpp"
#include "advanceml/nn/global_avg_pool2d.hpp"
#include "advanceml/nn/linear.hpp"
#include "advanceml/nn/loss.hpp"
#include "advanceml/nn/max_pool2d.hpp"
#include "advanceml/nn/relu.hpp"
#include "advanceml/nn/sequential.hpp"
#include "advanceml/optim/adamw.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace advanceml;

namespace {

constexpr size_t kImageSize = 28;
constexpr size_t kConv1Channels = 16;
constexpr size_t kConv2Channels = 32;
constexpr size_t kConv3Channels = 64;
constexpr size_t kEpochs = 10;
constexpr size_t kBatchSize = 128;
constexpr float kLearningRate = 0.001f;
constexpr float kWeightDecay = 0.01f;
constexpr unsigned kSeed = 0;

struct Model {
    std::shared_ptr<Sequential> net;
    std::shared_ptr<BatchNorm2D> bn1, bn2, bn3;
};

// Same layer order as BasicML's build_model(): kernel_size=3/stride=1/padding=1 convs, so
// only the two MaxPool2D(2) layers shrink the spatial size (28 -> 14 -> 7) before
// GlobalAvgPool2D collapses each channel's 7x7 map. BasicML ends in a Softmax layer feeding
// CrossEntropyLoss; here the model outputs logits for the fused SoftmaxCrossEntropyLoss, which
// is numerically stabler and one pass cheaper (argmax, and so accuracy, is unchanged).
Model build_model() {
    auto conv1 = std::make_shared<Conv2D>(1, kConv1Channels, 3, 1, 1, kSeed + 1);
    auto bn1 = std::make_shared<BatchNorm2D>(kConv1Channels);
    auto conv2 = std::make_shared<Conv2D>(kConv1Channels, kConv2Channels, 3, 1, 1, kSeed + 2);
    auto bn2 = std::make_shared<BatchNorm2D>(kConv2Channels);
    auto conv3 = std::make_shared<Conv2D>(kConv2Channels, kConv3Channels, 3, 1, 1, kSeed + 3);
    auto bn3 = std::make_shared<BatchNorm2D>(kConv3Channels);
    auto linear = std::make_shared<Linear>(kConv3Channels, 10, kSeed + 4);

    auto net = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        conv1, bn1, std::make_shared<ReLU>(), std::make_shared<MaxPool2D>(2, 2),
        conv2, bn2, std::make_shared<ReLU>(), std::make_shared<MaxPool2D>(2, 2),
        conv3, bn3, std::make_shared<ReLU>(), std::make_shared<GlobalAvgPool2D>(),
        std::make_shared<Flatten>(), linear,
    });
    return Model{net, bn1, bn2, bn3};
}

void set_training(Model& model, bool training) {
    for (auto* bn : {model.bn1.get(), model.bn2.get(), model.bn3.get()}) {
        if (training) {
            bn->train();
        } else {
            bn->eval();
        }
    }
}

// Copies rows `idx[begin:end]` of a row-major `(N, row_size)` buffer into one contiguous
// `(end - begin, row_size)` batch buffer, since Tensor has no gather/index-select op yet.
Tensor gather_rows(const FloatBuffer& data, size_t row_size, const std::vector<size_t>& idx,
                    size_t begin, size_t end, std::vector<size_t> batch_shape) {
    FloatBuffer batch(row_size * (end - begin));
    for (size_t i = begin; i < end; ++i) {
        std::copy_n(data.begin() + static_cast<long>(idx[i] * row_size), row_size,
                    batch.begin() + static_cast<long>((i - begin) * row_size));
    }
    batch_shape.insert(batch_shape.begin(), end - begin);
    return Tensor(std::move(batch), std::move(batch_shape));
}

struct EpochLog {
    float train_loss, train_acc, test_loss, test_acc, seconds;
};

std::pair<float, float> evaluate(Model& model, SoftmaxCrossEntropyLoss& criterion,
                                  const datasets::MnistDataset& data, const Tensor& targets) {
    set_training(model, false);
    // Evaluation never calls backward(); without the guard every batch would still build a full
    // graph and keep its saved activations and BatchNorm xhat buffers alive.
    NoGradGuard no_grad;
    metrics::Accuracy accuracy;
    float total_loss = 0.0f;
    size_t n_batches = 0;
    const size_t n = data.labels.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    for (size_t start = 0; start < n; start += kBatchSize) {
        size_t end = std::min(start + kBatchSize, n);
        Tensor xb = gather_rows(data.images.data(), kImageSize * kImageSize, idx, start, end,
                                 {1, kImageSize, kImageSize});
        Tensor yb = gather_rows(targets.data(), 10, idx, start, end, {10});
        Tensor logits = model.net->forward(xb);
        total_loss += criterion(logits, yb).data()[0];
        accuracy.update(logits, yb);
        ++n_batches;
    }
    set_training(model, true);
    return {total_loss / static_cast<float>(n_batches), accuracy.compute()};
}

std::vector<EpochLog> train(Model& model, const datasets::MnistDataset& train_data,
                             const Tensor& train_targets, const datasets::MnistDataset& test_data,
                             const Tensor& test_targets) {
    SoftmaxCrossEntropyLoss criterion;
    AdamW optimizer(model.net->parameters(), kLearningRate, 0.9f, 0.999f, 1e-8f, kWeightDecay);
    const size_t n = train_data.labels.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(kSeed);

    std::vector<EpochLog> logs;
    for (size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        auto epoch_start = std::chrono::steady_clock::now();
        std::shuffle(idx.begin(), idx.end(), rng);

        metrics::Accuracy train_accuracy;
        float train_loss_total = 0.0f;
        size_t n_batches = 0;
        for (size_t start = 0; start < n; start += kBatchSize) {
            size_t end = std::min(start + kBatchSize, n);
            Tensor xb = gather_rows(train_data.images.data(), kImageSize * kImageSize, idx, start,
                                     end, {1, kImageSize, kImageSize});
            Tensor yb = gather_rows(train_targets.data(), 10, idx, start, end, {10});

            Tensor logits = model.net->forward(xb);
            Tensor loss = criterion(logits, yb);
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();

            train_accuracy.update(logits, yb);
            train_loss_total += loss.data()[0];
            ++n_batches;
        }
        float train_loss = train_loss_total / static_cast<float>(n_batches);
        float train_acc = train_accuracy.compute();
        auto [test_loss, test_acc] = evaluate(model, criterion, test_data, test_targets);
        float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - epoch_start).count();
        logs.push_back({train_loss, train_acc, test_loss, test_acc, seconds});
        std::printf("epoch %2zu/%2zu  train_loss=%.4f  train_acc=%.4f  test_loss=%.4f  test_acc=%.4f  (%.2fs)\n",
                    epoch, kEpochs, train_loss, train_acc, test_loss, test_acc, seconds);
    }
    return logs;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = argc > 1 ? argv[1] : "AdvanceML/data/mnist";
    auto train_data = datasets::load_mnist(data_dir + "/train-images-idx3-ubyte",
                                            data_dir + "/train-labels-idx1-ubyte");
    auto test_data = datasets::load_mnist(data_dir + "/test-images-idx3-ubyte",
                                           data_dir + "/test-labels-idx1-ubyte");
    Tensor train_targets = train_data.one_hot_labels(10);
    Tensor test_targets = test_data.one_hot_labels(10);

    Model model = build_model();
    auto total_start = std::chrono::steady_clock::now();
    auto logs = train(model, train_data, train_targets, test_data, test_targets);
    float total_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - total_start).count();

    std::printf("\nFinal held-out test accuracy: %.4f on %zu unseen images (trained on %zu)\n",
                logs.back().test_acc, test_data.labels.size(), train_data.labels.size());
    std::printf("Total training wall time: %.2fs (%zu epochs)\n", total_seconds, kEpochs);
    return 0;
}
