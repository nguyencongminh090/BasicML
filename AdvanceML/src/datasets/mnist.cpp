#include "advanceml/datasets/mnist.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace advanceml::datasets {

namespace {

constexpr uint32_t kImageMagic = 2051;
constexpr uint32_t kLabelMagic = 2049;

// IDX headers are big-endian regardless of host byte order, so this reads
// the four bytes explicitly rather than relying on a platform byteswap.
uint32_t read_be_uint32(std::istream& in, const std::string& path) {
    unsigned char bytes[4];
    in.read(reinterpret_cast<char*>(bytes), 4);
    if (!in) {
        throw std::runtime_error("failed to read IDX header from " + path);
    }
    return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
}

}  // namespace

MnistDataset::MnistDataset(Tensor images, std::vector<int> labels, size_t rows, size_t cols)
    : images(std::move(images)), labels(std::move(labels)), rows(rows), cols(cols) {}

MnistDataset load_mnist(const std::string& images_path, const std::string& labels_path) {
    std::ifstream images_in(images_path, std::ios::binary);
    if (!images_in) {
        throw std::runtime_error("cannot open MNIST images file: " + images_path);
    }
    std::ifstream labels_in(labels_path, std::ios::binary);
    if (!labels_in) {
        throw std::runtime_error("cannot open MNIST labels file: " + labels_path);
    }

    if (read_be_uint32(images_in, images_path) != kImageMagic) {
        throw std::runtime_error("unexpected IDX magic number in " + images_path);
    }
    const uint32_t num_images = read_be_uint32(images_in, images_path);
    const uint32_t rows = read_be_uint32(images_in, images_path);
    const uint32_t cols = read_be_uint32(images_in, images_path);

    if (read_be_uint32(labels_in, labels_path) != kLabelMagic) {
        throw std::runtime_error("unexpected IDX magic number in " + labels_path);
    }
    const uint32_t num_labels = read_be_uint32(labels_in, labels_path);

    if (num_images != num_labels) {
        throw std::runtime_error("MNIST images/labels example count mismatch: " + std::to_string(num_images) +
                                  " vs " + std::to_string(num_labels));
    }

    const size_t image_size = static_cast<size_t>(rows) * cols;
    std::vector<unsigned char> pixel_bytes(static_cast<size_t>(num_images) * image_size);
    images_in.read(reinterpret_cast<char*>(pixel_bytes.data()), static_cast<std::streamsize>(pixel_bytes.size()));
    if (!images_in) {
        throw std::runtime_error("truncated MNIST images file: " + images_path);
    }

    std::vector<unsigned char> label_bytes(num_labels);
    labels_in.read(reinterpret_cast<char*>(label_bytes.data()), static_cast<std::streamsize>(label_bytes.size()));
    if (!labels_in) {
        throw std::runtime_error("truncated MNIST labels file: " + labels_path);
    }

    FloatBuffer pixels(pixel_bytes.size());
    for (size_t i = 0; i < pixel_bytes.size(); ++i) {
        pixels[i] = static_cast<float>(pixel_bytes[i]) / 255.0f;
    }

    return MnistDataset(Tensor(std::move(pixels), {static_cast<size_t>(num_images), image_size}),
                         std::vector<int>(label_bytes.begin(), label_bytes.end()), rows, cols);
}

Tensor MnistDataset::one_hot_labels(size_t num_classes) const {
    FloatBuffer data(labels.size() * num_classes, 0.0f);
    for (size_t i = 0; i < labels.size(); ++i) {
        const int label = labels[i];
        if (label < 0 || static_cast<size_t>(label) >= num_classes) {
            throw std::runtime_error("label " + std::to_string(label) + " out of range for one_hot_labels");
        }
        data[i * num_classes + static_cast<size_t>(label)] = 1.0f;
    }
    return Tensor(std::move(data), {labels.size(), num_classes});
}

}  // namespace advanceml::datasets
