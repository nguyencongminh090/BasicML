#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "advanceml/datasets/mnist.hpp"

using namespace advanceml;
using namespace advanceml::datasets;

namespace {

void write_be_uint32(std::ofstream& out, uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>((value >> 24) & 0xFF),
        static_cast<unsigned char>((value >> 16) & 0xFF),
        static_cast<unsigned char>((value >> 8) & 0xFF),
        static_cast<unsigned char>(value & 0xFF),
    };
    out.write(reinterpret_cast<const char*>(bytes), 4);
}

// Writes a synthetic 3-example, 2x2-pixel IDX image/label pair to a temp
// directory so load_mnist() can be tested without a real MNIST download.
struct SyntheticMnistFiles {
    std::filesystem::path images_path;
    std::filesystem::path labels_path;

    SyntheticMnistFiles() {
        const auto dir = std::filesystem::temp_directory_path();
        images_path = dir / "advanceml_test_mnist_images.idx";
        labels_path = dir / "advanceml_test_mnist_labels.idx";

        std::ofstream images(images_path, std::ios::binary);
        write_be_uint32(images, 2051);
        write_be_uint32(images, 3);
        write_be_uint32(images, 2);
        write_be_uint32(images, 2);
        const std::vector<unsigned char> pixels = {
            0,   0,   0,   0,    // example 0: all black
            255, 255, 255, 255,  // example 1: all white
            0,   255, 0,   255,  // example 2: checkerboard
        };
        images.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));

        std::ofstream labels(labels_path, std::ios::binary);
        write_be_uint32(labels, 2049);
        write_be_uint32(labels, 3);
        const std::vector<unsigned char> label_bytes = {0, 1, 2};
        labels.write(reinterpret_cast<const char*>(label_bytes.data()),
                     static_cast<std::streamsize>(label_bytes.size()));
    }

    ~SyntheticMnistFiles() {
        std::filesystem::remove(images_path);
        std::filesystem::remove(labels_path);
    }
};

}  // namespace

TEST_CASE("load_mnist parses IDX image/label pairs and normalizes pixels to [0, 1]", "[datasets]") {
    SyntheticMnistFiles files;

    MnistDataset dataset = load_mnist(files.images_path.string(), files.labels_path.string());

    REQUIRE(dataset.rows == 2);
    REQUIRE(dataset.cols == 2);
    REQUIRE(dataset.images.shape() == std::vector<size_t>{3, 4});
    REQUIRE(dataset.labels == std::vector<int>{0, 1, 2});

    const std::vector<float>& pixels = dataset.images.data();
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(pixels[i] == Catch::Approx(0.0f));
        REQUIRE(pixels[4 + i] == Catch::Approx(1.0f));
    }
    REQUIRE(pixels[8] == Catch::Approx(0.0f));
    REQUIRE(pixels[9] == Catch::Approx(1.0f));
}

TEST_CASE("MnistDataset::one_hot_labels encodes each label as a one-hot row", "[datasets]") {
    SyntheticMnistFiles files;
    MnistDataset dataset = load_mnist(files.images_path.string(), files.labels_path.string());

    Tensor one_hot = dataset.one_hot_labels(/*num_classes=*/3);

    REQUIRE(one_hot.shape() == std::vector<size_t>{3, 3});
    const std::vector<float> expected = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    REQUIRE(one_hot.data() == expected);
}

TEST_CASE("load_mnist rejects a file with the wrong IDX magic number", "[datasets]") {
    SyntheticMnistFiles files;

    // Labels file passed where an images file is expected: magic number
    // (2049) doesn't match the images magic number (2051).
    REQUIRE_THROWS_AS(load_mnist(files.labels_path.string(), files.labels_path.string()), std::runtime_error);
}
