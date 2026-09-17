#include "advanceml/optim/muon.hpp"

#include <algorithm>
#include <cmath>

namespace advanceml {

namespace {

std::vector<float> small_matmul(const std::vector<float>& a, size_t m, size_t k, const std::vector<float>& b,
                                 size_t n) {
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            const float a_ip = a[i * k + p];
            for (size_t j = 0; j < n; ++j) {
                out[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    return out;
}

std::vector<float> small_transpose(const std::vector<float>& a, size_t rows, size_t cols) {
    std::vector<float> out(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    return out;
}

// Newton-Schulz iteration approximating the orthogonal polar factor of `g`
// (rows x cols, row-major); matches basicml.optim.muon.newton_schulz5.
std::vector<float> newton_schulz5(const std::vector<float>& g, size_t rows, size_t cols, int steps, float eps) {
    constexpr float a = 3.4445f;
    constexpr float b = -4.7750f;
    constexpr float c = 2.0315f;

    float norm = 0.0f;
    for (float v : g) {
        norm += v * v;
    }
    norm = std::sqrt(norm) + eps;

    std::vector<float> x(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
        x[i] = g[i] / norm;
    }

    const bool transposed = rows > cols;
    size_t r = rows;
    size_t k = cols;
    if (transposed) {
        x = small_transpose(x, rows, cols);
        r = cols;
        k = rows;
    }

    for (int step = 0; step < steps; ++step) {
        std::vector<float> x_t = small_transpose(x, r, k);
        std::vector<float> mat_a = small_matmul(x, r, k, x_t, r);  // (r, r)
        std::vector<float> a_sq = small_matmul(mat_a, r, r, mat_a, r);  // (r, r)
        std::vector<float> mat_b(r * r);
        for (size_t i = 0; i < mat_b.size(); ++i) {
            mat_b[i] = b * mat_a[i] + c * a_sq[i];
        }
        std::vector<float> bx = small_matmul(mat_b, r, r, x, k);  // (r, k)
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = a * x[i] + bx[i];
        }
    }

    if (transposed) {
        x = small_transpose(x, r, k);
    }
    return x;
}

}  // namespace

Muon::Muon(std::vector<Tensor> parameters, float learning_rate, float momentum, bool nesterov,
           int newton_schulz_steps, float eps)
    : parameters_(std::move(parameters)),
      learning_rate_(learning_rate),
      momentum_(momentum),
      nesterov_(nesterov),
      newton_schulz_steps_(newton_schulz_steps),
      eps_(eps) {
    velocities_.reserve(parameters_.size());
    for (const Tensor& param : parameters_) {
        velocities_.emplace_back(param.data().size(), 0.0f);
    }
}

void Muon::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        if (!param.has_grad()) {
            continue;
        }
        Tensor grad = param.grad();
        FloatBuffer& values = param.mutable_data();
        std::vector<float>& velocity = velocities_[i];
        for (size_t j = 0; j < velocity.size(); ++j) {
            velocity[j] = momentum_ * velocity[j] + grad.data()[j];
        }

        std::vector<float> update(velocity.size());
        for (size_t j = 0; j < update.size(); ++j) {
            update[j] = nesterov_ ? momentum_ * velocity[j] + grad.data()[j] : velocity[j];
        }

        if (param.shape().size() >= 2) {
            const size_t rows = param.shape()[0];
            const size_t cols = values.size() / rows;
            std::vector<float> ortho = newton_schulz5(update, rows, cols, newton_schulz_steps_, eps_);
            const float scale = std::sqrt(std::max(1.0f, static_cast<float>(rows) / static_cast<float>(cols)));
            for (size_t j = 0; j < values.size(); ++j) {
                values[j] -= learning_rate_ * scale * ortho[j];
            }
        } else {
            for (size_t j = 0; j < values.size(); ++j) {
                values[j] -= learning_rate_ * update[j];
            }
        }
    }
}

void Muon::zero_grad() {
    for (Tensor& param : parameters_) {
        param.zero_grad();
    }
}

}  // namespace advanceml
