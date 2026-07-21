# Deep Learning from Scratch — Hành Trình Tự Học

> **Repo này tổng hợp quá trình tự học Deep Learning của tôi.**  
> Tôi xây dựng mọi thứ **from scratch** — từ Machine Learning cơ bản đến Deep Learning — cùng với tài liệu tự viết trình bày kiến thức một cách cô đọng nhất.

---

## Mục tiêu

Dự án này không nhằm tái tạo một thư viện production. Mục tiêu cốt lõi là **hiểu sâu từng khái niệm** bằng cách tự cài đặt lại từ đầu:

- Nắm vững toán học đằng sau từng thuật toán
- Hiểu luồng forward pass → loss → backward pass → optimizer step
- Xây dựng nền tảng vững chắc trước khi dùng PyTorch / JAX / TensorFlow

---

## Cấu trúc dự án

```
MachineLearning/
└── BasicML/
    ├── basicml/               # Thư viện tự xây dựng
    │   ├── tensor.py          # Tensor wrapper (NumPy-backed)
    │   ├── nn/
    │   │   ├── module.py      # Abstract base class cho mọi model
    │   │   ├── linear.py      # Linear Regression (forward + backward)
    │   │   └── loss.py        # Hàm mất mát (MSELoss, ...)
    │   └── optim/
    │       ├── optimizer.py   # Abstract base class cho optimizer
    │       └── sgd.py         # Stochastic Gradient Descent
    ├── examples/
    │   └── train_linear.py    # Ví dụ huấn luyện mô hình Linear
    ├── tepmlate_1(basic).py   # Template PyTorch-style (cơ bản)
    └── template_2_pytorch.py  # Template PyTorch-style (nâng cao)
```

---

## Kiến trúc thư viện `basicml`

Thư viện được thiết kế theo mô hình **PyTorch-inspired**, tách biệt rõ ràng các thành phần:

### `Tensor`
Wrapper nhẹ trên `numpy.ndarray`, hỗ trợ:
- Lưu trữ `data` và `grad`
- Cờ `requires_grad` để theo dõi gradient
- Các phép toán cơ bản: `+`, `-`, `*`, `/`, `@` (matmul)

### `nn.Module`
Abstract base class cho mọi model. Mọi model phải implement:
- `forward(X)` — tính đầu ra
- `parameters()` — trả về danh sách `Tensor` cần tối ưu
- `backward(grad_output)` — tính gradient thủ công

### `nn.LinearRegression`
Triển khai **Linear Regression** hoàn chỉnh với:
- Khởi tạo trọng số `w` và bias `b` bằng 0
- `forward`: $\hat{y} = X \cdot w + b$
- `backward`: Tính gradient theo quy tắc chain rule

### `nn.Loss` & `MSELoss`
- **MSELoss**: $\mathcal{L} = \frac{1}{2m}\sum(\hat{y} - y)^2$
- `backward()`: Trả về gradient $\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y$

### `optim.SGD`
**Stochastic Gradient Descent** chuẩn:
- `step()`: Cập nhật tham số theo $\theta \leftarrow \theta - \alpha \cdot \nabla\theta$
- `zero_grad()`: Đặt lại gradient về 0 trước mỗi iteration

---

## Ví dụ sử dụng

```python
import numpy as np
from basicml.nn.linear import LinearRegression
from basicml.nn.loss   import MSELoss
from basicml.optim.sgd import SGD

# Dữ liệu giả
X = np.random.randn(100, 3)
y = np.random.randn(100, 1)

# Khởi tạo model, loss, optimizer
model     = LinearRegression(features=3)
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Vòng lặp huấn luyện
for epoch in range(100):
    # Forward pass
    y_pred = model(X)
    loss   = criterion(y_pred, y)

    # Backward pass
    grad = criterion.backward()
    model.backward(grad)

    # Cập nhật tham số
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")
```

---

## Lộ trình học tập

| Giai đoạn | Nội dung | Trạng thái |
|-----------|----------|------------|
| **Machine Learning** | Linear Regression, Gradient Descent | Đang xây dựng |
| **Machine Learning** | Logistic Regression, Classification | Sắp tới |
| **Machine Learning** | Decision Tree, SVM, KNN | Sắp tới |
| **Deep Learning** | MLP, Backpropagation tự động | Sắp tới |
| **Deep Learning** | CNN, RNN, Attention | Sắp tới |
| **Deep Learning** | Transformer from scratch | Sắp tới |

---

## Yêu cầu

```
Python >= 3.10
numpy
```

Cài đặt:
```bash
pip install numpy
```

---

## Ghi chú

- Tất cả thuật toán được **cài đặt thuần NumPy** — không dùng PyTorch hay framework tương đương — trừ khi có ghi chú riêng.
- Mỗi module đi kèm **tài liệu tự viết** giải thích toán học và ý tưởng đằng sau thuật toán.
- Mã nguồn ưu tiên **sự rõ ràng** hơn hiệu năng để dễ hiểu và dễ học.

---

<p align="center">
  <i>Học bằng cách tự xây dựng — Build to understand.</i>
</p>
