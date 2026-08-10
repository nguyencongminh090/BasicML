# Deep Learning from Scratch

> Cài lại các thuật toán Machine Learning và Deep Learning **from scratch**, kèm ghi chú giải thích toán học đằng sau từng thuật toán.

---

## Mục tiêu

Không nhằm xây dựng một thư viện production — mọi thuật toán đều được cài lại từ đầu bằng NumPy thuần, kể cả backward pass, để hiểu rõ:

- Toán học đằng sau từng thuật toán
- Luồng forward pass → loss → backward pass → optimizer step

---

## Cấu trúc dự án

```
MachineLearning/
└── BasicML/
    ├── basicml/               # Thư viện tự xây dựng
    │   ├── tensor.py          # Tensor wrapper (NumPy-backed)
    │   ├── nn/
    │   │   ├── module.py      # Abstract base class
    │   │   ├── linear.py      # Linear Layer (Xavier & He Init)
    │   │   ├── activation.py  # Sigmoid, ReLU
    │   │   ├── models.py      # LogisticRegressionModel
    │   │   └── loss.py        # MSELoss, BinaryCrossEntropy
    │   └── optim/
    │       ├── optimizer.py   # Abstract base class cho optimizer
    │       ├── sgd.py         # Stochastic Gradient Descent
    │       └── momentum.py    # SGD với Momentum
    ├── examples/
    │   ├── train_linear.py    # Huấn luyện model Linear
    │   └── train_logistic.py  # Huấn luyện Logistic Regression
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

### `nn.Linear`
Triển khai **Linear Layer** hoàn chỉnh với:
- Khởi tạo trọng số `w` (hỗ trợ `xavier` hoặc `he` init) và bias `b` = 0
- `forward`: $y = X \cdot w + b$
- `backward`: Tính gradient theo quy tắc chain rule

### `nn.Activation`
Các activation function phi tuyến (kế thừa từ `Module`):
- **Sigmoid**: dùng cho binary classification
- **ReLU**: activation function phổ biến, $max(0, x)$

### `nn.models`
Composable models ghép từ các layer có sẵn:
- **LogisticRegressionModel**: Bao gồm `Linear` layer kết hợp cùng `Sigmoid` activation.

### `nn.Loss`
Loss function chịu trách nhiệm tính toán sai số và bắt đầu quá trình backward (bao gồm factor $1/m$):

**MSELoss**:

$$
\mathcal{L} = \frac{1}{2m}\sum(\hat{y} - y)^2
$$

**BinaryCrossEntropy**:

Loss function cho binary classification. Dùng `np.clip` chống `NaN`.

### `optim`
**SGD**: Gradient descent chuẩn — `step()` cập nhật tham số theo $\theta \leftarrow \theta - \alpha \cdot \nabla\theta$, `zero_grad()` đặt lại gradient về 0 trước mỗi iteration.

**Momentum**: SGD kèm vận tốc tích lũy (`velocities`, hệ số `momentum` mặc định 0.9) để làm mượt hướng cập nhật qua các bước.

---

## Ví dụ sử dụng

```python
import numpy as np
from basicml.nn.linear import Linear
from basicml.nn.loss   import MSELoss
from basicml.optim.sgd import SGD

# Dữ liệu giả
X = np.random.randn(100, 3)
y = np.random.randn(100, 1)

# Khởi tạo model, loss, optimizer
model     = Linear(features=3)
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

## TODO list

| Giai đoạn | Nội dung | Trạng thái |
|-----------|----------|------------|
| **Machine Learning** | Linear Regression, Gradient Descent | Hoàn thành |
| **Machine Learning** | Logistic Regression, Classification | Hoàn thành |
| **Deep Learning** | MLP, Backpropagation tự động | Sắp tới |
| **Deep Learning** | CNN, RNN, Attention | Sắp tới |
| **Deep Learning** | Transformer from scratch | Sắp tới |

---

## Yêu cầu

```
Python >= 3.13
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
