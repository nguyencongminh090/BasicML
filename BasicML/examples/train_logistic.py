import os
import sys
import random
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_data(n: int, a: float, b: float) -> List[Tuple[float, int]]:
    threshold = (a + b) / 4
    out = []
    for _ in range(n):
        x = random.uniform(a, b)
        out.append((x, 1 if x > threshold else 0))
    return out


data   = generate_data(100, 0, 10)
x_vals = np.array([[d[0] for d in data]]).reshape((-1, 1))
y_vals = np.array([d[1] for d in data]).reshape((-1, 1))

## Data Visulization
## -----------------
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ax1.hist([x for x, y in data if y == 0], bins=10, alpha=0.5, label='Class 0', color='blue')
# ax1.hist([x for x, y in data if y == 1], bins=10, alpha=0.5, label='Class 1', color='red')
# ax1.legend()
# ax1.set_title("Data Distribution")

# ax2.scatter(x_vals, y_vals, c=y_vals, cmap='bwr')
# ax2.set_title("Scatter Plot")

# plt.show()
# -------------------

from basicml.nn.models      import LogisticRegressionModel
from basicml.optim.momentum import Momentum
from basicml.nn.loss        import BinaryCrossEntropy, MSELoss

epochs    = 0
threshold = 1e-3

model = LogisticRegressionModel(1)
loss  = BinaryCrossEntropy()
opti  = Momentum(model.parameters(), lr=0.07, momentum=0.95)

while True:
    y_pred = model(x_vals)
    l      = loss(y_pred, y_vals)
    grad_l = loss.backward()
    model.backward(grad_l)
    opti.step()
    opti.zero_grad()
    epochs += 1
    if epochs % 100000 == 0:
        print(f'[Epochs] : {epochs:6d} | [Loss] : {l:.6f}')
    if l < threshold:
        break

print(f'w = {model.parameters()[0].data.item():.4f}, '
      f'b = {model.parameters()[1].data.item():.4f} |'
      f'f(x) = 1 / (1 + e^-({model.parameters()[0].data.item():.4f}x +'
      f'({model.parameters()[1].data.item():.4f})))')
print(f'Loss: {l}')

x_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = model(x_line)
y_pred_vals = model(x_vals)

plt.scatter(x_vals, y_vals, c=y_vals, cmap='bwr')
plt.plot(x_line, y_line, color='black')
plt.vlines(x_vals, y_vals, y_pred_vals, color='gray', linestyle='dashed', alpha=0.5)
plt.show()
