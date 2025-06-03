# optimgrad: A lightweight and mathematical autograd engine

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg)](https://numpy.org)

</div>

OptimGrad is a lightweight, educational automatic differentiation engine built in pure Python with NumPy. It provides a PyTorch-like API for building and training neural networks, with a focus on clarity and learning. The engine implements reverse-mode automatic differentiation (backpropagation) and includes various utilities for deep learning.

## Features

- **Automatic Differentiation**: Reverse-mode autodiff with dynamic computational graphs
- **Tensor Operations**: Broadcasting, element-wise operations, and matrix operations
- **Activation Functions**: ReLU, Sigmoid, Tanh with proper gradient computation
- **Loss Functions**: MSE and Binary Cross-Entropy with numerical stability
- **Optimizers**: SGD and Adam with configurable hyperparameters
- **Utilities**: 
  - Gradient clipping
  - Gradient flow monitoring
  - Bound estimation
  - Computational graph visualization
  - Chain rule walkthrough
  - Gradient path explanation

## Installation

```bash
git clone https://github.com/yourusername/optimgrad.git
cd optimgrad
pip install -r requirements.txt
```

## Quick Start

```python
from engine import Tensor
from loss import mse_loss

# Create tensors with gradients
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)

# Forward pass
z = x * y + x**2  # Computational graph is built automatically

# Backward pass
z.backward()

# Access gradients
print(x.grad)  # dy/dx = y + 2x = 3 + 4 = 7
print(y.grad)  # dy/dy = x = 2
```

## Core Components

### 1. Tensor Class

The fundamental building block with support for:
- Automatic gradient computation
- Broadcasting operations
- In-place operation detection
- Computational graph tracking

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward_fn = lambda: None
        self.children = ()
        self._in_graph = False
```

### 2. Activation Functions

Built-in activation functions with their derivatives:

#### ReLU
```math
f(x) = \max(0, x)
```
```math
f'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
0 & \text{otherwise}
\end{cases}
```

#### Sigmoid
```math
f(x) = \frac{1}{1 + e^{-x}}
```
```math
f'(x) = f(x)(1 - f(x))
```

#### Tanh
```math
f(x) = \tanh(x)
```
```math
f'(x) = 1 - \tanh^2(x)
```

### 3. Loss Functions

#### Mean Squared Error (MSE)
```math
L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
```

#### Binary Cross-Entropy (BCE)
```math
L = -\frac{1}{n}\sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
```

### 4. Optimizers

#### SGD (Stochastic Gradient Descent)
```python
w = w - lr * w.grad
```

#### Adam
```math
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
```
```math
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
```
```math
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
```
```math
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
```
```math
w = w - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t
```

### 5. Utilities

#### Gradient Clipping
```python
# Clip gradients to have a maximum norm
clip_gradients(params, max_norm=1.0)
```

#### Bound Estimation
```python
# Estimate output bounds of a function
lower, upper = estimate_bounds(f, x, eps=1e-4)
```

## Testing

The project includes comprehensive tests for all components:
```bash
python test.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by PyTorch's autograd system
- Built with educational purposes in mind
- Special thanks to all contributors

## Citation

If you use OptimGrad in your research, please cite:

```bibtex
@software{optimgrad2025,
  author = {Your Name},
  title = {optimgrad: A lightweight and mathematical autograd engine},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/optimgrad}
}
``` 