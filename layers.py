import numpy as np
from typing import Tuple, Optional
from engine import Tensor

def xavier_init(shape: Tuple[int, ...]) -> np.ndarray: # for tanh and sigmoid functions 
    """Xavier/Glorot initialization"""
    n_in, n_out = shape[0], shape[-1]
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, shape)

def he_init(shape: Tuple[int, ...]) -> np.ndarray: # for relu functions 
    """He/Kaiming initialization"""
    n_in = shape[0]
    std = np.sqrt(2 / n_in)
    return np.random.normal(0, std, shape)

class Module:
    """Base class for all neural network modules"""
    def __init__(self):
        self.training = True
        self._parameters = []
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self) -> list:
        return self._parameters
        
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, init_type='default'):
        """Initialize a linear layer with specified initialization method.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            init_type (str): Initialization type ('default', 'xavier', 'he')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights based on the specified method
        if init_type == 'xavier':
            # Xavier/Glorot initialization
            bound = np.sqrt(6.0 / (in_features + out_features))
            self.weight = Tensor(np.random.uniform(-bound, bound, (in_features, out_features)), requires_grad=True)
        elif init_type == 'he':
            # He initialization
            std = np.sqrt(2.0 / in_features)
            self.weight = Tensor(np.random.normal(0, std, (in_features, out_features)), requires_grad=True)
        else:
            # Default initialization (uniform [-1/sqrt(n), 1/sqrt(n)])
            bound = 1 / np.sqrt(in_features)
            self.weight = Tensor(np.random.uniform(-bound, bound, (in_features, out_features)), requires_grad=True)
        
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self._parameters.extend([self.weight, self.bias])
        
    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight + self.bias
        return out

class Conv1D(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weight = Tensor(
            he_init((out_channels, in_channels, kernel_size)),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_channels),
            requires_grad=True
        )
        self._parameters.extend([self.weight, self.bias])
        
    def _extract_windows(self, x: np.ndarray) -> np.ndarray:
        """Extract windows for convolution using NumPy's stride tricks"""
        # Add padding if needed
        if self.padding > 0:
            pad_width = [(0, 0)] * (x.ndim - 1) + [(self.padding, self.padding)]
            x = np.pad(x, pad_width, mode='constant')
            
        # Calculate output shape
        N, C, L = x.shape
        out_length = ((L - self.kernel_size + 2 * self.padding) // self.stride) + 1
        
        # Extract windows
        windows = np.zeros((N, C, out_length, self.kernel_size))
        for i in range(out_length):
            start_idx = i * self.stride
            windows[:, :, i, :] = x[:, :, start_idx:start_idx + self.kernel_size]
        return windows
        
    def forward(self, x: Tensor) -> Tensor:
        N = x.data.shape[0]  # Batch size
        windows = self._extract_windows(x.data)
        
        # Perform convolution
        out_data = np.tensordot(windows, self.weight.data, axes=[(1, 3), (1, 2)])
        out_data = np.transpose(out_data, (0, 2, 1))  # Adjust dimensions
        out_data = out_data + self.bias.data.reshape(1, -1, 1)  # Add bias
        
        return Tensor(out_data, requires_grad=x.requires_grad)

class MaxPool1D(Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.indices = None  # Store indices for backward pass
        
    def forward(self, x: Tensor) -> Tensor:
        N, C, L = x.data.shape
        out_length = ((L - self.kernel_size) // self.stride) + 1
        
        # Initialize output and indices arrays
        out_data = np.zeros((N, C, out_length))
        self.indices = np.zeros((N, C, out_length), dtype=int)
        
        # Perform max pooling
        for i in range(out_length):
            start_idx = i * self.stride
            end_idx = start_idx + self.kernel_size
            window = x.data[:, :, start_idx:end_idx]
            out_data[:, :, i] = np.max(window, axis=2)
            self.indices[:, :, i] = start_idx + np.argmax(window, axis=2)
            
        return Tensor(out_data, requires_grad=x.requires_grad)

class BatchNorm1D(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self._parameters.extend([self.gamma, self.beta])
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backward pass
        self.cache = {}
        
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Calculate batch statistics
            batch_mean = np.mean(x.data, axis=0)
            batch_var = np.mean((x.data - batch_mean) ** 2, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize
            x_centered = x.data - batch_mean
            x_normalized = x_centered / np.sqrt(batch_var + self.eps)
            
            # Scale and shift
            out = self.gamma.data * x_normalized + self.beta.data
            
            # Cache variables needed for backward pass
            self.cache = {
                'x_normalized': x_normalized,
                'x_centered': x_centered,
                'batch_mean': batch_mean,
                'batch_var': batch_var,
                'x': x.data,
                'gamma': self.gamma.data,
                'ivar': 1.0 / np.sqrt(batch_var + self.eps)
            }
        else:
            # Use running statistics
            x_normalized = (x.data - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma.data * x_normalized + self.beta.data
        
        out_tensor = Tensor(out, requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad)
        
        if out_tensor.requires_grad:
            def _backward():
                if not self.training:
                    return
                    
                N = x.data.shape[0]
                cache = self.cache
                
                # Gradient with respect to output
                dout = out_tensor.grad
                
                # Gradient with respect to beta
                if self.beta.requires_grad:
                    self.beta.grad += np.sum(dout, axis=0)
                
                # Gradient with respect to gamma
                if self.gamma.requires_grad:
                    self.gamma.grad += np.sum(dout * cache['x_normalized'], axis=0)
                
                # Gradient with respect to x_normalized
                dx_normalized = dout * cache['gamma']
                
                # Gradient with respect to variance
                dvar = np.sum(dx_normalized * cache['x_centered'] * -0.5 * 
                            cache['ivar']**3, axis=0)
                
                # Gradient with respect to mean
                dmu = np.sum(dx_normalized * -cache['ivar'], axis=0) + \
                      dvar * -2.0 * np.mean(cache['x_centered'], axis=0)
                
                # Gradient with respect to x
                if x.requires_grad:
                    dx = dx_normalized * cache['ivar'] + \
                         dvar * 2.0 * cache['x_centered'] / N + \
                         dmu / N
                    x.grad += dx
            
            out_tensor._backward = _backward
            out_tensor.children = {x, self.gamma, self.beta}
            x._in_graph = True
            self.gamma._in_graph = True
            self.beta._in_graph = True
        
        return out_tensor

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                x.grad += (x.data > 0) * out.grad
                
        if out.requires_grad:
            out._backward = _backward
            out.children = {x}
            x._in_graph = True
            
        return out

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(1 / (1 + np.exp(-x.data)), requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                x.grad += out.data * (1 - out.data) * out.grad
                
        if out.requires_grad:
            out._backward = _backward
            out.children = {x}
            x._in_graph = True
            
        return out

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(np.tanh(x.data), requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                x.grad += (1 - out.data**2) * out.grad
                
        if out.requires_grad:
            out._backward = _backward
            out.children = {x}
            x._in_graph = True
            
        return out 