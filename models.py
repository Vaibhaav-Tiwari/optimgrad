from typing import List, Union, Optional
import numpy as np
from engine import Tensor
from layers import Module, Linear, ReLU, Sigmoid, Tanh
from engine import relu, tanh, sigmoid

class Sequential(Module):
    def __init__(self, layers: List[Module]):
        super().__init__()
        self.layers = layers
        # Collect parameters from all layers
        for layer in layers:
            self._parameters.extend(layer.parameters())
            
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()
            
    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()

class MLP(Module):
    def __init__(self, layer_sizes: List[int], 
                 activation: str = 'relu',
                 dropout_prob: Optional[float] = None):
        super().__init__()
        
        # Set activation function
        self.activation_fn = {
            'relu': ReLU,
            'tanh': Tanh,
            'sigmoid': Sigmoid
        }[activation.lower()]
        
        # Build layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add activation for all layers (including last for binary classification)
            layers.append(self.activation_fn())
                
            # Add dropout if specified (except for last layer)
            if dropout_prob is not None and i < len(layer_sizes) - 2:
                from engine import Dropout
                layers.append(Dropout(dropout_prob))
        
        self.model = Sequential(layers)
        self._parameters = self.model.parameters()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()

def save_model(model: Module, path: str):
    """Save model parameters to file"""
    import pickle
    state_dict = {}
    for i, param in enumerate(model.parameters()):
        state_dict[f'param_{i}'] = param.data
    with open(path, 'wb') as f:
        pickle.dump(state_dict, f)

def load_model(model: Module, path: str):
    """Load model parameters from file"""
    import pickle
    with open(path, 'rb') as f:
        state_dict = pickle.load(f)
    for i, param in enumerate(model.parameters()):
        param.data = state_dict[f'param_{i}'] 