import numpy as np
from typing import Callable, List, Tuple, Optional
from engine import Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_taylor_approx(f: Callable[[Tensor], Tensor], 
                      x0: Tensor,
                      order: int,
                      range_vals: Tuple[float, float],
                      num_points: int = 100) -> None:
    """Plot function and its Taylor approximation.
    
    Args:
        f: Function to approximate
        x0: Point around which to expand
        order: Order of Taylor approximation
        range_vals: (min, max) range for x-axis
        num_points: Number of points to plot
    """
    x_vals = np.linspace(range_vals[0], range_vals[1], num_points)
    y_true = [f(Tensor([x])).data.item() for x in x_vals]
    
    # Compute derivatives at x0
    derivatives = []
    x = Tensor([x0.data.item()], requires_grad=True)
    for n in range(order + 1):
        if n == 0:
            y = f(x)
            derivatives.append(y.data.item())
        else:
            y.backward()
            derivatives.append(x.grad.item() / np.math.factorial(n))
            x.zero_grad()
            y = f(x)
    
    # Compute Taylor approximation
    y_taylor = np.zeros_like(x_vals)
    for n in range(order + 1):
        y_taylor += derivatives[n] * (x_vals - x0.data.item()) ** n
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_true, 'b-', label='True function')
    plt.plot(x_vals, y_taylor, 'r--', label=f'Taylor order {order}')
    plt.axvline(x=x0.data.item(), color='k', linestyle=':')
    plt.grid(True)
    plt.legend()
    plt.title(f'Taylor Approximation (order {order})')
    plt.show()

def plot_gradient_field(f: Callable[[Tensor], Tensor],
                       x_range: Tuple[float, float],
                       y_range: Tuple[float, float],
                       n_points: int = 20) -> None:
    """Plot gradient field of a 2D function.
    
    Args:
        f: Function R^2 -> R
        x_range: Range for x coordinate
        y_range: Range for y coordinate
        n_points: Number of points in each dimension
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(n_points):
        for j in range(n_points):
            point = Tensor([X[i,j], Y[i,j]], requires_grad=True)
            z = f(point)
            z.backward()
            U[i,j] = point.grad[0]
            V[i,j] = point.grad[1]
    
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V)
    plt.title('Gradient Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def plot_gradient_heatmap(param: Tensor,
                         title: Optional[str] = None) -> None:
    """Plot gradient values as a heatmap.
    
    Args:
        param: Parameter tensor whose gradients to visualize
        title: Optional title for the plot
    """
    if param.grad is None:
        raise ValueError("Parameter has no gradients")
        
    plt.figure(figsize=(10, 8))
    plt.imshow(param.grad, cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='Gradient magnitude')
    if title:
        plt.title(title)
    plt.show()

def saliency_map(output: Tensor,
                 input_tensor: Tensor,
                 abs_val: bool = True) -> np.ndarray:
    """Compute saliency map showing input importance.
    
    Args:
        output: Model output tensor
        input_tensor: Input tensor
        abs_val: Whether to take absolute value of gradients
    
    Returns:
        Numpy array of gradient magnitudes
    """
    if not input_tensor.requires_grad:
        raise ValueError("Input tensor must have requires_grad=True")
        
    # Compute gradients
    output.backward()
    gradients = input_tensor.grad
    
    if abs_val:
        gradients = np.abs(gradients)
    
    # Normalize for visualization
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
    
    return gradients

def plot_activation_distributions(activation_stats: List[Dict[str, float]],
                                layer_name: str) -> None:
    """Plot evolution of activation statistics over time.
    
    Args:
        activation_stats: List of activation statistics dictionaries
        layer_name: Name of the layer to plot
    """
    stats = list(zip(*[(d['mean'], d['std']) for d in activation_stats]))
    means, stds = stats[0], stats[1]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(means, label='Mean')
    plt.fill_between(range(len(means)), 
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.3)
    plt.title(f'{layer_name} Activation Statistics')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(means, bins=30, alpha=0.7, label='Mean dist')
    plt.title('Distribution of Means')
    plt.xlabel('Mean activation')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.show() 