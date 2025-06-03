import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Set
from engine import Tensor

# Computational graph visualization using matplotlib
def visualize_graph(tensor: Tensor, save_path: str = None):
    nodes, edges = {}, []
    
    def build_graph(node: Tensor, visited: Set[Tensor]):
        if node in visited:
            return
        visited.add(node)
        nodes[id(node)] = f"{node.op}\\n{node.data.shape}"
        
        for child in node.children:
            edges.append((id(child), id(node)))
            build_graph(child, visited)
    
    build_graph(tensor, set())
    
    # Use matplotlib to create a simple directed graph
    plt.figure(figsize=(10, 8))
    pos = {}
    levels = _assign_levels(tensor)
    
    # Position nodes by level
    for node_id, label in nodes.items():
        level = levels[node_id]
        x = level
        y = sum(1 for _, l in levels.items() if l == level)
        pos[node_id] = (x, y)
    
    # Draw edges
    for start, end in edges:
        plt.plot([pos[start][0], pos[end][0]], 
                [pos[start][1], pos[end][1]], 'k-')
    
    # Draw nodes
    for node_id, (x, y) in pos.items():
        plt.plot(x, y, 'wo', markersize=20)
        plt.text(x, y, nodes[node_id], ha='center', va='center')
    
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def _assign_levels(tensor: Tensor) -> Dict[int, int]:
    levels = {}
    def assign_level(node: Tensor, level: int):
        levels[id(node)] = level
        for child in node.children:
            assign_level(child, level - 1)
    assign_level(tensor, 0)
    return levels

# Taylor series visualization
def taylor_plot(f: Callable[[Tensor], Tensor], x0: float, order: int, 
               x_range: tuple = (-2, 2), num_points: int = 100):
    x = np.linspace(x_range[0], x_range[1], num_points)
    plt.figure(figsize=(10, 6))
    
    # Plot original function
    y_true = [f(Tensor([xi])).data[0] for xi in x]
    plt.plot(x, y_true, 'b-', label='f(x)')
    
    # Compute derivatives at x0
    derivatives = []
    x_tensor = Tensor([x0], requires_grad=True)
    for n in range(order + 1):
        if n == 0:
            y = f(x_tensor)
            derivatives.append(y.data[0])
        else:
            y.backward()
            derivatives.append(x_tensor.grad[0] / np.math.factorial(n))
            x_tensor.grad = np.zeros_like(x_tensor.grad)
            y = f(x_tensor)
    
    # Plot Taylor approximations
    colors = plt.cm.rainbow(np.linspace(0, 1, order + 1))
    for n in range(order + 1):
        y_taylor = np.zeros_like(x)
        for k in range(n + 1):
            y_taylor += derivatives[k] * (x - x0) ** k
        plt.plot(x, y_taylor, c=colors[n], 
                label=f'Taylor order {n}', alpha=0.7)
    
    plt.grid(True)
    plt.legend()
    plt.title(f'Taylor Series Approximation around x₀={x0}')
    plt.show()
    plt.close() 