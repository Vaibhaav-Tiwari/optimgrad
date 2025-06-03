import numpy as np
from typing import List, Dict, Set, Optional
from engine import Tensor

# Gradient path explanation
class GradientExplainer:
    def __init__(self, tensor: Tensor):
        self.tensor = tensor
        self.paths: List[List[Tensor]] = []
        self.explanations: List[str] = []
        
    def explain_backward(self, target_tensor: Optional[Tensor] = None):
        """Explain gradient paths from target to root"""
        self.paths = []
        self.explanations = []
        
        def find_paths(node: Tensor, current_path: List[Tensor], visited: Set[Tensor]):
            if node in visited:
                return
            visited.add(node)
            
            current_path.append(node)
            if not node.children:  # Leaf node
                if target_tensor is None or node == target_tensor:
                    self.paths.append(current_path.copy())
            else:
                for child in node.children:
                    find_paths(child, current_path, visited.copy())
            current_path.pop()
        
        find_paths(self.tensor, [], set())
        
        # Generate explanations for each path
        for path in self.paths:
            explanation = self._explain_path(path)
            self.explanations.append(explanation)
            
        return self.explanations
    
    def _explain_path(self, path: List[Tensor]) -> str:
        """Generate human-readable explanation for a gradient path"""
        if not path:
            return ""
        
        explanation = []
        for i in range(len(path)-1):
            current, next_node = path[i], path[i+1]
            if current.op:
                explanation.append(f"∂({current.op})/∂({next_node.op or 'input'})")
        
        full_path = " * ".join(reversed(explanation))
        return f"∂L/∂x = {full_path}"

# Chain rule walkthrough
class ChainRuleWalkthrough:
    def __init__(self, tensor: Tensor):
        self.tensor = tensor
        self.steps: List[Dict] = []
        
    def generate_walkthrough(self):
        """Generate step-by-step chain rule application"""
        self.steps = []
        
        def process_node(node: Tensor, visited: Set[Tensor]):
            if node in visited:
                return
            visited.add(node)
            
            # Record step information
            step = {
                'node': node,
                'op': node.op,
                'shape': node.data.shape,
                'parents': node.children,
                'local_grad': self._get_local_grad(node),
                'output_grad': node.grad
            }
            self.steps.append(step)
            
            # Process children
            for child in node.children:
                process_node(child, visited)
        
        process_node(self.tensor, set())
        return self._format_walkthrough()
    
    def _get_local_grad(self, node: Tensor) -> str:
        """Get human-readable local gradient expression"""
        if not node.op:
            return "1"
        
        grad_expressions = {
            '+': '1',
            '*': str(node.children[0].data if node.children else ''),
            'ReLU': '1 if x > 0 else 0',
            'sigmoid': 'σ(x)(1 - σ(x))',
            'tanh': '1 - tanh²(x)'
        }
        
        if node.op.startswith('**'):
            power = float(node.op.split('**')[1])
            return f'{power}x^({power-1})'
        
        return grad_expressions.get(node.op, 'Unknown')
    
    def _format_walkthrough(self) -> List[str]:
        """Format steps into human-readable strings"""
        formatted = []
        for step in reversed(self.steps):
            parents = ', '.join(p.op or 'input' for p in step['parents'])
            formatted.append(
                f"Node: {step['op'] or 'input'} "
                f"(shape: {step['shape']}) ← [{parents}]\n"
                f"Local gradient: {step['local_grad']}\n"
                f"Output gradient: {step['output_grad']}\n"
            )
        return formatted 