import numpy as np
import sys
print("Python version:", sys.version)
print("NumPy version:", np.__version__)

try:
    from engine import Tensor
    print("Successfully imported Tensor")
except ImportError as e:
    print("Failed to import Tensor:", e)

try:
    from scientific import clip_gradients, estimate_bounds
    print("Successfully imported scientific functions")
except ImportError as e:
    print("Failed to import scientific functions:", e)

try:
    from diagnostics import GradientTracker, detect_gradient_pathologies, ActivationHook
    print("Successfully imported diagnostic functions")
except ImportError as e:
    print("Failed to import diagnostic functions:", e)

try:
    from layers import Linear, ReLU
    print("Successfully imported layers")
except ImportError as e:
    print("Failed to import layers:", e)

try:
    from models import Sequential
    print("Successfully imported Sequential")
except ImportError as e:
    print("Failed to import Sequential:", e)

import traceback

def run_test(test_fn):
    """Run a test function and print any errors."""
    print(f"\nRunning {test_fn.__name__}...")
    try:
        test_fn()
    except Exception as e:
        print(f"\nError in {test_fn.__name__}:")
        print(traceback.format_exc())
        return False
    return True

def test_gradient_clipping():
    """Test gradient clipping."""
    params = [
        Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True),
        Tensor([5.0, 6.0], requires_grad=True)
    ]
    
    # Set some large gradients
    params[0].grad = np.array([[10.0, 20.0], [30.0, 40.0]])
    params[1].grad = np.array([50.0, 60.0])
    
    max_norm = 1.0
    clip_gradients(params, max_norm)
    
    # Check that total gradient norm equals max_norm
    total_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in params))
    assert np.abs(total_norm - max_norm) < 1e-6
    print("Gradient clipping test passed!")

def test_bound_estimation():
    """Test bound estimation."""
    def f(x: Tensor) -> Tensor:
        return x * x
    
    x = Tensor([1.0])
    lower, upper = estimate_bounds(f, x, eps=0.1)
    
    # Function should be monotonic increasing for x > 0
    assert lower < f(x).data < upper
    print("Bound estimation test passed!")

def test_gradient_tracking():
    """Test gradient tracking utilities."""
    # Create a simple model
    model = Sequential([
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    ])
    
    tracker = GradientTracker(model)
    
    # Forward and backward pass with scalar output
    x = Tensor(np.random.randn(3, 2))
    y = model(x)
    # Sum the output to get a scalar for backward
    y.sum().backward()
    
    # Track gradients
    grad_norms = tracker.track_gradients()
    
    # Should have entries for all parameters
    assert len(grad_norms) == len(list(model.parameters()))
    print("Gradient tracking test passed!")

def test_activation_hook():
    """Test activation hook."""
    layer = Linear(2, 4)
    hook = ActivationHook(layer, "linear1")
    
    # Forward pass
    x = Tensor(np.random.randn(3, 2))
    _ = layer(x)
    
    # Check statistics
    stats = hook.get_statistics()
    assert 'mean' in stats
    assert 'std' in stats
    assert 'fraction_dead' in stats
    print("Activation hook test passed!")

def test_gradient_pathology_detection():
    """Test gradient pathology detection."""
    # Simulate gradient history
    grad_history = {
        'param_1': [1e-7] * 10,  # Vanishing
        'param_2': [1e3] * 10,   # Exploding
        'param_3': [1.0] * 10    # Normal
    }
    
    warnings = detect_gradient_pathologies(grad_history)
    
    assert 'param_1' in warnings  # Should detect vanishing
    assert 'param_2' in warnings  # Should detect exploding
    assert 'param_3' not in warnings  # Should be fine
    print("Gradient pathology detection test passed!")

if __name__ == "__main__":
    print("\nStarting tests...")
    tests = [
        test_gradient_clipping,
        test_bound_estimation,
        test_gradient_tracking,
        test_activation_hook,
        test_gradient_pathology_detection
    ]
    
    all_passed = True
    for test in tests:
        if not run_test(test):
            all_passed = False
            
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.") 