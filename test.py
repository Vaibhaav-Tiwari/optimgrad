import numpy as np
from engine import Tensor, relu, sigmoid, tanh, Dropout
from layers import Linear, Conv1D, MaxPool1D, BatchNorm1D, ReLU, Sigmoid, Tanh
from models import MLP, Sequential, save_model, load_model
from losses import mse_loss, bce_loss, cross_entropy_loss
from train import SGD, Adam, DataLoader, LRScheduler, train
from viz import visualize_graph, taylor_plot
from explain import GradientExplainer, ChainRuleWalkthrough

def test_scalar_ops():
    # Test basic scalar operations
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    
    z = x * y + x**2
    z.backward()
    
    assert np.allclose(x.grad, [7.0])  # dy/dx = y + 2x = 3 + 4 = 7
    assert np.allclose(y.grad, [2.0])  # dy/dy = x = 2
    print("Scalar operations test passed!")

def test_broadcasting():
    # Test broadcasting operations
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x * 2 + 1
    y.sum().backward()
    
    assert np.allclose(x.grad, [[2.0, 2.0], [2.0, 2.0]])
    print("Broadcasting operations test passed!")

def test_activation_functions():
    # Test activation functions
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # ReLU
    y = relu(x)
    y.sum().backward()
    assert np.allclose(x.grad, [0.0, 0.0, 0.0, 1.0, 1.0])
    x.grad = np.zeros_like(x.data)
    
    # Sigmoid
    y = sigmoid(x)
    y.sum().backward()
    sigmoid_grad = y.data * (1 - y.data)
    assert np.allclose(x.grad, sigmoid_grad)
    x.grad = np.zeros_like(x.data)
    
    # Tanh
    y = tanh(x)
    y.sum().backward()
    tanh_grad = 1 - y.data**2
    assert np.allclose(x.grad, tanh_grad)
    
    print("Activation functions test passed!")

def test_loss_functions():
    # Test MSE Loss
    pred = Tensor([0.5, 0.2, 0.1], requires_grad=True)
    target = Tensor([1.0, 0.0, 0.0])
    
    loss = mse_loss(pred, target)
    loss.backward()
    expected_mse = ((0.5 - 1.0)**2 + (0.2 - 0.0)**2 + (0.1 - 0.0)**2) / 3
    assert np.allclose(loss.data, expected_mse)
    
    # Test BCE Loss
    pred = Tensor([0.7, 0.3], requires_grad=True)
    target = Tensor([1.0, 0.0])
    
    loss = bce_loss(pred, target)
    loss.backward()
    
    print("Loss functions test passed!")

def test_non_scalar_backward():
    # Test non-scalar output gradients
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x * 2
    # Initialize gradient for non-scalar output
    grad = np.ones_like(y.data)
    y.backward(grad)  # Pass gradient for matrix output
    assert np.allclose(x.grad, np.ones_like(x.data) * 2)  # Should be 2 everywhere
    print("Non-scalar backward test passed!")

def test_dropout():
    # Test dropout layer
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    dropout = Dropout(p=0.5)
    
    # Test training mode
    y = dropout(x)
    assert y.data.shape == x.data.shape
    
    # Test eval mode
    dropout.eval()
    y = dropout(x)
    assert np.allclose(y.data, x.data)
    print("Dropout test passed!")

def test_gradient_explanation():
    # Test gradient explanation
    x = Tensor([2.0], requires_grad=True)
    y = x * x + x
    y.backward()
    
    explainer = GradientExplainer(y)
    explanations = explainer.explain_backward()
    assert len(explanations) > 0
    
    walkthrough = ChainRuleWalkthrough(y)
    steps = walkthrough.generate_walkthrough()
    assert len(steps) > 0
    print("Gradient explanation test passed!")

def test_linear_layer():
    # Test linear layer with proper dimensions
    batch_size, in_features, out_features = 2, 3, 4
    layer = Linear(in_features, out_features)
    x = Tensor(np.random.randn(batch_size, in_features), requires_grad=True)
    out = layer(x)
    assert out.data.shape == (batch_size, out_features)
    print("Linear layer test passed!")

def test_conv1d():
    # Test Conv1D layer
    conv = Conv1D(in_channels=2, out_channels=3, kernel_size=2, stride=1)
    x = Tensor(np.random.randn(4, 2, 5))  # (batch, channels, length)
    out = conv(x)
    assert out.data.shape == (4, 3, 4)  # Output length = (L - K + 2P)/S + 1
    print("Conv1D test passed!")

def test_maxpool1d():
    # Test MaxPool1D layer
    pool = MaxPool1D(kernel_size=2, stride=2)
    x = Tensor(np.random.randn(4, 2, 6))
    out = pool(x)
    assert out.data.shape == (4, 2, 3)
    print("MaxPool1D test passed!")

def test_batchnorm1d():
    # Test BatchNorm1D layer
    bn = BatchNorm1D(num_features=3)
    x = Tensor(np.random.randn(10, 3))
    out = bn(x)
    assert out.data.shape == (10, 3)
    
    # Check mean and variance (using population variance)
    mean = np.mean(out.data, axis=0)
    var = np.mean((out.data - mean) ** 2, axis=0)  # Population variance
    print(f"\nBatchNorm1D test details:")
    print(f"Mean (should be close to 0): {mean}")
    print(f"Variance (should be close to 1): {var}")
    
    assert np.allclose(mean, 0.0, atol=1e-6), f"Mean {mean} is not close to 0"
    assert np.allclose(var, 1.0, atol=1e-4), f"Variance {var} is not close to 1"
    print("BatchNorm1D test passed!")

def test_mlp():
    # Test MLP model
    model = MLP([2, 4, 1], activation='relu')
    x = Tensor(np.random.randn(3, 2))
    out = model(x)
    assert out.data.shape == (3, 1)
    print("MLP test passed!")

def test_sequential():
    # Test Sequential model
    model = Sequential([
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    ])
    x = Tensor(np.random.randn(3, 2))
    out = model(x)
    assert out.data.shape == (3, 1)
    print("Sequential test passed!")

def test_cross_entropy():
    # Test CrossEntropy loss
    pred = Tensor(np.random.randn(5, 3))
    target = np.array([0, 2, 1, 1, 0])
    loss = cross_entropy_loss(pred, target)
    assert isinstance(loss.data, (float, np.ndarray))
    print("CrossEntropy loss test passed!")

def test_save_load():
    # Test model save/load
    model = MLP([2, 4, 1])
    x = Tensor(np.random.randn(3, 2))
    out1 = model(x)
    
    # Save model
    save_model(model, 'model.pkl')
    
    # Load model
    model2 = MLP([2, 4, 1])
    load_model(model2, 'model.pkl')
    out2 = model2(x)
    
    assert np.allclose(out1.data, out2.data)
    print("Model save/load test passed!")

def test_lr_scheduler():
    # Test learning rate scheduler
    model = MLP([2, 4, 1])
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = LRScheduler(optimizer, mode='step', gamma=0.1, step_size=2)
    
    initial_lr = optimizer.lr
    scheduler.step()  # epoch 1
    assert optimizer.lr == initial_lr
    
    scheduler.step()  # epoch 2
    assert optimizer.lr == initial_lr * 0.1
    print("Learning rate scheduler test passed!")

def test_integration():
    """Integration test with small XOR dataset"""
    # Create XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Create model with proper initialization
    model = Sequential([
        Linear(2, 8, init_type='he'),  # He initialization for ReLU
        ReLU(),
        Linear(8, 8, init_type='he'),  # He initialization for ReLU
        ReLU(),
        Linear(8, 1, init_type='xavier'),  # Xavier for sigmoid output
        Sigmoid()
    ])
    
    # Use Adam optimizer with proper learning rate
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Create data loader with full batch (since dataset is tiny)
    train_loader = DataLoader(X, y, batch_size=4, shuffle=True)
    
    # Train model
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        loss_fn=bce_loss,
        optimizer=optimizer,
        num_epochs=2000,
        verbose=False
    )
    
    # Test predictions
    model.eval()
    predictions = model(Tensor(X))
    print("\nXOR Problem Predictions:")
    print("Input | Target | Prediction")
    print("-" * 30)
    for i in range(len(X)):
        print(f"{X[i]} | {y[i][0]:.1f}    | {predictions.data[i][0]:.3f}")
    
    # Convert predictions to binary
    binary_predictions = (predictions.data > 0.5).astype(float)
    accuracy = np.mean(binary_predictions == y)
    print(f"\nFinal accuracy: {accuracy:.3f}")
    assert accuracy > 0.9, f"XOR accuracy {accuracy} is too low"
    print("Integration test passed!")

if __name__ == "__main__":
    test_scalar_ops()
    test_broadcasting()
    test_activation_functions()
    test_loss_functions()
    test_non_scalar_backward()
    test_dropout()
    test_gradient_explanation()
    test_linear_layer()
    test_conv1d()
    test_maxpool1d()
    test_batchnorm1d()
    test_mlp()
    test_sequential()
    test_cross_entropy()
    test_save_load()
    test_lr_scheduler()
    test_integration() 