import numpy as np
from NeuralNetwork import forward, cross_entropy_loss, one_hot, X, y, W1, b1, W2, b2

# Use only a small slice of data for speed - numerical gradient checking
# requires re-running the forward pass thousands of times
X_small = X[:5]
y_small = y[:5]

# --- Analytical gradients (your backprop) ---
a1, a2 = forward(X_small, W1, b1, W2, b2)
Y_onehot = one_hot(y_small)

dz2 = (a2 - Y_onehot) / a2.shape[0]
dW2_analytical = a1.T @ dz2
db2_analytical = np.sum(dz2, axis=0)

da1 = dz2 @ W2.T
dz1 = da1 * (a1 > 0)
dW1_analytical = X_small.T @ dz1
db1_analytical = np.sum(dz1, axis=0)


# --- Numerical gradient checking ---
def compute_loss(X, y, W1, b1, W2, b2):
    _, a2 = forward(X, W1, b1, W2, b2)
    return cross_entropy_loss(a2, y)


def numerical_gradient(param, X, y, W1, b1, W2, b2, epsilon=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original_value = param[idx]

        param[idx] = original_value + epsilon
        loss_plus = compute_loss(X, y, W1, b1, W2, b2)

        param[idx] = original_value - epsilon
        loss_minus = compute_loss(X, y, W1, b1, W2, b2)

        param[idx] = original_value
        grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        it.iternext()
    return grad


dW1_numerical = numerical_gradient(W1, X_small, y_small, W1, b1, W2, b2)
dW2_numerical = numerical_gradient(W2, X_small, y_small, W1, b1, W2, b2)
db1_numerical = numerical_gradient(b1, X_small, y_small, W1, b1, W2, b2)
db2_numerical = numerical_gradient(b2, X_small, y_small, W1, b1, W2, b2)


# --- Compare ---
def check(name, analytical, numerical, tol=1e-5):
    diff = np.max(np.abs(analytical - numerical))
    status = "PASS" if diff < tol else "FAIL"
    print(f"{name}: {status} (max difference: {diff})")


check("dW1", dW1_analytical, dW1_numerical)
check("db1", db1_analytical, db1_numerical)
check("dW2", dW2_analytical, dW2_numerical)
check("db2", db2_analytical, db2_numerical)
