import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

W1 = np.random.randn(64, 32) * 0.01
b1 = np.zeros(32)

W2 = np.random.randn(32, 10) * 0.01
b2 = np.zeros(10)


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return a1, a2


def cross_entropy_loss(probs, y):
    n = probs.shape[0]
    correct_probs = probs[np.arange(n), y]
    loss = -np.mean(np.log(correct_probs))
    return loss


def one_hot(y, num_classes=10):
    n = y.shape[0]
    Y = np.zeros((n, num_classes))
    Y[np.arange(n), y] = 1
    return Y


learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    a1, a2 = forward(X, W1, b1, W2, b2)
    loss = cross_entropy_loss(a2, y)

    Y_onehot = one_hot(y)
    dz2 = (a2 - Y_onehot) / a2.shape[0]
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (a1 > 0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
