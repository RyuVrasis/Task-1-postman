import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
x = digits.data
y = digits.target


w1 = np.random.randn(64, 32)*0.01
b1 = np.zeros(32)

w2 = np.random.randn(32, 10)*0.01
b2 = np.zeros(10)


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, w1, b1, w2, b2):
    z1 = X @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = softmax(z2)
    return a2


output = forward(x, w1, b1, w2, b2)
print(output.shape)
print(output[0])
