import numpy as np


def relu(x):
    return np.maximum(0, x)


def dense_forward(inputs, weights, biases):
    return inputs @ weights + biases


# --------------------------------
# Batch
# --------------------------------

X = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.5],
    [3.0, 2.0, 1.0],
    [0.5, 1.5, 2.0],
    [4.0, 1.0, 2.0],
])


# --------------------------------
# Hidden layer
# --------------------------------

W1 = np.array([
    [0.2, -0.5, 1.0, 0.3],
    [0.7, 0.1, -0.2, 0.8],
    [-0.4, 0.9, 0.5, -0.6],
])

b1 = np.array([
    0.1,
    -0.2,
    0.0,
    0.3,
])


# --------------------------------
# Output layer
# --------------------------------

W2 = np.array([
    [0.5, -0.3],
    [0.8, 0.2],
    [-0.6, 1.0],
    [0.4, -0.5],
])

b2 = np.array([
    0.2,
    -0.1,
])


# --------------------------------
# Forward pass
# --------------------------------

Z1 = dense_forward(
    X,
    W1,
    b1,
)

A1 = relu(Z1)

Z2 = dense_forward(
    A1,
    W2,
    b2,
)


# --------------------------------
# Results
# --------------------------------

print("X:")
print(X)
print("X shape:", X.shape)

print("\nW1 shape:", W1.shape)
print("b1 shape:", b1.shape)

print("\nZ1:")
print(Z1)
print("Z1 shape:", Z1.shape)

print("\nA1 after ReLU:")
print(A1)
print("A1 shape:", A1.shape)

print("\nW2 shape:", W2.shape)
print("b2 shape:", b2.shape)

print("\nZ2:")
print(Z2)
print("Z2 shape:", Z2.shape)

print("\nParameters:")
print(
    W1.size
    + b1.size
    + W2.size
    + b2.size
)