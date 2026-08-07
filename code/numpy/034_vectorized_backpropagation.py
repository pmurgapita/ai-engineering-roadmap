import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def mse(predictions, targets):
    return np.mean(
        (predictions - targets) ** 2
    )


# --------------------------------
# Data
# --------------------------------

X = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.5],
    [3.0, 2.0, 1.0],
    [0.5, 1.5, 2.0],
    [4.0, 1.0, 2.0],
])

Y = np.array([
    [1.0, 2.0],
    [0.0, 1.0],
    [0.0, 1.5],
    [1.0, 1.0],
    [-2.0, 4.0],
])


# --------------------------------
# Parameters
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

learning_rate = 0.01


# --------------------------------
# Forward
# --------------------------------

Z1 = X @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2

loss = mse(Z2, Y)


# --------------------------------
# Backward
# --------------------------------

dZ2 = (
    2
    * (Z2 - Y)
    / Z2.size
)

dW2 = A1.T @ dZ2

db2 = np.sum(
    dZ2,
    axis=0,
)

dA1 = dZ2 @ W2.T

dZ1 = (
    dA1
    * relu_derivative(Z1)
)

dW1 = X.T @ dZ1

db1 = np.sum(
    dZ1,
    axis=0,
)


# --------------------------------
# Display
# --------------------------------

print("Initial predictions:")
print(Z2)

print("\nInitial loss:")
print(loss)

print("\ndZ2:")
print(dZ2)
print("Shape:", dZ2.shape)

print("\ndW2:")
print(dW2)
print("Shape:", dW2.shape)

print("\ndb2:")
print(db2)
print("Shape:", db2.shape)

print("\ndA1 shape:")
print(dA1.shape)

print("\ndZ1:")
print(dZ1)
print("Shape:", dZ1.shape)

print("\ndW1:")
print(dW1)
print("Shape:", dW1.shape)

print("\ndb1:")
print(db1)
print("Shape:", db1.shape)


# --------------------------------
# Update
# --------------------------------

W1 -= learning_rate * dW1
b1 -= learning_rate * db1

W2 -= learning_rate * dW2
b2 -= learning_rate * db2


# --------------------------------
# New forward
# --------------------------------

new_Z1 = X @ W1 + b1
new_A1 = relu(new_Z1)

new_Z2 = new_A1 @ W2 + b2

new_loss = mse(
    new_Z2,
    Y,
)

print("\nNew predictions:")
print(new_Z2)

print("\nNew loss:")
print(new_loss)

for epoch in range(100):
    # Forward
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2

    loss = mse(Z2, Y)

    # Backward
    dZ2 = 2 * (Z2 - Y) / Z2.size

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T

    dZ1 = (
        dA1
        * relu_derivative(Z1)
    )

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 10 == 0:
        print(epoch, loss)