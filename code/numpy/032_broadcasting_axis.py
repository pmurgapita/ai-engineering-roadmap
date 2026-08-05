import numpy as np


# --------------------------------
# Broadcasting with scalar
# --------------------------------

X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print("X:")
print(X)
print("Shape:", X.shape)

print("\nX + 10:")
print(X + 10)


# --------------------------------
# Broadcasting with row vector
# --------------------------------

row_bias = np.array([10.0, 20.0, 30.0])

print("\nRow bias:")
print(row_bias)
print("Shape:", row_bias.shape)

print("\nX + row bias:")
print(X + row_bias)


# --------------------------------
# Broadcasting with column vector
# --------------------------------

column_values = np.array([
    [100.0],
    [200.0],
])

print("\nColumn values:")
print(column_values)
print("Shape:", column_values.shape)

print("\nX + column values:")
print(X + column_values)


# --------------------------------
# Neural layer
# --------------------------------

inputs = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

weights = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
])

biases = np.array([0.1, 0.2, 0.3, 0.4])

layer_output = inputs @ weights + biases

print("\nInputs shape:", inputs.shape)
print("Weights shape:", weights.shape)
print("Biases shape:", biases.shape)

print("\nLayer output:")
print(layer_output)
print("Layer output shape:", layer_output.shape)


# --------------------------------
# Axis
# --------------------------------

matrix = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print("\nMatrix:")
print(matrix)

print("\nTotal sum:")
print(np.sum(matrix))

print("\nSum axis=0:")
print(np.sum(matrix, axis=0))

print("\nSum axis=1:")
print(np.sum(matrix, axis=1))

print("\nMean axis=0:")
print(np.mean(matrix, axis=0))

print("\nMean axis=1:")
print(np.mean(matrix, axis=1))


# --------------------------------
# keepdims
# --------------------------------

row_sums = np.sum(
    matrix,
    axis=1,
    keepdims=True,
)

print("\nRow sums with keepdims:")
print(row_sums)
print("Shape:", row_sums.shape)

normalized = matrix / row_sums

print("\nRows normalized by their sum:")
print(normalized)


# --------------------------------
# Batch loss
# --------------------------------

predictions = np.array([
    2.5,
    4.0,
    6.5,
])

targets = np.array([
    3.0,
    5.0,
    6.0,
])

errors = predictions - targets
squared_errors = errors ** 2
loss = np.mean(squared_errors)

print("\nPredictions:")
print(predictions)

print("Targets:")
print(targets)

print("Errors:")
print(errors)

print("Squared errors:")
print(squared_errors)

print("MSE:")
print(loss)