import numpy as np


# --------------------------------
# Python lists vs NumPy arrays
# --------------------------------

python_vector = [1, 2, 3]
python_other_vector = [10, 20, 30]

numpy_vector = np.array(
    [1, 2, 3],
    dtype=float,
)

numpy_other_vector = np.array(
    [10, 20, 30],
    dtype=float,
)

print("Python list * 2:")
print(python_vector * 2)

print("\nNumPy array * 2:")
print(numpy_vector * 2)

print("\nPython list + list:")
print(python_vector + python_other_vector)

print("\nNumPy array + array:")
print(numpy_vector + numpy_other_vector)


# --------------------------------
# Array attributes
# --------------------------------

matrix = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ],
    dtype=float,
)

tensor_3d = np.array(
    [
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ],
    ],
    dtype=float,
)

print("\nVector:")
print(numpy_vector)
print("Shape:", numpy_vector.shape)
print("Dimensions:", numpy_vector.ndim)
print("Size:", numpy_vector.size)
print("Data type:", numpy_vector.dtype)

print("\nMatrix:")
print(matrix)
print("Shape:", matrix.shape)
print("Dimensions:", matrix.ndim)
print("Size:", matrix.size)
print("Data type:", matrix.dtype)

print("\nTensor 3D:")
print(tensor_3d)
print("Shape:", tensor_3d.shape)
print("Dimensions:", tensor_3d.ndim)
print("Size:", tensor_3d.size)
print("Data type:", tensor_3d.dtype)


# --------------------------------
# Element-wise vs matrix operation
# --------------------------------

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print("\nElement-wise multiplication:")
print(a * b)

print("\nDot product with @:")
print(a @ b)


# --------------------------------
# Multiple predictions
# --------------------------------

X = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

weights = np.array([0.5, 1.0, -1.0])
bias = 2.0

raw_predictions = X @ weights
predictions = raw_predictions + bias

print("\nX:")
print(X)
print("X shape:", X.shape)

print("\nWeights:")
print(weights)
print("Weights shape:", weights.shape)

print("\nRaw predictions:")
print(raw_predictions)
print("Raw predictions shape:", raw_predictions.shape)

print("\nPredictions with bias:")
print(predictions)