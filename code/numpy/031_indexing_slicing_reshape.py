import numpy as np


# --------------------------------
# Indexing
# --------------------------------

X = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
    [100, 110, 120],
])

print("Original X:")
print(X)
print("Shape:", X.shape)

print("\nFirst row:")
print(X[0])
print("Shape:", X[0].shape)

print("\nElement at row 1, column 2:")
print(X[1, 2])

print("\nFirst column:")
print(X[:, 0])
print("Shape:", X[:, 0].shape)

print("\nRows 1 to 2:")
print(X[1:3])
print("Shape:", X[1:3].shape)

print("\nRows 1 to 3, columns 0 to 1:")
print(X[1:4, 0:2])
print("Shape:", X[1:4, 0:2].shape)

print("\nLast row:")
print(X[-1])

print("\nLast element:")
print(X[-1, -1])


# --------------------------------
# Slicing with steps
# --------------------------------

numbers = np.array([0, 1, 2, 3, 4, 5, 6, 7])

print("\nEvery second element:")
print(numbers[::2])

print("\nReversed:")
print(numbers[::-1])


# --------------------------------
# Keeping or removing dimensions
# --------------------------------

column_vector = X[:, 1]
column_matrix = X[:, 1:2]

print("\nColumn as 1D vector:")
print(column_vector)
print("Shape:", column_vector.shape)

print("\nColumn as 2D matrix:")
print(column_matrix)
print("Shape:", column_matrix.shape)


# --------------------------------
# Reshape
# --------------------------------

vector = np.arange(1, 13)

print("\nOriginal vector:")
print(vector)
print("Shape:", vector.shape)

matrix_3x4 = vector.reshape(3, 4)

print("\nReshaped to 3 x 4:")
print(matrix_3x4)
print("Shape:", matrix_3x4.shape)

matrix_4x3 = vector.reshape(4, -1)

print("\nReshaped to 4 x 3 using -1:")
print(matrix_4x3)
print("Shape:", matrix_4x3.shape)

row_matrix = vector.reshape(1, -1)
column_matrix = vector.reshape(-1, 1)

print("\nRow matrix:")
print(row_matrix)
print("Shape:", row_matrix.shape)

print("\nColumn matrix:")
print(column_matrix)
print("Shape:", column_matrix.shape)

flat_vector = matrix_3x4.reshape(-1)

print("\nFlattened with reshape:")
print(flat_vector)
print("Shape:", flat_vector.shape)


# --------------------------------
# Views vs copies
# --------------------------------

original = np.array([
    [1, 2],
    [3, 4],
])

view = original[0:1, :]
view[0, 0] = 999

print("\nOriginal after modifying a view:")
print(original)

original_copy_example = np.array([
    [1, 2],
    [3, 4],
])

independent_copy = original_copy_example[0:1, :].copy()
independent_copy[0, 0] = 999

print("\nIndependent copy:")
print(independent_copy)

print("\nOriginal after modifying the copy:")
print(original_copy_example)


# --------------------------------
# Splitting features and target
# --------------------------------

data = np.array([
    [50, 1, 8, 180],
    [80, 3, 5, 280],
    [120, 4, 2, 420],
    [60, 2, 7, 220],
    [100, 3, 3, 350],
], dtype=float)

features = data[:, 0:3]
targets = data[:, 3]
targets_column = data[:, 3:4]

print("\nFeatures:")
print(features)
print("Features shape:", features.shape)

print("\nTargets as vector:")
print(targets)
print("Targets shape:", targets.shape)

print("\nTargets as column matrix:")
print(targets_column)
print("Targets column shape:", targets_column.shape)

images = np.arange(1, 49).reshape(4, 3, 4)

print(images)
print(images[0])
print(images[0].shape)
print(images[0:1].shape)
print(images[:,0,:])
print(images[:,0,:].shape)