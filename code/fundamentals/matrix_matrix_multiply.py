def matrix_shape(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    return rows, columns


def get_column(matrix, column_index):
    column = []

    for row in matrix:
        column.append(row[column_index])

    return column


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def matrix_multiply(A, B):
    a_rows, a_cols = matrix_shape(A)
    b_rows, b_cols = matrix_shape(B)

    if a_cols != b_rows:
        raise ValueError("No se pueden multiplicar: dimensiones incompatibles")

    result = []

    for row in A:
        result_row = []

        for column_index in range(b_cols):
            column = get_column(B, column_index)
            value = dot_product(row, column)
            result_row.append(value)

        result.append(result_row)

    return result


def add_bias(matrix, bias):
    result = []

    for row in matrix:
        result_row = []

        for i in range(len(row)):
            result_row.append(row[i] + bias[i])

        result.append(result_row)

    return result


X = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

W = [
    [2.5, 0.3],
    [12, 5],
    [-7, -2],
]

b = [40, 10]

raw_outputs = matrix_multiply(X, W)
outputs = add_bias(raw_outputs, b)

print("Shape of X:", matrix_shape(X))
print("Shape of W:", matrix_shape(W))
print("Raw outputs:")
for row in raw_outputs:
    print(row)

print("\nOutputs:")
for row in outputs:
    print(row)


def count_parameters(W, b):
    rows, columns = matrix_shape(W)

    number_of_weights = rows * columns
    number_of_biases = len(b)
    total_parameters = number_of_weights + number_of_biases

    return number_of_weights, number_of_biases, total_parameters


weights_count, bias_count, total_count = count_parameters(W, b)

print("Weights:", weights_count)
print("Biases:", bias_count)
print("Total parameters:", total_count)