def matrix_shape(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    return rows, columns


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def matrix_vector_multiply(matrix, vector):
    result = []

    for row in matrix:
        value = dot_product(row, vector)
        result.append(value)

    return result


def add_bias(predictions, bias):
    result = []

    for prediction in predictions:
        result.append(prediction + bias)

    return result


X = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

weights = [2.5, 12, -7]
bias = 40

print("Shape of X:", matrix_shape(X))

raw_predictions = matrix_vector_multiply(X, weights)
predictions = add_bias(raw_predictions, bias)

print("Raw predictions:", raw_predictions)
print("Predictions:", predictions)

def mean_squared_error(y_true, y_pred):
    count = 0
    squared_error = []
    sum_squared_errors = 0
    for i in range(len(y_true)):
        squared_error.append((y_true[i] - y_pred[i]) ** 2)
    for each in squared_error:
        count += 1
        sum_squared_errors += each
    return sum_squared_errors / count 

real_prices = [139, 241, 376, 167, 319]
predictions = [121.0, 241.0, 374.0, 140.0, 289.0]

print(mean_squared_error(real_prices, predictions))