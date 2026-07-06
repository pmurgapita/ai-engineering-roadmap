import math


def vector_add(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    result = []

    for i in range(len(a)):
        result.append(a[i] + b[i])

    return result


def vector_subtract(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    result = []

    for i in range(len(a)):
        result.append(a[i] - b[i])

    return result


def scalar_multiply(scalar, vector):
    result = []

    for value in vector:
        result.append(scalar * value)

    return result


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def vector_norm(vector):
    total = 0

    for value in vector:
        total += value ** 2

    return math.sqrt(total)


a = [3, 4]
b = [1, 2]

print("a + b:", vector_add(a, b))
print("a - b:", vector_subtract(a, b))
print("3 * a:", scalar_multiply(3, a))
print("a · b:", dot_product(a, b))
print("||a||:", vector_norm(a))


features = [90, 3, 4]
weights = [2.5, 12, -7]
bias = 40

prediction = dot_product(features, weights) + bias

print("\nPrediction:", prediction)

def cosine_similarity(a, b):
    return dot_product(a, b) / (vector_norm(a) * vector_norm(b))

x = [1, 1]
y = [10, 10]
z = [-1, 1]

print(cosine_similarity(x, y))
print(cosine_similarity(x, z))