def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def neuron_output(inputs, weights, bias):
    return dot_product(inputs, weights) + bias


inputs = [2, 4]
weights = [3, 10]
bias = 5
houses = [
    [50, 1, 8],
    [90, 3, 4],
    [120, 4, 2],
]



for house in houses:
    print("Predicción:", neuron_output(house, weights, bias))