import math


def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def linear(x, w, b):
    return x * w + b


inputs = [-10, -5, -2, 0, 2, 5, 10]

print("ReLU:")
for x in inputs:
    print(f"x = {x} | ReLU(x) = {relu(x)}")

print("\nSigmoid:")
for x in inputs:
    print(f"x = {x} | sigmoid(x) = {sigmoid(x):.4f}")

print("\nLinear layer without activation:")
for x in inputs:
    z = linear(x, 2, 1)
    print(f"x = {x} | z = {z}")

print("\nLinear layer with ReLU:")
for x in inputs:
    z = linear(x, 2, 1)
    a = relu(z)
    print(f"x = {x} | z = {z} | ReLU(z) = {a}")