def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def relu(x):
    return max(0, x)


def neuron_forward(inputs, weights, bias, activation=None):
    z = dot_product(inputs, weights) + bias

    if activation is None:
        return z

    return activation(z)


def layer_forward(inputs, weights, biases, activation=None):
    if len(weights) != len(biases):
        raise ValueError("Debe haber un bias por neurona")

    outputs = []

    for i in range(len(weights)):
        output = neuron_forward(
            inputs,
            weights[i],
            biases[i],
            activation,
        )

        outputs.append(output)

    return outputs


inputs = [-2, 1]

hidden_weights = [
    [0.5, -1.0],
    [1.5, 1.0],
]

hidden_biases = [0.5, -1.0]

output_weights = [
    [2.0, -0.5],
]

output_biases = [1.0]

hidden_outputs = layer_forward(
    inputs,
    hidden_weights,
    hidden_biases,
    relu,
)

prediction = layer_forward(
    hidden_outputs,
    output_weights,
    output_biases,
)

print("Inputs:", inputs)
print("Hidden outputs:", hidden_outputs)
print("Prediction:", prediction)