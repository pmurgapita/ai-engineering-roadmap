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


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)


def output_layer_gradients(
    hidden_outputs,
    loss_prediction_gradient,
):
    weight_gradients = []

    for hidden_output in hidden_outputs:
        gradient = loss_prediction_gradient * hidden_output
        weight_gradients.append(gradient)

    bias_gradient = loss_prediction_gradient

    return weight_gradients, bias_gradient


inputs = [2, 3]
y_true = 2.0

hidden_weights = [
    [0.5, -1.0],
    [1.5, 1.0],
]

hidden_biases = [0.5, -1.0]

output_weights = [
    [2.0, -0.5],
]

output_biases = [1.0]

learning_rate = 0.01

# Forward pass
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
)[0]

loss = squared_error(y_true, prediction)

# Beginning of backward pass
loss_prediction_gradient = squared_error_derivative(
    y_true,
    prediction,
)

weight_gradients, bias_gradient = output_layer_gradients(
    hidden_outputs,
    loss_prediction_gradient,
)

print("Initial hidden outputs:", hidden_outputs)
print("Initial prediction:", prediction)
print("Initial loss:", loss)

print("\nLoss gradient with respect to prediction:")
print(loss_prediction_gradient)

print("\nOutput weight gradients:")
print(weight_gradients)

print("Output bias gradient:")
print(bias_gradient)

# Update output layer
for i in range(len(output_weights[0])):
    output_weights[0][i] -= (
        learning_rate * weight_gradients[i]
    )

output_biases[0] -= learning_rate * bias_gradient

# New forward pass
new_prediction = layer_forward(
    hidden_outputs,
    output_weights,
    output_biases,
)[0]

new_loss = squared_error(y_true, new_prediction)

print("\nUpdated output weights:")
print(output_weights)

print("Updated output biases:")
print(output_biases)

print("\nNew prediction:")
print(new_prediction)

print("New loss:")
print(new_loss)

hidden_outputs = [0, 5.0]

output_weights = [
    [2.0, -0.5],
]

output_biases = [1.0]

y_true = 2.0
learning_rate = 0.01

for epoch in range(10):
    prediction = layer_forward(
        hidden_outputs,
        output_weights,
        output_biases,
    )[0]

    loss = squared_error(y_true, prediction)

    loss_prediction_gradient = squared_error_derivative(
        y_true,
        prediction,
    )

    weight_gradients, bias_gradient = output_layer_gradients(
        hidden_outputs,
        loss_prediction_gradient,
    )

    print(f"Epoch: {epoch}")
    print(f"Prediction: {prediction:.6f}")
    print(f"Loss: {loss:.6f}")
    print(f"Output weights: {output_weights}")
    print(f"Output bias: {output_biases[0]:.6f}")
    print()

    for i in range(len(output_weights[0])):
        output_weights[0][i] -= (
            learning_rate * weight_gradients[i]
        )

    output_biases[0] -= learning_rate * bias_gradient