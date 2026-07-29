def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return 1 if x > 0 else 0


def squared_error(y_true, y_pred):
    return (y_pred - y_true) ** 2


def squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)


def hidden_layer_forward(inputs, weights, biases):
    z_values = []
    activations = []

    for neuron_index in range(len(weights)):
        z = (
            dot_product(inputs, weights[neuron_index])
            + biases[neuron_index]
        )

        z_values.append(z)
        activations.append(relu(z))

    return z_values, activations


def output_layer_forward(hidden_outputs, weights, bias):
    return dot_product(hidden_outputs, weights) + bias


def train_step(
    inputs,
    y_true,
    hidden_weights,
    hidden_biases,
    output_weights,
    output_bias,
    learning_rate,
):
    # -------------------------
    # Forward pass
    # -------------------------

    z_hidden, a_hidden = hidden_layer_forward(
        inputs,
        hidden_weights,
        hidden_biases,
    )

    prediction = output_layer_forward(
        a_hidden,
        output_weights,
        output_bias,
    )

    loss = squared_error(y_true, prediction)

    # -------------------------
    # Backward: output
    # -------------------------

    loss_prediction_gradient = (
        squared_error_derivative(y_true, prediction)
    )

    output_weight_gradients = [
        loss_prediction_gradient * activation
        for activation in a_hidden
    ]

    output_bias_gradient = loss_prediction_gradient

    # -------------------------
    # Backward: hidden
    # -------------------------

    hidden_activation_gradients = [
        loss_prediction_gradient * output_weight
        for output_weight in output_weights
    ]

    hidden_z_gradients = [
        hidden_activation_gradients[i]
        * relu_derivative(z_hidden[i])
        for i in range(len(z_hidden))
    ]

    hidden_weight_gradients = []

    for hidden_gradient in hidden_z_gradients:
        neuron_gradients = []

        for input_value in inputs:
            neuron_gradients.append(
                hidden_gradient * input_value
            )

        hidden_weight_gradients.append(
            neuron_gradients
        )

    hidden_bias_gradients = hidden_z_gradients.copy()

    # -------------------------
    # Update output layer
    # -------------------------

    for i in range(len(output_weights)):
        output_weights[i] -= (
            learning_rate
            * output_weight_gradients[i]
        )

    output_bias -= (
        learning_rate * output_bias_gradient
    )

    # -------------------------
    # Update hidden layer
    # -------------------------

    for neuron_index in range(len(hidden_weights)):
        for weight_index in range(
            len(hidden_weights[neuron_index])
        ):
            hidden_weights[neuron_index][weight_index] -= (
                learning_rate
                * hidden_weight_gradients[neuron_index][weight_index]
            )

        hidden_biases[neuron_index] -= (
            learning_rate
            * hidden_bias_gradients[neuron_index]
        )

    return prediction, loss, output_bias


inputs = [2, 3]
y_true = 2.0
learning_rate = 0.01

hidden_weights = [
    [0.5, -1.0],
    [1.5, 1.0],
]

hidden_biases = [0.5, -1.0]

output_weights = [2.0, -0.5]
output_bias = 1.0

for epoch in range(10):
    prediction, loss, output_bias = train_step(
        inputs,
        y_true,
        hidden_weights,
        hidden_biases,
        output_weights,
        output_bias,
        learning_rate,
    )

    print(
        f"Epoch {epoch:02d} | "
        f"prediction = {prediction:.6f} | "
        f"loss = {loss:.6f}"
    )