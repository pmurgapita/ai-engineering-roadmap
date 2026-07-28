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
    if x > 0:
        return 1

    return 0


def squared_error(y_true, y_pred):
    return (y_pred - y_true) ** 2


def squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)


def hidden_layer_forward(inputs, weights, biases):
    z_values = []
    activations = []

    for i in range(len(weights)):
        z = dot_product(inputs, weights[i]) + biases[i]
        activation = relu(z)

        z_values.append(z)
        activations.append(activation)

    return z_values, activations


def output_layer_forward(hidden_outputs, weights, bias):
    return dot_product(hidden_outputs, weights) + bias


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
# Backward pass: output
# -------------------------

loss_prediction_gradient = squared_error_derivative(
    y_true,
    prediction,
)

output_weight_gradients = []

for hidden_output in a_hidden:
    gradient = loss_prediction_gradient * hidden_output
    output_weight_gradients.append(gradient)

output_bias_gradient = loss_prediction_gradient

# -------------------------
# Backward pass: hidden activations
# -------------------------

hidden_activation_gradients = []

for output_weight in output_weights:
    gradient = loss_prediction_gradient * output_weight
    hidden_activation_gradients.append(gradient)

# -------------------------
# Backward pass: through ReLU
# -------------------------

hidden_z_gradients = []

for i in range(len(z_hidden)):
    gradient = (
        hidden_activation_gradients[i]
        * relu_derivative(z_hidden[i])
    )

    hidden_z_gradients.append(gradient)

# -------------------------
# Backward pass: hidden parameters
# -------------------------

hidden_weight_gradients = []
hidden_bias_gradients = []

for hidden_gradient in hidden_z_gradients:
    neuron_weight_gradients = []

    for input_value in inputs:
        gradient = hidden_gradient * input_value
        neuron_weight_gradients.append(gradient)

    hidden_weight_gradients.append(
        neuron_weight_gradients
    )

    hidden_bias_gradients.append(hidden_gradient)

# -------------------------
# Display gradients
# -------------------------

print("Initial z hidden:", z_hidden)
print("Initial hidden activations:", a_hidden)
print("Initial prediction:", prediction)
print("Initial loss:", loss)

print("\nLoss-prediction gradient:")
print(loss_prediction_gradient)

print("\nOutput weight gradients:")
print(output_weight_gradients)

print("Output bias gradient:")
print(output_bias_gradient)

print("\nHidden activation gradients:")
print(hidden_activation_gradients)

print("Hidden z gradients:")
print(hidden_z_gradients)

print("\nHidden weight gradients:")
print(hidden_weight_gradients)

print("Hidden bias gradients:")
print(hidden_bias_gradients)

# -------------------------
# Update all parameters
# -------------------------

for i in range(len(output_weights)):
    output_weights[i] -= (
        learning_rate * output_weight_gradients[i]
    )

output_bias -= learning_rate * output_bias_gradient

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

# -------------------------
# New forward pass
# -------------------------

new_z_hidden, new_a_hidden = hidden_layer_forward(
    inputs,
    hidden_weights,
    hidden_biases,
)

new_prediction = output_layer_forward(
    new_a_hidden,
    output_weights,
    output_bias,
)

new_loss = squared_error(
    y_true,
    new_prediction,
)

print("\nUpdated hidden weights:")
print(hidden_weights)

print("Updated hidden biases:")
print(hidden_biases)

print("Updated output weights:")
print(output_weights)

print("Updated output bias:")
print(output_bias)

print("\nNew hidden activations:")
print(new_a_hidden)

print("New prediction:")
print(new_prediction)

print("New loss:")
print(new_loss)

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
        loss_prediction_gradient * hidden_output
        for hidden_output in a_hidden
    ]

    output_bias_gradient = loss_prediction_gradient

    # -------------------------
    # Backward: hidden layer
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

    hidden_weight_gradients = [
        [
            hidden_z_gradients[neuron_index] * input_value
            for input_value in inputs
        ]
        for neuron_index in range(len(hidden_z_gradients))
    ]

    hidden_bias_gradients = hidden_z_gradients.copy()

    # Mostramos el estado antes de actualizar
    print(f"\nEpoch: {epoch}")
    print(f"Prediction: {prediction:.6f}")
    print(f"Loss: {loss:.6f}")
    print(f"Hidden activations: {a_hidden}")
    print(f"Hidden weights: {hidden_weights}")
    print(f"Output weights: {output_weights}")
    print(f"Hidden z gradients: {hidden_z_gradients}")

    # -------------------------
    # Actualización
    # -------------------------

    for i in range(len(output_weights)):
        output_weights[i] -= (
            learning_rate * output_weight_gradients[i]
        )

    output_bias -= learning_rate * output_bias_gradient

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