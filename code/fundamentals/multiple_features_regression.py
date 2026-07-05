def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def predict(features, weights, bias):
    return dot_product(features, weights) + bias


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def gradients(features, y_true, y_pred):
    error = y_true - y_pred

    weight_gradients = []

    for feature in features:
        weight_gradients.append(-2 * feature * error)

    bias_gradient = -2 * error

    return weight_gradients, bias_gradient


houses = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

prices = [
    139,
    241,
    376,
    167,
    319,
]

weights = [0.0, 0.0, 0.0]
bias = 0.0

learning_rate = 0.00001
epochs = 1000

for epoch in range(epochs):
    total_loss = 0

    total_weight_gradients = [0.0, 0.0, 0.0]
    total_bias_gradient = 0

    for features, price in zip(houses, prices):
        prediction = predict(features, weights, bias)

        loss = squared_error(price, prediction)
        weight_gradients, bias_gradient = gradients(features, price, prediction)

        total_loss += loss

        for i in range(len(weights)):
            total_weight_gradients[i] += weight_gradients[i]

        total_bias_gradient += bias_gradient

    mean_loss = total_loss / len(houses)

    mean_weight_gradients = [
        gradient / len(houses)
        for gradient in total_weight_gradients
    ]

    mean_bias_gradient = total_bias_gradient / len(houses)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:04d} | "
            f"weights = {[round(w, 4) for w in weights]} | "
            f"bias = {bias:.4f} | "
            f"loss = {mean_loss:.4f}"
        )

    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * mean_weight_gradients[i]

    bias = bias - learning_rate * mean_bias_gradient

print("Final weights:", weights)
print("Final bias:", bias)
print("Final loss:", mean_loss)

def predict_price(meters, rooms, distance):
    return meters * weights[0] + rooms * weights[1] + distance * weights[2] + bias

print(predict_price(90, 3, 4))