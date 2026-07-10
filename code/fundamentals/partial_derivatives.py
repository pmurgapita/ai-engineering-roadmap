def predict(features, weights, bias):
    total = 0

    for i in range(len(features)):
        total += features[i] * weights[i]

    return total + bias


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def gradients(features, y_true, y_pred):
    error = y_true - y_pred

    weight_gradients = []

    for feature in features:
        gradient = -2 * feature * error
        weight_gradients.append(gradient)

    bias_gradient = -2 * error

    return weight_gradients, bias_gradient


features = [2, 3]
weights = [1, 1]
bias = 0
y_true = 10

learning_rate = 0.01

y_pred = predict(features, weights, bias)
loss = squared_error(y_true, y_pred)

weight_gradients, bias_gradient = gradients(features, y_true, y_pred)

print("Initial prediction:", y_pred)
print("Initial loss:", loss)
print("Weight gradients:", weight_gradients)
print("Bias gradient:", bias_gradient)

for i in range(len(weights)):
    weights[i] = weights[i] - learning_rate * weight_gradients[i]

bias = bias - learning_rate * bias_gradient

new_prediction = predict(features, weights, bias)
new_loss = squared_error(y_true, new_prediction)

print("\nUpdated weights:", weights)
print("Updated bias:", bias)
print("New prediction:", new_prediction)
print("New loss:", new_loss)

for epoch in range(10):
    y_pred = predict(features, weights, bias)
    loss = squared_error(y_true, y_pred)
    weight_gradients, bias_gradient = gradients(features, y_true, y_pred)
    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * weight_gradients[i]
    bias = bias - learning_rate * bias_gradient

    print('\nEpoch:', epoch)
    print(' | Prediction: ', y_pred)
    print(' | Loss: ', loss)
    print(' | Weights: ', weights)
    print(' | Bias: ', bias)
