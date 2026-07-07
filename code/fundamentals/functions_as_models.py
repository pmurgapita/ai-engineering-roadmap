def linear_function(x, w, b):
    return x * w + b


def house_price_function(features, weights, bias):
    total = 0

    for i in range(len(features)):
        total += features[i] * weights[i]

    return total + bias


def squared_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def apply_function_to_batch(batch, weights, bias):
    predictions = []

    for features in batch:
        prediction = house_price_function(features, weights, bias)
        predictions.append(prediction)

    return predictions


print("Simple function:")
print(linear_function(3, 2, 1))

features = [90, 3, 4]
weights = [2.5, 12, -7]
bias = 40

prediction = house_price_function(features, weights, bias)

print("\nHouse prediction:")
print(prediction)

loss = squared_loss(273, prediction)

print("\nLoss:")
print(loss)

batch = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
]

batch_predictions = apply_function_to_batch(batch, weights, bias)

print("\nBatch predictions:")
print(batch_predictions)

def mean_squared_error(y_true_values, y_pred_values):
    squared = []
    count = 0
    sum = 0
    for i in range(len(y_true_values)):
        count += 1
        squared.append((y_true_values[i] - y_pred_values[i])**2)
    for i in range(len (squared)):
        sum += squared[i]
    return sum/count

real_values = [139, 241, 376]
predicted_values = [121, 241, 374]

print(mean_squared_error(real_values, predicted_values))