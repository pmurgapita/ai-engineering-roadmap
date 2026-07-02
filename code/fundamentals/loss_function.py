def squared_error(y_true, y_pred):
    error = y_true - y_pred
    return error ** 2


def mean_squared_error(y_true_values, y_pred_values):
    if len(y_true_values) != len(y_pred_values):
        raise ValueError("Las listas deben tener la misma longitud")

    total_loss = 0

    for i in range(len(y_true_values)):
        total_loss += squared_error(y_true_values[i], y_pred_values[i])

    return total_loss / len(y_true_values)


real_price = 360000
predicted_price = 340000

print("Error cuadrático:", squared_error(real_price, predicted_price))


real_values = [100, 200, 300]
predictions = [90, 220, 310]

print("MSE:", mean_squared_error(real_values, predictions))