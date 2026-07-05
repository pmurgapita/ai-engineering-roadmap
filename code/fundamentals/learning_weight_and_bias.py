def predict(x, w, b):
    return x * w + b


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def gradients(x, y_true, y_pred):
    error = y_true - y_pred

    gradient_w = -2 * x * error
    gradient_b = -2 * error

    return gradient_w, gradient_b


xs = [1, 2, 3, 4]
ys = [1, 4, 7, 10]

w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    total_gradient_w = 0
    total_gradient_b = 0

    for x, y_true in zip(xs, ys):
        y_pred = predict(x, w, b)

        loss = squared_error(y_true, y_pred)
        gradient_w, gradient_b = gradients(x, y_true, y_pred)

        total_loss += loss
        total_gradient_w += gradient_w
        total_gradient_b += gradient_b

    mean_loss = total_loss / len(xs)
    mean_gradient_w = total_gradient_w / len(xs)
    mean_gradient_b = total_gradient_b / len(xs)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"w = {w:.4f} | "
            f"b = {b:.4f} | "
            f"loss = {mean_loss:.4f} | "
            f"grad_w = {mean_gradient_w:.4f} | "
            f"grad_b = {mean_gradient_b:.4f}"
        )

    w = w - learning_rate * mean_gradient_w
    b = b - learning_rate * mean_gradient_b

print(f"Final weight: {w:.4f}")
print(f"Final bias: {b:.4f}")
print(f"Final loss: {mean_loss:.4f}")