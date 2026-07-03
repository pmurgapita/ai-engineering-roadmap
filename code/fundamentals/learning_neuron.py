def predict(x, w):
    return x * w


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def gradient(x, y_true, y_pred):
    return -2 * x * (y_true - y_pred)


xs = [1, 2, 3, 4]
ys = [3, 6, 9, 12]

w = 0.0
learning_rate = 0.01
epochs = 50

for epoch in range(epochs):
    total_loss = 0
    total_gradient = 0

    for x, y_true in zip(xs, ys):
        y_pred = predict(x, w)

        loss = squared_error(y_true, y_pred)
        grad = gradient(x, y_true, y_pred)

        total_loss += loss
        total_gradient += grad

    mean_loss = total_loss / len(xs)
    mean_gradient = total_gradient / len(xs)

    print(
        f"Epoch {epoch:02d} | "
        f"w = {w:.4f} | "
        f"loss = {mean_loss:.4f} | "
        f"gradient = {mean_gradient:.4f}"
    )

    w = w - learning_rate * mean_gradient

print(f"Final weight: {w:.4f}")