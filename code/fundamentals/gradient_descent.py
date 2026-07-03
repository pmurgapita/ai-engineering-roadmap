def loss_function(w):
    return (w - 3) ** 2


def gradient(w):
    return 2 * (w - 3)


w = 0
learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    current_loss = loss_function(w)
    current_gradient = gradient(w)

    print(
        f"Epoch {epoch:02d} | "
        f"w = {w:.4f} | "
        f"loss = {current_loss:.4f} | "
        f"gradient = {current_gradient:.4f}"
    )

    if current_loss < 0.0001:
        print("Early stopping: loss is small enough.")
        break

    w = w - learning_rate * current_gradient

final_loss = loss_function(w)

print(f"Final weight: {w:.4f}")
print(f"Final loss: {final_loss:.4f}")