def f_linear(x):
    return 3 * x + 2


def derivative_linear(x):
    return 3


def f_square(x):
    return x ** 2


def derivative_square(x):
    return 2 * x


def loss_function(w):
    return (w - 3) ** 2


def derivative_loss(w):
    return 2 * (w - 3)


print("Linear function:")
for x in [0, 1, 2, 3]:
    print(f"x = {x} | f(x) = {f_linear(x)} | f'(x) = {derivative_linear(x)}")

print("\nSquare function:")
for x in [0, 1, 2, 3]:
    print(f"x = {x} | f(x) = {f_square(x)} | f'(x) = {derivative_square(x)}")

print("\nLoss function:")
for w in [0, 1, 2, 3, 4, 5]:
    print(f"w = {w} | loss = {loss_function(w)} | loss'(w) = {derivative_loss(w)}")


print('Gradient: ', derivative_loss(5))
print('new_w = ', 5 - (0.1 * derivative_loss(5))) 