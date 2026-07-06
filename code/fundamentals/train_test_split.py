def split_train_test(data, targets, test_size):
    if len(data) != len(targets):
        raise ValueError("data y targets deben tener la misma longitud")

    split_index = int(len(data) * (1 - test_size))

    x_train = data[:split_index]
    y_train = targets[:split_index]

    x_test = data[split_index:]
    y_test = targets[split_index:]

    return x_train, y_train, x_test, y_test


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

x_train, y_train, x_test, y_test = split_train_test(
    houses,
    prices,
    test_size=0.4
)

print("X train:")
for row in x_train:
    print(row)

print("\ny train:")
print(y_train)

print("\nX test:")
for row in x_test:
    print(row)

print("\ny test:")
print(y_test)

print("\nTrain examples: ")
print(len(x_train))
print("\nTest examples: ")
print(len(x_test))