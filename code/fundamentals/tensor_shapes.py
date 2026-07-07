def get_shape(data):
    if not isinstance(data, list):
        return ()

    if len(data) == 0:
        return (0,)

    return (len(data),) + get_shape(data[0])


scalar = 5

vector = [1, 2, 3]

matrix = [
    [1, 2, 3],
    [4, 5, 6],
]

tensor_3d = [
    [
        [1, 2],
        [3, 4],
    ],
    [
        [5, 6],
        [7, 8],
    ],
]

image_rgb = [
    [
        [255, 0, 0],
        [0, 255, 0],
    ],
    [
        [0, 0, 255],
        [255, 255, 255],
    ],
]

batch_of_sequences = [
    [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ],
    [
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
    ],
]

print("Scalar shape:", get_shape(scalar))
print("Vector shape:", get_shape(vector))
print("Matrix shape:", get_shape(matrix))
print("Tensor 3D shape:", get_shape(tensor_3d))
print("RGB image shape:", get_shape(image_rgb))
print("Batch of sequences shape:", get_shape(batch_of_sequences))

def count_values(shape):
    total = 1

    for dimension in shape:
        total *= dimension

    return total


print("Values in RGB image:", count_values(get_shape(image_rgb)))
print("Values in batch of sequences:", count_values(get_shape(batch_of_sequences)))

print(count_values((32, 224, 224, 3)))
print(count_values((8, 128, 768)))