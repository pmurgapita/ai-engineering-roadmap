def get_columns(data):
    columns = []

    num_features = len(data[0])

    for feature_index in range(num_features):
        column = []

        for row in data:
            column.append(row[feature_index])

        columns.append(column)

    return columns


def min_max_scale(data):
    columns = get_columns(data)

    mins = []
    maxs = []

    for column in columns:
        mins.append(min(column))
        maxs.append(max(column))

    scaled_data = []

    for row in data:
        scaled_row = []

        for i, value in enumerate(row):
            min_value = mins[i]
            max_value = maxs[i]

            scaled_value = (value - min_value) / (max_value - min_value)
            scaled_row.append(scaled_value)

        scaled_data.append(scaled_row)

    return scaled_data, mins, maxs


houses = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

scaled_houses, mins, maxs = min_max_scale(houses)

print("Original houses:")
for house in houses:
    print(house)

print("\nMins:", mins)
print("Maxs:", maxs)

print("\nScaled houses:")
for house in scaled_houses:
    print([round(value, 4) for value in house])


def scale_one(row, mins, maxs):
    return [(row[0] - mins[0]) / (maxs[0] - mins[0]), (row[1] - mins[1]) / (maxs[1] - mins[1]), (row[2] - mins[2]) / (maxs[2] - mins[2])]

house = [90, 3, 4]

print('House scaled: ', scale_one(house, mins, maxs))