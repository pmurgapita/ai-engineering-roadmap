def predict_price(meters, rooms, distance):
    return meters * 2500 + rooms * 12000 + distance * (-7000) + 40000

house1 = [100, 3, 7]
house2 = [120, 2, 5]
house4 = [50, 1, 1]

print('Price first house: ', predict_price(house1[0],house1[1],house1[2]))
print('Price second house: ', predict_price(house2[0], house2[1], house2[2]))
print('Price third house: ', predict_price(house4[0], house4[1], house4[2]))