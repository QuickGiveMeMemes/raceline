y = 0
t = 1


for i in range(20):
    y += 0.05 * 3 * t**2 / (3 * y ** 2 - 4)
    t += 0.05
    print(t, y)

