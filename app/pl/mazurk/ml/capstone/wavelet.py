import numpy as np

def upsample(data, i):
    n = 2**i
    size = n*len(data)
    refined = np.zeros(size)
    for j in range(size):
        index = int(j / n)
        reminder = j % n
        if reminder == 0:
            refined[j] = data[index]
        elif index < len(data) - 1:
            y3 = data[index + 1]
            y1 = data[index]
            x3 = n
            x1 = 0
            y2 = ((y3-y1)/(x3-x1))*(reminder - x1) + y1
            refined[j] = y2
        else:
            refined[j] = data[index]

    return list(map(lambda x: x/i, refined))

