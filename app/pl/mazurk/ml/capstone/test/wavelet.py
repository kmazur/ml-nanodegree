import pywt
import numpy as np

import pandas
import math
from keras.models import Sequential
from keras.layers import Dense

import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt

plt.ion()

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()

df = web.DataReader("F", 'google', start, end)

tail = df.tail(min(df.size, 1000))
close = tail['Close']
index = tail.index.values

wp = pywt.WaveletPacket(data=close, wavelet='haar')

def ma(data, window_size):
    result = np.repeat(data[0], len(data))
    sum = 0
    for i in range(0, window_size):
        sum += data[i]
        result[i] = sum/(i+1)

    avg = sum/window_size
    for i in range(window_size, len(data)):
        avg += (data[i] - data[i - window_size])/window_size
        result[i] = avg
    return result

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


depthLevel = 2
maWindow = 5

def plotData(data, label):
    plt.plot(range(len(data)), data, label=label)


plotData(upsample(wp['aa'].data, depthLevel), "A" + str(depthLevel))
plotData(close, "A0")
plotData(ma(close, maWindow)[maWindow:], "MA")
# plotData(tail['Volume']/max(tail['Volume']), "Volume")

# for i in range(depthLevel):
#     packet = wp['a' * i]
#     data = packet.data
#     refined = upsample(data, 2**i)
#     plt.plot(range(len(refined)), refined, label="L"+str(i))
#     refined = upsample(wp[('a' * (i - 1)) + 'd'].data, 2**i)
#     plt.plot(range(len(refined)), refined, label="D"+str(i))


plt.legend()
plt.show(block=True)

