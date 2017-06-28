import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pandas as pd

plt.ion()

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2013, 1, 27)

df = web.DataReader("F", 'google', start, end)

tail = df.tail(min(df.size, 1000))
close = tail['Close']
index = tail.index

dataset = close.values.astype('float32')

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
print(len(train), len(test))
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1-look_forward):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[(i + look_back):(i+look_back+look_forward)])
    return np.array(dataX), np.array(dataY)
# reshape into X=t and Y=t+1
look_back = 20
look_forward = 1
trainX, trainY = create_dataset(train, look_back, look_forward)
testX, testY = create_dataset(test, look_back, look_forward)
# create and fit Multilayer Perceptron model
model = Sequential()
# model.add(Embedding(len(dataset), 30, input_length=look_back))
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dropout(0.1))
model.add(Dense(1, activation='relu'))

# model.add(Dense(look_back, input_dim=look_back, activation='relu'))
# model.add(Dense(int(look_back/2), activation='relu'))
# model.add(Dense(look_forward))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=50, batch_size=32, verbose=2)
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = np.zeros((len(dataset), 1))
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.zeros((len(dataset)))
testPredictPlot[:len(trainPredict)] = np.nan

rest = len(trainPredict)
arr = np.copy(dataset[rest - look_back:rest])
while rest < len(dataset):
    example = np.array([arr])
    prediction = model.predict(example)
    value = prediction[0]
    arr[0:len(arr) - 1] = arr[1:len(arr)]
    arr[len(arr) - 1] = value
    testPredictPlot[rest] = value
    rest += 1


#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset, label="Original")
plt.plot(trainPredictPlot[:, 0], label="Train prediction")
plt.plot(testPredictPlot, label="Test prediction")
plt.legend()
plt.show(block=True)
