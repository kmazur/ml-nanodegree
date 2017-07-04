import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

import app.pl.mazurk.ml.capstone.data as data
import app.pl.mazurk.ml.capstone.series as series
import app.pl.mazurk.ml.capstone.score as score
import app.pl.mazurk.ml.capstone.vis as vis
# import app.pl.mazurk.ml.capstone.statistics as stats
import app.pl.mazurk.ml.capstone.rnn as rnn

vis.init()

# Fetching
# =====================================
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 6, 1)
df = data.get_ticker('GOOG', start, end)

# Dataset preparation
# =====================================
dataset = data.dataframe_to_ndarray(pd.rolling_mean(df, 4), ['Close'])
dataset = np.reshape(dataset, len(dataset))  # 1-D series

# Baseline
# =====================================
rmse = score.calculate_baseline(dataset)
print('Baseline RMSE: %.3f' % rmse)


# Data preparation for supervised learning
# =====================================
diff_lag = 1
sequence_length = 20
prediction_length = 15
epochs = 20

dataset_examples = series.to_examples(dataset, diff_lag, sequence_length)



def predict_sequences_multiple(model, data, window_size, prediction_len) -> list:
    from numpy import newaxis
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            example = curr_frame[newaxis, :, :]
            prediction = model.predict(example)
            predicted.append(prediction[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)

    # for i in range(len(data)):
    #     curr_frame = data[i]
    #     predicted = []
    #     for j in range(prediction_len):
    #         predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
    #         curr_frame = curr_frame[1:]
    #         curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    #     prediction_seqs.append(predicted)
    #
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len, input_length):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len + input_length)]
        if len(padding) > 0:
            padding[-1] = true_data[len(padding) - 1]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show(block=True)


#
# # Scaling data
# # =====================================
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = scaler.fit(train_diff)
# train_scaled = scaler.transform(train_diff)
# train_unscaled = scaler.inverse_transform(train_scaled)


# dataset_examples = np.array([
#     [1, 0, 1],
#     [0, 1, 0],
#     [1, 0, 1],
#     [0, 1, 0],
#     [1, 0, 1],
#     [0, 1, 0]
# ])
# dataset = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4])

train_test_ratio = 0.8
train_size = int(len(dataset_examples) * train_test_ratio)

X_train, X_test, y_train, y_test = series.split_train_test(dataset_examples, train_test_ratio)
X_train2 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test2 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = rnn.create_model()
import time

start = time.time()
history = model.fit(X_train2, y_train, batch_size=train_size, epochs=epochs, validation_split=0.1)
end = time.time()
print(end - start)
exit()

vis.plot_history(history)

predictions = predict_sequences_multiple(model, X_test2, sequence_length, prediction_length)

# dataset = np.reshape(dataset, len(dataset))
# dataset_diffs = series.diff(dataset, lag=diff_lag)
# dataset_examples = series.timeseries_to_supervised(pd.DataFrame(dataset_diffs), sequence_length)
# dataset_matrix = dataset_examples.as_matrix()


true_data = dataset[train_size:len(dataset)]
real_predictions = series.from_predictions(predictions, X_test, true_data, diff_lag, prediction_length)


plot_results_multiple(real_predictions, true_data, prediction_length, sequence_length + 1)

exit()


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1 - look_forward):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[(i + look_back):(i + look_back + look_forward)])
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
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
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

# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset, label="Original")
plt.plot(trainPredictPlot[:, 0], label="Train prediction")
plt.plot(testPredictPlot, label="Test prediction")
plt.legend()
plt.show(block=True)
