import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from keras.callbacks import EarlyStopping

import app.pl.mazurk.ml.capstone.data as data
import app.pl.mazurk.ml.capstone.series as series
import app.pl.mazurk.ml.capstone.vis as vis
# import app.pl.mazurk.ml.capstone.statistics as stats
import app.pl.mazurk.ml.capstone.rnn as rnn
import app.pl.mazurk.ml.capstone.wavelet as wavelet

WAVELET = 'haar'

vis.init()

# Fetching
# =====================================
print("Fetching data")
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2017, 7, 1)

# tickers = ['GS', 'GOOGL', 'AAPL', 'FB', 'ORCL', 'INTC', 'CSCO', 'IBM', 'NVDA', 'QCOM', 'CRM', 'ADBE']
# tickers = ['GS', 'GOOGL', 'AAPL', 'FB', 'ORCL', 'INTC', 'CSCO', 'IBM']
tickers = ['GOOGL']
# frames = [data.get_ticker(d, start, end)[['Close']] for d in tickers]

# Dataset preparation
# =====================================
print("Dataset preparation")
# from functools import reduce
# dataset = reduce(lambda x, y: x.append(y, ignore_index=True), frames)
original = data.get_ticker('GS', start, end)['Close'].reset_index(drop=True)
wp = pywt.WaveletPacket(data=original, wavelet=WAVELET)

original.plot(label="Original")
averageDataset = wavelet.upsample(wp['aaaa'].data, 4)
plt.plot(averageDataset, label="Average upsampled")
plt.plot(wavelet.upsample(wp['aaad'].data, 4), label="Detail upsampled")
plt.legend()

dataset = pd.DataFrame({'0': wp['a'].data, '1': wp['d'].data})
print(dataset.head())

# Baseline
# =====================================
# rmse = score.calculate_baseline(dataset)
# print('Baseline RMSE: %.3f' % rmse)


# Data preparation for supervised learning
# =====================================
diff_lag = 1
y_length = 1
sequence_length = 15
prediction_length = 4
epochs = 150
features_count = 2
# the features/X (without label)
window_size = sequence_length - y_length

print("Transforming data")
transform = data.DataTransformation(sequence_length, diff_lag, y_length)
dataset_examples = transform.transform(dataset)


print("Splitting trin/test")
train_test_ratio = 0.8
train_size = int(len(dataset_examples) * train_test_ratio)

X_train, X_test, y_train, y_test = series.split_train_test(dataset_examples, train_test_ratio, y_length)
X_train2 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(dataset.columns)))
X_test2 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(dataset.columns)))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[2]))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[2]))

print("X_train2.shape = " + str(X_train2.shape))
print("y_train.shape  = " + str(y_train.shape))

model = rnn.create_model(features_count, window_size, y_length)
rnn.print_summary(model)
print("Fitting the model")
history = model.fit(X_train2, y_train, batch_size=2000, epochs=epochs, validation_split=0.1,
                    shuffle=False)
                    # callbacks=[EarlyStopping('val_loss', verbose=0, patience=8)])

print("Plotting history")
vis.plot_history(history)

print("Predicting")
predictions = rnn.predict_sequences_multiple(model, X_test2, window_size, prediction_length)

print("Inverse transform")
true_data = dataset.reset_index(drop=True).values
test_true_data = true_data[-(len(X_test) + sequence_length):-1]
test_initial_values = test_true_data[0:-1:prediction_length]
real_predictions = transform.inverse_transform(np.array(predictions), X_test[0:-1:prediction_length], test_initial_values)


vis.plot_results_multiple(
    np.reshape(real_predictions[:, :, 0], (real_predictions.shape[0], real_predictions.shape[1], 1)),
    np.reshape(test_true_data[:, 0], (test_true_data.shape[0], 1)),
    prediction_length, sequence_length)
vis.plot_results_multiple(
    np.reshape(real_predictions[:, :, 1], (real_predictions.shape[0], real_predictions.shape[1], 1)),
    np.reshape(test_true_data[:, 1], (test_true_data.shape[0], 1)),
    prediction_length, sequence_length)

cA = real_predictions[:, :, 0]
cD = real_predictions[:, :, 1]
joined_predictions = []
for i in range(len(cA)):
    cA_0 = cA[i]
    cD_0 = cD[i]
    # cD_0 = test_true_data[i * prediction_length + sequence_length:i * prediction_length + sequence_length + prediction_length][:, 1]
    inversed = pywt.idwt(cA_0, cD_0, wavelet=WAVELET)
    joined_predictions.append(inversed)
joined_predictions = np.array(joined_predictions)
real_predictions = np.reshape(joined_predictions, (joined_predictions.shape[0], joined_predictions.shape[1], 1))
test_true_data = original[-(len(X_test) + sequence_length)*2:-1]
test_true_data = np.reshape(test_true_data, (test_true_data.shape[0], 1))
prediction_length *= 2
sequence_length *= 2
print("Plotting predictions")
vis.plot_results_multiple(real_predictions, test_true_data, prediction_length, sequence_length)

plt.show(block=True)