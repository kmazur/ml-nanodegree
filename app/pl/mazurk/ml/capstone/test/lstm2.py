import datetime

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping

import app.pl.mazurk.ml.capstone.data as data
import app.pl.mazurk.ml.capstone.series as series
import app.pl.mazurk.ml.capstone.vis as vis
# import app.pl.mazurk.ml.capstone.statistics as stats
import app.pl.mazurk.ml.capstone.rnn as rnn

vis.init()

# Fetching
# =====================================
print("Fetching data")
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 6, 1)

# tickers = ['GS', 'GOOGL', 'AAPL', 'FB', 'ORCL', 'INTC', 'CSCO', 'IBM', 'NVDA', 'QCOM', 'CRM', 'ADBE']
# tickers = ['GS', 'GOOGL', 'AAPL', 'FB', 'ORCL', 'INTC', 'CSCO', 'IBM']
tickers = ['GS']
frames = [data.get_ticker(d, start, end)[['Close']] for d in tickers]

# Dataset preparation
# =====================================
print("Dataset preparation")
from functools import reduce
dataset = reduce(lambda x, y: x.append(y, ignore_index=True), frames)


# Baseline
# =====================================
# rmse = score.calculate_baseline(dataset)
# print('Baseline RMSE: %.3f' % rmse)


# Data preparation for supervised learning
# =====================================
diff_lag = 1
y_length = 2
sequence_length = 30
prediction_length = 10
epochs = 1
features_count = len(dataset.columns)

print("Transforming data")
transform = data.DataTransformation(sequence_length, diff_lag, y_length)
dataset_examples = transform.transform(dataset)


print("Splitting trin/test")
train_test_ratio = 0.9
train_size = int(len(dataset_examples) * train_test_ratio)

X_train, X_test, y_train, y_test = series.split_train_test(dataset_examples, train_test_ratio, y_length)
X_train2 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(dataset.columns)))
X_test2 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(dataset.columns)))

model = rnn.create_model(features_count, y_length)
rnn.print_summary(model)
print("Fitting the model")
history = model.fit(X_train2, y_train, batch_size=2000, epochs=epochs, validation_split=0.1,
                    shuffle=False,
                    callbacks=[EarlyStopping('val_loss', verbose=0, patience=4)])

print("Plotting history")
vis.plot_history(history)

print("Predicting")
# the features/X (without label)
window_size = sequence_length - 1
predictions = rnn.predict_sequences_multiple(model, X_test2, window_size, prediction_length)

print("Inverse transform")
true_data = dataset.reset_index(drop=True).values
test_true_data = true_data[-(len(X_test) + sequence_length):-1]
test_initial_values = test_true_data[0:-1:prediction_length]
real_predictions = transform.inverse_transform(np.array(predictions), X_test[0:-1:prediction_length], test_initial_values)

print("Plotting predictions")
vis.plot_results_multiple(real_predictions, test_true_data, prediction_length, sequence_length)
