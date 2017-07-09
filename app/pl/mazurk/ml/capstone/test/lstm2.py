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
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 6, 1)
df = data.get_ticker('GS', start, end)

# Dataset preparation
# =====================================
# dataset = data.dataframe_to_ndarray(pd.rolling_mean(df, 4), ['Close'])
# dataset = np.reshape(dataset, len(dataset))  # 1-D series
df['Change'] = df['High'] - df['Low']
dataset = df[['Close', 'Change']]


# Baseline
# =====================================
# rmse = score.calculate_baseline(dataset)
# print('Baseline RMSE: %.3f' % rmse)


# Data preparation for supervised learning
# =====================================
diff_lag = 1
sequence_length = 30
prediction_length = 10
epochs = 50

transform = data.DataTransformation(sequence_length, diff_lag)
dataset_examples = transform.transform(dataset)


def predict_sequences_multiple(model, data, window_size, prediction_length) -> list:
    prediction_seqs = []
    for i in range(int(len(data) / prediction_length)):
        curr_frame = data[i * prediction_length]
        predicted = []
        for j in range(prediction_length):
            example = curr_frame[np.newaxis, :, :]
            prediction = model.predict(example)
            predicted.append(prediction[0])
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


def plot_results_multiple(predicted_data, true_data, prediction_length, input_length):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data[:, 0], label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    paddingFeatures = [None for _ in range(true_data.shape[1])]
    for i, data in enumerate(predicted_data):
        padding = [paddingFeatures for _ in range(i * prediction_length + input_length)]
        if len(padding) > 0:
            padding[-1] = true_data[len(padding) - 1]
        else:
            padding = np.zeros((0, data.shape[1]))
        prediction_plot = np.concatenate((padding, data))
        plt.plot(prediction_plot[:, 0], label='Prediction')
#        for feature_index in range(prediction_plot.shape[1]):
        plt.legend()
    plt.show(block=True)


#
# # Scaling data
# # =====================================
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = scaler.fit(train_diff)
# train_scaled = scaler.transform(train_diff)
# train_unscaled = scaler.inverse_transform(train_scaled)

train_test_ratio = 0.9
train_size = int(len(dataset_examples) * train_test_ratio)

X_train, X_test, y_train, y_test = series.split_train_test(dataset_examples, train_test_ratio)
X_train2 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(dataset.columns)))
X_test2 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(dataset.columns)))

model = rnn.create_model()
history = model.fit(X_train2, y_train, batch_size=train_size, epochs=epochs, validation_split=0.2,
                    shuffle=False,
                    callbacks=[EarlyStopping('val_loss', verbose=0, patience=4)])

vis.plot_history(history)

# the features/X (without label)
window_size = sequence_length - 1
predictions = predict_sequences_multiple(model, X_test2, window_size, prediction_length)

true_data = dataset.reset_index(drop=True).values
test_true_data = true_data[-(len(X_test) + sequence_length):-1]  # TODO: didn't we lose some initial values?
test_initial_values = test_true_data[0:-1:prediction_length]
real_predictions = transform.inverse_transform(np.array(predictions), X_test[0:-1:prediction_length], test_initial_values)

plot_results_multiple(real_predictions, test_true_data, prediction_length, sequence_length)
