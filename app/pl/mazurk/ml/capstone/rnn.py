from keras.layers import Activation, GaussianDropout, TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
import numpy as np


def create_model(features_count, y_length):
    model = Sequential()
    model.add(TimeDistributed(
        LSTM(input_shape=(None, features_count), units=100, return_sequences=True),
        input_shape=(2000, features_count, y_length))
    )
    model.add(GaussianDropout(0.02))
    model.add(LSTM(input_shape=(None, features_count), units=100, return_sequences=False))
    model.add(GaussianDropout(0.02))
    model.add(Dense(units=200))
    model.add(GaussianDropout(0.1))
    model.add(Dense(units=200))
    model.add(GaussianDropout(0.1))
    model.add(Dense(units=200))
    model.add(GaussianDropout(0.1))
    model.add(Dense(units=features_count))
    model.add(Activation("tanh"))
    model.compile(loss="mean_squared_logarithmic_error", optimizer='adam', metrics=['accuracy'])
    return model

def print_summary(model):
    model.summary()
    print("Inputs(batch dimension, timesteps, features):                        {}".format(model.input_shape))
    print("ret_seq=True:  Outputs(batch dimension, timesteps, output features): {}".format(model.output_shape))
    print("ret_seq=False: Outputs(batch dimension, output features):            {}".format(model.output_shape))
    # print("Actual input:                                                        {}".format(input.shape))
    # print("Actual output:                                                       {}".format(target.shape))


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
    return prediction_seqs

