from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential


def create_model():
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=7, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation("tanh"))
    model.compile(loss="mean_squared_logarithmic_error", optimizer='adam', metrics=['accuracy'])
    return model
