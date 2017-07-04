from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential


def create_model():
    model = Sequential()
    model.add(LSTM(input_shape=(None, 1), units=100, return_sequences=False))
    model.add(Dense(units=1500))
    model.add(Dense(units=1200))
    model.add(Dense(units=1200))
    model.add(Dense(units=200))
    model.add(Dense(units=1))
    model.add(Activation("tanh"))
    model.compile(loss="mean_squared_logarithmic_error", optimizer='adam', metrics=['accuracy'])
    return model
