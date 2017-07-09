from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential


def create_model():
    model = Sequential()
    model.add(LSTM(input_shape=(None, 2), units=100, return_sequences=True))
    model.add(LSTM(input_shape=(None, 2), units=100, return_sequences=False))
    model.add(Dense(units=500))
    model.add(Dense(units=500))
    model.add(Dense(units=2))
    model.add(Activation("tanh"))
    model.compile(loss="mean_squared_logarithmic_error", optimizer='adam', metrics=['accuracy'])
    return model
