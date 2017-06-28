import numpy as np
import pandas as pd
from keras.engine import Layer

from keras.layers import Activation, GaussianDropout, Dropout, GaussianNoise, K
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential

import app.pl.mazurk.ml.capstone.data as data
import app.pl.mazurk.ml.capstone.series as series
import app.pl.mazurk.ml.capstone.score as score
import app.pl.mazurk.ml.capstone.vis as vis


vis.init()

return_sequences = False
model = Sequential()
model.add(LSTM(2,                                       # how many features at each time step in example are observed
               input_shape=(3, 2),                      # (sequence_length/timesteps, feature count)
               unroll=True,                             # speed up on CPU
               return_sequences=return_sequences))      # output will be one value with feature count in it e.g: [1, 0]
model.add(GaussianDropout(0.01))
model.add(Dense(2))
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])

all_data = np.array([
    [ [1, 1], [2, 0], [1, 1], [0, 2] ],
    [ [2, 0], [1, 1], [0, 2], [1, 1] ],
    [ [1, 1], [0, 2], [1, 1], [2, 0] ],
    [ [0, 2], [1, 1], [2, 0], [1, 1] ]
], dtype=float)

input = all_data[:, :-1, :]
target = all_data[:, -1, :]
if return_sequences:
    target = np.array([
        [ [2, 0], [1, 1], [0, 2] ],
        [ [1, 1], [0, 2], [1, 1] ],
        [ [0, 2], [1, 1], [2, 0] ],
        [ [1, 1], [2, 0], [1, 1] ]
    ])

model.summary()
print("Inputs(batch dimension, timesteps, features):                        {}".format(model.input_shape))
print("ret_seq=True:  Outputs(batch dimension, timesteps, output features): {}".format(model.output_shape))
print("ret_seq=False: Outputs(batch dimension, output features):            {}".format(model.output_shape))
print("Actual input:                                                        {}".format(input.shape))
print("Actual output:                                                       {}".format(target.shape))

history = model.fit(input, target, nb_epoch=1000, batch_size=1, validation_split=0.1)
vis.plot_history(history)

predictions = model.predict(input)
# predictions =
# [
#
#
#
# ]

print(repr(np.round(predictions)))

import matplotlib.pyplot as plt
plt.show(block=True)

