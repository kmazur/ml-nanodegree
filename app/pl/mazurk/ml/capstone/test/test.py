import numpy as np
import pandas as pd

import app.pl.mazurk.ml.capstone.data as data
import app.pl.mazurk.ml.capstone.series as series
import app.pl.mazurk.ml.capstone.score as score
import app.pl.mazurk.ml.capstone.vis as vis

real = np.array([1, 2, 0, 3])

x = series.to_examples(real, 1, 1)
preds = series.from_predictions([3, 1], x, real, 1)
print(x)
print(preds)
