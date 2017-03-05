# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
DATA_FILE = 'data.csv'
SEED = 7
np.random.seed(SEED)


def read_data():
    import os.path
    if not os.path.isfile(DATA_FILE):
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    else:
        print("Data is cached")
    return np.loadtxt(fname=DATA_FILE, delimiter=',')


dataset = read_data()
X = dataset[:, 0:8]     # there are 8 features  (n, 8)
Y = dataset[:, 8]       # last is label         (n)


model = Sequential([
    Dense(12, input_dim=8, init='uniform', activation='relu'),
    Dense(8, init='uniform', activation='relu'),
    Dense(1, init='uniform', activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, batch_size=10, nb_epoch=150)

scores = model.evaluate(X, Y)
print("\n")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(Y)
print(rounded)
