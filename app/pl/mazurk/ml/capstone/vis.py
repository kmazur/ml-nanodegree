import matplotlib.pyplot as plt
import numpy as np

def init() -> None:
    plt.ion()


def plot_history(history):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_length, input_length):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    paddingFeatures = [None for _ in range(true_data.shape[1])]
    for i, data in enumerate(predicted_data):
        padding = [paddingFeatures for _ in range(i * prediction_length + input_length)]
        if len(padding) > 0:
            padding[-1] = true_data.iloc[len(padding) - 1].values
        else:
            padding = np.zeros((0, data.shape[1]))
        prediction_plot = np.concatenate((padding, data))
        plt.plot(prediction_plot, label='Prediction')
        #        for feature_index in range(prediction_plot.shape[1]):
        plt.legend()
    plt.show(block=False)
