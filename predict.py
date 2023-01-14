import numpy as np
import load

labels_map = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 
    6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M",
    12: "N", 13: "O", 14: "P", 15: "Q", 16: "R", 17: "S",
    18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y",
}


def cnn_argmax(prediction):
    pred = prediction.cpu()
    pred = pred.detach().numpy()        # converts to <class 'numpy.ndarray'>
    maxIdx = np.argmax(pred, axis=1)    # returns index of the max value
    score = np.amax(np.exp(pred)/np.sum(np.exp(pred)))  # returns exponentiated normailized max value
    #score = np.amax(prediction)        # returns max value
    label = labels_map[maxIdx[0]]       # returns the label
    return label, score


def bcnn_argmax(prediction):
    pred = prediction.cpu()
    pred = pred.detach().numpy()        # converts to <class 'numpy.ndarray'>
    maxIdx = np.argmax(pred, axis=1)    # returns index of the max value
    score = np.amax(np.exp(pred))       # returns exponentiated max value
    #score = np.amax(prediction)        # returns max value
    label = labels_map[maxIdx[0]]       # returns the label
    return label, score


def CNN(image_data):
    cnn = load.CNN()                                # load pre-trained model
    prediction = cnn(image_data)                    # make prediction using input image
    label, score = cnn_argmax(prediction=prediction)    # get the most likely class
    return label, score

def BCNN(image_data, nsamples):
    bcnn = load.BCNN()                                              # load pre-trained model
    prediction = bcnn.predict(image_data, num_predictions=nsamples) # make prediction using input image
    label, score = bcnn_argmax(prediction)                               # get the most likely class
    return label, score