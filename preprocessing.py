import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import  lfilter
import math
from wfdb import processing
from helpers import readfile , butter_bandpass,zscorenormalization

fs= 360
lowcutfreq= 0.5
highcutfreq=40
b, a = butter_bandpass(lowcutfreq, highcutfreq, fs, order=5)

# normal train set 
normal_train=readfile("dataset/Normal_Train.txt")
filtered_normal_train = lfilter(b, a, normal_train)
normTrainSignals=zscorenormalization(filtered_normal_train)

# normal test set
normal_test=readfile("dataset/Normal_Test.txt")
filtered_normal_test = lfilter(b, a, normal_test)
normTestSignals=zscorenormalization(filtered_normal_test)

# pvc train set
pvc_train=readfile("dataset/PVC_Train.txt")
filtered_pvc_train = lfilter(b, a, pvc_train)
normPvcTrain=zscorenormalization(filtered_pvc_train)

# pvc test set
pvc_test=readfile("dataset/PVC_Test.txt")
filtered_pvc_test = lfilter(b, a, pvc_test)
normPvcTest=zscorenormalization(filtered_pvc_test)

# train data
X_train = np.vstack([normTrainSignals, normPvcTrain])
y_train = np.array([0]*len(normTrainSignals) + [1]*len(normPvcTrain))

# test data
X_test = np.vstack([normTestSignals, normPvcTest])
y_test = np.array([0]*len(normTestSignals) + [1]*len(normPvcTest))

# to use i gui & be predected in model
def input_signal_preprocessing(inputsig):
    inputsignal=readfile(inputsig)
    filtered_signal = lfilter(b, a, inputsignal)
    normalizedSignal=zscorenormalization(filtered_signal)
    return normalizedSignal