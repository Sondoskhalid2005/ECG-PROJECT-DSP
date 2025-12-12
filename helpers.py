import numpy as np
from scipy.signal import butter

def readfile(file):
    if file is None:
        raise ValueError("readfile() got None — no file provided.")

    lines = []
    if isinstance(file, str):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
    else:
        content = file.getvalue().decode("utf-8")
        lines = content.splitlines()

    signals = []
    for line in lines:
        line = line.strip()
        values = line.split('|')                
        values = [float(v) for v in values if v] 
        signals.append(values)

    return np.array(signals, dtype=float)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def zscorenormalization(data):
    mean = np.mean(data, axis=1, keepdims=True)  
    std_dev = np.std(data, axis=1, keepdims=True)  
    std_dev = np.where(std_dev == 0, 1, std_dev) 
    z_scores = (data - mean) / std_dev
    return z_scores