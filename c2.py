import time
import librosa
from dtw import dtw

import librosa.display
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
# from dtaidistance import dtw_ndim
import scipy.io.wavfile as wav
from python_speech_features import mfcc

train_path = "traindata"
test_path = "testdata"
files = [file for file in os.listdir(train_path)]


def predict_result(file_test):
    audio_test, scl1 = librosa.load(file_test)
    mfcc = librosa.feature.mfcc(audio_test, scl1)
    dis = np.inf
    predicted_label = None
    for i in range(len(files)):
        audio_comp, scl_comp = librosa.load(train_path + "/" + files[i])
        mfcc1 = librosa.feature.mfcc(audio_comp, scl_comp)
        distance, _, _, _ = dtw(mfcc.T, mfcc1.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

        if distance < dis:
            dis = distance
            predicted_label = files[i]

    return dis, predicted_label

for file_name in os.listdir(test_path):
    test = test_path + "/" + file_name
    dis,predicted_label = predict_result(test)
    print('distance', dis)
    print(f'The word {file_name} was predicted: ', predicted_label.split('_'))