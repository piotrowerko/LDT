import numpy as np
from scipy.signal import hilbert, butter, lfilter, freqz
from scipy.optimize import least_squares

from . signal import Signal


class MinuteSignal(Signal):
    def __init__(self, name):
        self.name = name
        self.x = np.array([1.0, 1.0, 1.0])
        self.time = np.zeros(shape=(1), dtype=np.float64)  # zminimalizowanie wymiarów wartości domyślnej atrybutu np. array z 1,1 do 1
        self.Amplitude = np.zeros(shape=(1), dtype=np.float64)
        self.amplitude_envelope = np.zeros(shape=(1), dtype=np.float64)  # dodanie atrybutu

    def __str__(self):
        return f'{self.name}'

    def load_data_in_minutes(self, filePath):
        with open(filePath) as file_object:
            data = file_object.readlines()
        for i in range(len(data)):
            data[i] = data[i].split()
            data[i][0] = float(data[i][0][0:2]) * 60 + float((data[i][0][3:]))
            for j in range(1, len(data[i]), 1):
                data[i][j] = float(data[i][j])
        data = np.array(data)
        t = data[:, 0]
        A = data[:, 1]
        #A2 = A - np.mean(A)  # korzystam z numpaja, skoro i tak musi być
        # loc_max_A2_begin_ind = np.argmax(A2[:int(0.2 * len(t))])
        # loc_max_A2_end_ind = np.argmax(A2[int(0.3 * len(t)):]) + len(A2[:int(0.3 * len(t))])
        loc_max_A2_begin_ind = 4000
        loc_max_A2_end_ind = 10000
        t2 = t[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
        self.time = t2 - t2[0]
        self.time_cutted = self.time[int(0.02 * len(self.time)):int(0.90 * len(self.time))] # docinka 
        A_trimed = A[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
        A2 = A - np.mean(A_trimed)
        self.Amplitude_ = A2[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
        