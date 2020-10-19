from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math


def plot_timeseriaes(t, A):
    fig, ax = plt.subplots()
    ax.plot(t, A)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    plt.show()


def fft_calc(fs, A):
    fourierTransform = np.fft.fft(A) / len(A)  # Normalize amplitude
    fourierTransform = fourierTransform[range(int(len(A) / 2))]  # Exclude sampling frequency
    tpCount = len(A)
    values = np.arange(int(tpCount / 2))
    timePeriod = tpCount / fs
    frequencies = values / timePeriod
    return frequencies, fourierTransform

def plot_fft(frequencies, fourierTransform):
    # Frequency domain representation
    fig, ax = plt.subplots()
    ax.set_title('Fourier transform depicting the frequency components')
    ax.plot(frequencies, abs(fourierTransform))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')
    plt.show()

if __name__ == '__main__':


    t = []
    A = []

    tm = []
    Am = []

    with open('murckowskaW.txt', 'r') as f:
        content = f.readlines()
        for x in content:
            row = x.split()
            t.append(float(row[0]))
            A.append(float(row[1]))

    with open('data_4_rbin53_ZTPI.txt', 'r') as f:
        content = f.readlines()
        for x in content:
            row = x.split()
            tm.append(float(row[0]))
            Am.append(float(row[1]))
    # print(t, A)

    for index, item in enumerate(t):
        if item > 302 and item < 303:
            print(index, item)

    Amex = Am[5388:7151]
    tmp = statistics.mean(Amex)

    Aex = [x - tmp for x in Amex]
    tex = tm[5388:7151]

    Amex2 = A[56550:60425]
    tex2 = t[56550:60425]
    tmp = statistics.mean(Amex2)

    Aex2 = [x -tmp for x in Amex2]


    print(len(Aex2))

    plot_timeseriaes(t, A)
    plot_timeseriaes(tm, Am)
    plot_timeseriaes(tex, Aex)
    plot_timeseriaes(tex2, Aex2)
    frequencies, fourierTransform = fft_calc(200, Aex)
    plot_fft(frequencies, fourierTransform)

    frequencies, fourierTransform = fft_calc(200, Aex2)
    plot_fft(frequencies, fourierTransform)

    analytical_signal = hilbert(Aex)
    amplitude_envelope = np.abs(analytical_signal)

    fig, ax = plt.subplots()
    ax.plot(tex, Aex, label='signal')
    ax.plot(tex, amplitude_envelope, label='envelope')
    plt.show()

    analytical_signal = hilbert(Aex2)
    amplitude_envelope = np.abs(analytical_signal)

    fig, ax = plt.subplots()
    ax.plot(tex2, Aex2, label='signal')
    ax.plot(tex2, amplitude_envelope, label='envelope')
    plt.show()






