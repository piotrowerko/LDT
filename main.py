from scipy.signal import hilbert
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math

sensor = np.loadtxt('signal.txt')
analytical_signal = hilbert(sensor)
amplitude_envelope = np.abs(analytical_signal)

fig, ax = plt.subplots()
ax.plot(sensor)
ax.plot(amplitude_envelope)
plt.show()

t = []
A = []
with open('ztpi_data.txt', 'r') as f:
    content = f.readlines()
    for x in content:
        row = x.split()
        t.append(float(row[0]))
        A.append(float(row[1]))

print(statistics.mean(A))
A2 = [x - statistics.mean(A) for x in A]
t2 = t[int(0.05 * len(t)):int(0.95 * len(t))]
A3 = A2[int(0.05 * len(t)):int(0.95 * len(t))]

amplitude_envelope = np.abs(hilbert(A3))

#
def damping(x, a, b, c):
    return a*np.exp(-b*x) +c

param, param_conv = curve_fit(damping, t2, A3)
print(f'a: {param[0]}, b: {param[1]} c: {param[2]}')


fig, ax = plt.subplots()
ax.plot(t, A2)
ax.plot(t2, amplitude_envelope)
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
plt.show()
# Frequency domain representation

fourierTransform = np.fft.fft(A2) / len(A2)  # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(A2) / 2))]  # Exclude sampling frequency
tpCount = len(A2)
values = np.arange(int(tpCount / 2))
timePeriod = tpCount / 200
frequencies = values / timePeriod

# Frequency domain representation
fig, ax = plt.subplots()
ax.set_title('Fourier transform depicting the frequency components')
ax.plot(frequencies, abs(fourierTransform))
ax.set_xlabel('Frequency')
ax.set_ylabel('Amplitude')
plt.show()
