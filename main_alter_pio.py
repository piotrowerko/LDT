import statistics
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

ztpi_data = np.loadtxt('ztpi_data.txt', dtype=np.float64)
t = ztpi_data[:, 0]
A = ztpi_data[:, 1]
A2 = A - statistics.mean(A)
loc_max_A2_begin_ind = np.argmax(A2[:int(0.1 * len(t))])
loc_max_A2_end_ind = np.argmax(A2[int(0.9 * len(t)):]) + len(A2[:int(0.9 * len(t))])
t2 = t[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
t3 = t2 - t2[0]
A3 = A2[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
amplitude_envelope = np.abs(hilbert(A3))


def damp_func(x, a, b, c):
    """formula for approximattion of upper hilbert envelope"""
    return a * np.exp(-b * x) + c


def fun(x, t, y):
    return x[0] * np.exp(-x[1] * t) + x[2] - y


x0 = np.array([1.0, 1.0, 1.0])

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

# Standardowa metoda najmniejszych kawdartow
res_lsq = least_squares(fun, x0, args=(t3, amplitude_envelope))
# Funcja tłumienia soft l1
res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t3, amplitude_envelope))
# Funcja tłumienia huber
res_hub = least_squares(fun, x0, loss='huber', f_scale=0.1, args=(t3, amplitude_envelope))

param, param_conv = curve_fit(damp_func, t3, amplitude_envelope)
print('Rozwiązanie Curve Fit Piotra')
print(f'Curve Fit a: {param[0]:.3}, b: {param[1]:.3} c: {param[2]:.3}')
print('To samo inna biblioteka - zwykle matoda najmniejszych kawdratow - wynik ten sam')
print(f'LSF a: {res_lsq.x[0]:.3}, b: {res_lsq.x[1]:.3} c: {res_lsq.x[2]:.3}')
print('Przyklad funkcji odpornej 1')
print(f'Soft L1 a: {res_soft_l1.x[0]:.3}, b: {res_soft_l1.x[1]:.3} c: {res_soft_l1.x[2]:.3}')
print('Przyklad funkcji odpornej 2')
print(f'Huber a: {res_hub.x[0]:.3}, b: {res_hub.x[1]:.3} c: {res_hub.x[2]:.3}')


def gen_data(t, a, b, c):
    return a * np.exp(-b * t) + c


y_lsq = gen_data(t3, *res_lsq.x)
y_soft_l1 = gen_data(t3, *res_soft_l1.x)
y_huber = gen_data(t3, *res_hub.x)

plt.plot(t3, A3, 'o')
plt.plot(t3, y_lsq, label='linear loss')
plt.plot(t3, y_soft_l1, label='soft_l1 loss')
plt.plot(t3, y_huber, label='huber loss')
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()

# fig, ax = plt.subplots()
# ax.plot(t3, A3)
# ax.plot(t3, amplitude_envelope)
# ax.plot(t3, damp_fun_points, label='curve_fit')
# # plt.text(6, 0.12, r'a * exp(-Bt) + c', {'color': 'C2'})
# # plt.text(6, 0.10, f'a: {np.round(param[0], 4)},  b: {np.round(param[1], 4)},  c: {np.round(param[2], 5)}',
# #          {'color': 'C2'})
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Rel. vert. displacement [mm]')
# plt.show()

# Frequency domain representation
fourierTransform = np.fft.fft(A3) / len(A3)  # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(A3) / 2))]  # Exclude sampling frequency
tpCount = len(A3)
values = np.arange(int(tpCount / 2))
timePeriod = tpCount / 200
frequencies = values / timePeriod

# Frequency domain representation - plot
fig, ax = plt.subplots()
ax.set_title('Fourier transform depicting the frequency components')
ax.plot(frequencies, abs(fourierTransform))
ax.set_xlabel('Frequency')
ax.set_ylabel('Amplitude')
plt.show()
