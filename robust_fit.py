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

# print(ztpi_data)
fig, ax = plt.subplots()
ax.set_title('Fourier transform depicting the frequency components')
ax.plot(t3, A3)

plt.show()
