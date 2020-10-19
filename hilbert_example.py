from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt

sensor = np.loadtxt('signal.txt')
analytical_signal = hilbert(sensor)
amplitude_envelope = np.abs(analytical_signal)

fig, ax = plt.subplots()
# ax.plot(analytical_signal.real)
# ax.plot(analytical_signal.imag)

ax.plot(sensor)
ax.plot(amplitude_envelope)
plt.show()


