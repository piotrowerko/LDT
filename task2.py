from pylab import *
from scipy.signal import *

# Generate AM-modulated sinusoid
N = 256
t = linspace(0,2,N)
# Modulator
m = 2*np.exp(-1.5*t)
# Carrier
c = sin(2*pi*20*t)
# Signal is modulator times carrier
x = m*c
# Calculate envelope, called m_hat via hilbert transform
m_hat = abs(hilbert(x))

with open("example.txt", "w+") as f:
    for n, data in enumerate(x):
        f.write(f'{n} {t[n]:.3f} {x[n]:.3f}\n')


# Plot x
plot(t, x)
plot(t, m_hat)
axis('tight')
xlabel('Time (seconds)')
ylabel('Amplitude')
title('X, with calculated envelope')
legend(['x', 'm_hat'])
ylim(-3,3)
show()