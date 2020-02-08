import numpy as np
import matplotlib.pyplot as plt

proj = np.load('proj.npy')

plt.figure()
plt.title('intensity')
plt.imshow(proj, cmap='gray', interpolation='none')
plt.colorbar()

plt.figure()
plt.plot(proj[proj.shape[1]//2])

pixel_size = .964e-6
distance = 8.57e-3

mu = 691.
delta = 2.6e-6

v = 2. * np.pi * np.fft.fftfreq(proj.shape[0], d=pixel_size)
u = 2. * np.pi * np.fft.fftfreq(proj.shape[1], d=pixel_size)
ky, kx = np.meshgrid(v, u, indexing='ij')

Paganin = mu / (distance * delta * (kx**2 + ky**2) + mu)
trace = np.real(
    -1. / mu * np.log(np.fft.ifft2(Paganin * np.fft.fft2(proj))))


plt.figure()
plt.title('trace')
plt.imshow(trace * 1e6, cmap='gray', interpolation='none')
plt.colorbar()

plt.figure()
plt.plot(trace[proj.shape[1]//2] * 1e6)