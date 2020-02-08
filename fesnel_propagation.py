import numpy as np
import matplotlib.pyplot as plt

psize = 1e-5  # detector pixelsize
wlen = 6e-7  # wavelength (600nm = visible light)
prop_dist = 3e-3  # propagation distance

img = plt.imread('tum.png')
img = img.sum(axis = 2, dtype = float)

img /= img.max()
w = np.exp(1j * np.pi * img)

plt.figure(1)
plt.imshow(np.angle(w), interpolation='none')
plt.title('Wavefront phase')
plt.colorbar()

N = np.asarray(w.shape)

u = 2. * np.pi * np.fft.fftfreq(img.shape[1], psize)
v = 2. * np.pi * np.fft.fftfreq(img.shape[0], psize)

uu, vv = np.meshgrid(u, v, indexing='xy')
k = 2 * np.pi / wlen

kernel = np.exp(-.5j * prop_dist / k * (uu**2 + vv**2))

out = np.fft.ifft2(np.fft.fft2(w) * kernel)

plt.figure(2)
plt.imshow(np.fft.fftshift(np.angle(kernel)), cmap = 'jet', interpolation='none')
plt.title('Fresnel kernel')
plt.colorbar()

I = np.abs(out)**2

plt.figure(3)
plt.imshow(I[int(img.shape[0]/2-256):int(img.shape[0]/2+256),
             int(img.shape[1]/2-256):int(img.shape[1]/2+256)],
           cmap='gray', interpolation='none')
plt.title('Intensity')
plt.colorbar()
