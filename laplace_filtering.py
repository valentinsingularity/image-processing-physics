import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('venice.jpg')[:, :, 0] / 255.

plt.figure(1)
plt.imshow(img, cmap='gray')
plt.colorbar()

v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])

vv, uu = np.meshgrid(v, u, indexing='ij')
H = -4. * np.pi**2 * (vv**2 + uu**2)

img_ft = np.fft.fft2(img)
img_filtered = np.real(np.fft.ifft2(img_ft * H))

plt.figure(2)
plt.imshow(img_filtered, cmap='gray')
plt.colorbar()
