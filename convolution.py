import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

tree = plt.imread('tree.jpg')

img = tree[:, :, 0] / 255. 

plt.figure(1)
plt.imshow(img, cmap='gray')
plt.colorbar()
kernel = .2 * np.eye(5)
result_function = nd.convolve(img, kernel, mode='wrap')

plt.figure(2)
plt.imshow(result_function, cmap='gray')
plt.colorbar()
result_explicit = np.zeros_like(img)
h, w = img.shape
kh, kw = kernel.shape
for y in range(h):
    for x in range(w):
        val = 0.
        for j in range(kh):
            for i in range(kw):
                imageY = (y + kh // 2 - j) % h
                imageX = (x + kw // 2 - i) % w
                val += img[imageY, imageX] * kernel[j, i]
        result_explicit[y, x] = val

plt.figure(3)
plt.imshow(result_explicit, cmap='gray')
plt.colorbar()

img_ft = np.fft.fft2(img)

kernel_pad = np.zeros_like(img, dtype=float)
kernel_pad[h//2-kh//2:h//2+kh//2+1, w//2-kw//2:w//2+kw//2+1] = kernel
kernel_pad = np.fft.ifftshift(kernel_pad)
kernel_ft = np.fft.fft2(kernel_pad)
result_fourier = np.real(np.fft.ifft2(img_ft * kernel_ft))

plt.figure(4)
plt.imshow(result_fourier, cmap='gray')
plt.colorbar()
