import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd


img = plt.imread('tree.jpg')[:, :, 0] / 255.
sh = np.shape(img)

factor = 5
img_sub = img[::factor, ::factor]
img_sub = np.mean(np.reshape(
    img, (sh[0]//factor, factor, sh[1]//factor, factor)), axis=(1, 3))


img_up = np.zeros(sh)
img_up[factor//2::factor, factor//2::factor] = img_sub


kernel_nearest = np.ones((factor, factor))

img_nearest = nd.convolve(img_up, kernel_nearest, mode='wrap')


kernel_rect = np.zeros((2*factor - factor % 2, 2*factor - factor % 2))
kernel_rect[factor//2:3*factor//2, factor//2:3*factor//2] = 1
kernel_linear = nd.convolve(kernel_rect, kernel_rect)
kernel_linear /= factor**2  # normalization
img_linear = nd.convolve(img_up, kernel_linear, mode='wrap')


kernel_sinc = np.zeros(sh)
w = sh[0]//2//factor
kernel_sinc[sh[0]//2-w:sh[0]//2+w, sh[1]//2-w:sh[1]//2+w] = 1
kernel_sinc = np.fft.ifftshift(kernel_sinc)
img_sinc = np.real(np.fft.ifft2(np.fft.fft2(img_up) * kernel_sinc))


plt.figure(1)
plt.subplot(2, 3, 1)
plt.title('original')
plt.imshow(img, cmap='gray', interpolation='none')
plt.subplot(2, 3, 2)
plt.imshow(img_sub, cmap='gray', interpolation='none')
plt.title('downsampled')
plt.subplot(2, 3, 3)
plt.imshow(img_up, cmap='gray', interpolation='none')
plt.title('upsampled again')
plt.subplot(2, 3, 4)
plt.imshow(img_nearest, cmap='gray', interpolation='none')
plt.title('nearest interpolated')
plt.subplot(2, 3, 5)
plt.imshow(img_linear, cmap='gray', interpolation='none')
plt.title('linear interpolated')
plt.subplot(2, 3, 6)
plt.imshow(img_sinc, cmap='gray', interpolation='none')
plt.title('sinc interpolated')
