import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd



mean = 0
sigma = 1
M = 100
white_noise = mean + sigma * np.random.randn(M, M)
low_pass = nd.gaussian_filter(white_noise, sigma)
high_pass = white_noise - low_pass


nps_white = np.fft.fftshift(np.abs(np.fft.fft2(white_noise)))**2
nps_low = np.fft.fftshift(np.abs(np.fft.fft2(low_pass)))**2
nps_high = np.fft.fftshift(np.abs(np.fft.fft2(high_pass)))**2


corr_white = np.real(np.fft.fftshift(np.fft.ifft2(
    np.abs(np.fft.fft2(white_noise))**2)))
corr_low = np.real(np.fft.fftshift(np.fft.ifft2(
    np.abs(np.fft.fft2(low_pass))**2)))
corr_high = np.real(np.fft.fftshift(np.fft.ifft2(
    np.abs(np.fft.fft2(high_pass))**2)))


im_shifted1 = plt.imread('worldA.jpg') / 255.
im_shifted2 = plt.imread('worldB.jpg') / 255.
im_shifted1 = im_shifted1.mean(axis=2)
im_shifted2 = im_shifted2.mean(axis=2)


ccorr = np.real(np.fft.ifftshift(np.fft.ifft2(
    np.conj(np.fft.fft2(im_shifted1)) * np.fft.fft2(im_shifted2))))


shift_y, shift_x = np.unravel_index(ccorr.argmax(), ccorr.shape)


print("shift in y = %i" % (shift_y - ccorr.shape[0] // 2))
print("shift in x = %i" % (shift_x - ccorr.shape[1] // 2))



plt.figure(1, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(white_noise, cmap='gray', interpolation='none')
plt.title('white noise spatial domain')
plt.subplot(1, 3, 2)
plt.imshow(low_pass, cmap='gray', interpolation='none')
plt.title('low pass spatial domain')
plt.subplot(1, 3, 3)
plt.imshow(high_pass, cmap='gray', interpolation='none')
plt.title('high pass spatial domain')

plt.figure(2, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(nps_white, cmap='gray', interpolation='none')
plt.title('white noise power spectrum')
plt.subplot(1, 3, 2)
plt.imshow(nps_low, cmap='gray', interpolation='none')
plt.title('low pass power spectrum')
plt.subplot(1, 3, 3)
plt.imshow(nps_high, cmap='gray', interpolation='none')
plt.title('high pass power spectrum')

plt.figure(3, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(corr_white, cmap='gray', interpolation='none')
plt.title('white noise autocorrelation')
plt.subplot(1, 3, 2)
plt.imshow(corr_low, cmap='gray', interpolation='none')
plt.title('low pass noise autocorrelation')
plt.subplot(1, 3, 3)
plt.imshow(corr_high, cmap='gray', interpolation='none')
plt.title('high pass noise autocorrelation')


plt.figure(4, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(im_shifted1, cmap='gray', interpolation='none')
plt.title('image1')
plt.subplot(1, 3, 2)
plt.imshow(im_shifted2, cmap='gray', interpolation='none')
plt.title('image2')
plt.subplot(1, 3, 3)
plt.imshow(ccorr, cmap='gray', interpolation='none')
plt.title('crosscorrelation')
