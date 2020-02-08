import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd


img = plt.imread('tree.jpg') / 255.
img = img.mean(axis=2)
sh = img.shape
M = 51
psf = np.diag(np.ones(M)) / M
img_conv = nd.convolve(img, psf, mode='wrap')

sigma = .01
img_noisy = img_conv + sigma * np.random.randn(sh[0], sh[1])

psf_pad = np.zeros_like(img)
psf_pad[sh[0]//2-M//2:sh[0]//2+M//2+1, sh[1]//2-M//2:sh[1]//2+M//2+1] = psf

img_deconv = np.real(np.fft.fftshift(np.fft.ifft2(
    np.fft.fft2(img_noisy) / np.fft.fft2(psf_pad))))


def wiener_deconv(img, psf, nps):
    f_psf = np.fft.fft2(psf)
    sps_psf = np.abs(f_psf)**2
    sps = np.abs(np.fft.fft2(img))**2

    wiener_filter = 1. / f_psf * sps_psf / (sps_psf + nps/sps)

    deconv_img = np.fft.fftshift(np.real(np.fft.ifft2(
        np.fft.fft2(img) * wiener_filter)))

    return deconv_img

nps = 1.

img_deconv_W = wiener_deconv(img_noisy, psf_pad, nps)

plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', interpolation='none')
plt.title('original image')
plt.subplot(2, 2, 2)
plt.imshow(img_noisy, cmap='gray', interpolation='none')
plt.title('acquired noisy image')
plt.subplot(2, 2, 3)
plt.imshow(img_deconv, cmap='gray', interpolation='none')
plt.title('naive deconvolution')
plt.subplot(2, 2, 4)
plt.imshow(img_deconv_W, cmap='gray', interpolation='none')
plt.title('Wiener deconvolution')
