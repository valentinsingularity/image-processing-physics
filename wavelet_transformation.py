import numpy as np
import matplotlib.pyplot as plt
import IPPtools as IPPT

img = plt.imread('tree.jpg')[:, :, 0] / 255.

compression_level = 0.1


nLevel = 3        # number of decompositions
wavelet = 'haar'  # mother wavelet
mode = 'per'      # zero padding mode

coeffs, (A, H, V, D) = IPPT.dwt_multiscale(
    img, nLevel=nLevel, mode=mode, wavelet=wavelet)

A0 = coeffs[-1][0]
allcoeffs = np.hstack([A0.ravel(), H, V, D])**2

Nzeros = int((1 - compression_level) * len(allcoeffs))
iarg = allcoeffs.argsort()
lowest_power = allcoeffs[iarg[Nzeros]]

newcoeffs = [
    [iCoeffs*(iCoeffs**2 >= lowest_power) for iCoeffs in iLevels]
    for iLevels in coeffs
]

rec = IPPT.idwt_multiscale(newcoeffs, mode=mode, wavelet=wavelet)
power0 = allcoeffs.sum()
power1 = allcoeffs[iarg[Nzeros:]].sum()

print(
    'compression by %3.1f%% leads to %3.1f%% relative error' %
    (100-100*compression_level, 100*(1-power1/power0))
)

plt.figure(1, figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(2, 2, 2)
plt.imshow(rec, cmap='gray')
plt.title('Compression by %3.1f%%' % (100 - 100 * compression_level))
plt.subplot(2, 2, 3)
plt.imshow(IPPT.tile_dwt(coeffs, img.shape)**(1 / 4.), cmap='gray')
plt.title('Wavelet decomposition (gamma 0.25)')
plt.subplot(2, 2, 4)
plt.imshow(IPPT.tile_dwt(newcoeffs, img.shape)**(1 / 4.), cmap='gray')
plt.title('Wavelet thresholded (gamma 0.25)')
plt.show()
