import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import sys


def roundmask(shape, radius=1):
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    xx, yy = np.meshgrid(x, y)
    return xx**2 + yy**2 < radius**2


def forwardproject(sample, angles):
    sh = np.shape(sample)                
    Nproj = len(angles)                  

    sinogram = np.zeros((Nproj, sh[1]))

    for proj in np.arange(Nproj):  
        sys.stdout.write("\r Simulating:     %03i/%i" % (proj+1, Nproj))
        sys.stdout.flush()
        im_rot = nd.rotate(sample, angles[proj], reshape=False)
        sinogram[proj, :] = np.sum(im_rot, axis=0)
    return sinogram


def filter_sino(sinogram):

    Nproj, Npix = np.shape(sinogram)
    ramp_filter = np.abs(np.fft.fftfreq(Npix))
    sino_ft = np.fft.fft(sinogram, axis=1)
    sino_filtered = np.real(np.fft.ifft(sino_ft * ramp_filter, axis=1))

    return sino_filtered


def backproject(sinogram, angles):

    Nproj, Npix = np.shape(sinogram)
    reconstruction = np.zeros((Npix, Npix))

    for proj in np.arange(Nproj):  
        sys.stdout.write("\r Reconstructing: %03i/%i" % (proj+1, Nproj))
        sys.stdout.flush()

        backprojection = np.tile(sinogram[proj, :], (Npix, 1))
        backprojection /= Npix 
        rotated_backprojection = nd.rotate(backprojection, -angles[proj], reshape=False)

        reconstruction += rotated_backprojection * roundmask((Npix, Npix))

    return reconstruction


sample = plt.imread('Head_CT_scan.jpg')

Nangles = 301
angles = np.linspace(0, 360, Nangles, False) 

sino = forwardproject(sample, angles)

filtered_sino = filter_sino(sino)

reco = backproject(filtered_sino, angles)

plt.figure(1, figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0., cmap='gray', interpolation='none')


Nangles = 301
angles = np.linspace(0, 360, Nangles, False)

sino = forwardproject(sample, angles)
sino[:, 120] = 0

filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(2, figsize=(12, 12))
plt.suptitle('dead pixel')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=sample.min(), vmax=sample.max(),
           cmap='gray', interpolation='none')



Nangles = 301
angles = np.linspace(0, 360, Nangles, False)

sino = forwardproject(sample, angles)
sino = np.append(sino, np.ones((Nangles, 10)), axis=1)

filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(3, figsize=(12, 12))
plt.suptitle('center shift')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0, cmap='gray', interpolation='none')


Nangles = 91
angles = np.linspace(0, 360, Nangles, False)

sino = forwardproject(sample, angles)

filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(4, figsize=(12, 12))
plt.suptitle('undersampling')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0., cmap='gray', interpolation='none')


Nangles = 301
angles = np.linspace(0, 180, Nangles, False)

sino = forwardproject(sample, angles)
sino[:100] = 0

filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(5, figsize=(12, 12))
plt.suptitle('missing projections')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0., cmap='gray', interpolation='none')


Nangles = 301
angles = np.linspace(0, 360, Nangles, False)

sino = forwardproject(sample, angles)

sino += 5000 * np.random.standard_normal(sino.shape)
filtered_sino = filter_sino(sino)
reco = backproject(filtered_sino, angles)

plt.figure(6, figsize=(12, 12))
plt.suptitle('noise')
plt.subplot(2, 2, 1)
plt.imshow(sample, cmap='gray', interpolation='none')
plt.subplot(2, 2, 2)
plt.imshow(sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 3)
plt.imshow(filtered_sino, cmap='gray', interpolation='none')
plt.subplot(2, 2, 4)
plt.imshow(reco, vmin=0, cmap='gray', interpolation='none')