import os
import numpy as np
import matplotlib.pyplot as plt


def wrap_phase(inarray):
    outarray = np.angle(np.exp(1j * inarray))
    return outarray

PATH = 'data' + os.sep

NUMIMGS = 11
FILEPATTERN = '%s_stepping_%04d.npy'

imglist = []
flatlist = []
for i in range(NUMIMGS):
    # load the image
    img = np.load(os.path.join(os.getcwd(), PATH, FILEPATTERN % ('data', i)))
    imglist.append(img)
    # load the flatfield
    flat = np.load(os.path.join(os.getcwd(), PATH, FILEPATTERN % ('flat', i)))
    flatlist.append(flat)

imgarr = np.array(imglist)
flatarr = np.array(flatlist)

stepping_curve = imgarr[:, 50, 200]
ref_curve = flatarr[:, 50, 200]

plt.figure(1)
plt.plot(stepping_curve, '*', label='stepping curve')
plt.plot(ref_curve, 'bo', label='reference curve')

point_ft = np.fft.fft(stepping_curve)

cons = np.abs(point_ft[0]) / NUMIMGS
ang = np.angle(point_ft[1])
mod = np.abs(point_ft[1]) / NUMIMGS

x = np.linspace(0, NUMIMGS, 1000)
fit_stepping_curve = cons + 2. * mod * np.cos(x / NUMIMGS * 2 * np.pi + ang)

plt.plot(x, fit_stepping_curve, label='fit')
plt.legend()

# cropping
data_cropped = imgarr[:, :, 72:430]
flatfield_cropped = flatarr[:, :, 72:430]

# fourier processing
data_fft = np.fft.fft(data_cropped, axis=0) / NUMIMGS
flat_fft = np.fft.fft(flatfield_cropped, axis=0) / NUMIMGS

data_absorption = np.abs(data_fft[0])
data_differential_phase = np.angle(data_fft[1])
data_darkfield = np.abs(data_fft[1]) / data_absorption

flatfield_absorption = np.abs(flat_fft[0])
flatfield_differential_phase = np.angle(flat_fft[1])
flatfield_darkfield = np.abs(flat_fft[1]) / flatfield_absorption

#flatfield correction
absorption = -np.log(data_absorption / flatfield_absorption)
differential_phase = wrap_phase(
    data_differential_phase - flatfield_differential_phase)
darkfield = -np.log(data_darkfield / flatfield_darkfield )

plt.figure(2)
plt.subplot(3, 1, 1)
plt.title('absorption')
plt.imshow(absorption, cmap='gray')
plt.subplot(3, 1, 2)
plt.title('differential phase')
plt.imshow(differential_phase, cmap='gray')
plt.subplot(3, 1, 3)
plt.title('darkfield')
plt.imshow(darkfield, cmap='gray')
