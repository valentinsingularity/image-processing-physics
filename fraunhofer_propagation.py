import numpy as np
import matplotlib.pylab as plt

N = 1024  # square dimension of phase screen

radius = 128  # radius of the circular aperture in pixels

aperture = np.zeros((N, N))

x = np.linspace(-N/2, N/2, N)
y = np.linspace(-N/2, N/2, N)
xx, yy = np.meshgrid(x, y)
circle = xx**2 + yy**2

aperture[circle < radius**2] = 1

plt.figure(1)
plt.imshow(aperture, cmap='gray', interpolation='none')
plt.colorbar()

screen = np.loadtxt('wavefront.txt')
plt.figure(2)
plt.imshow(screen, cmap = 'jet', interpolation='none')
plt.colorbar()


speckle = np.abs(np.fft.fftshift(np.fft.fft2(aperture *
                                             np.exp(1j * screen))))**2

plt.figure(3)
plt.imshow(speckle[int(N/2-64):int(N/2+64), int(N/2-64):int(N/2+64)],  cmap='jet',
           aspect='auto', interpolation='none')
plt.colorbar()
plt.title('Intensity')
