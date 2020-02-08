import numpy as np
import matplotlib.pylab as plt

N = 512  # square dimension of phase screen

x = np.linspace(-N//2, N//2, N)
y = np.linspace(-N//2, N//2, N)
xx, yy = np.meshgrid(x, y)
aperture = xx**2 + yy**2 < 128**2

plt.figure(1)
plt.imshow(aperture, cmap='gray')
plt.title('Support constraint')

x = range(N) - N/2*np.ones(N) + 0.5
y = range(N) - N/2*np.ones(N) + 0.5
xx, yy = np.meshgrid(y,x)
tip = xx / np.max(xx)
tip = tip * aperture
tilt = yy / np.max(yy)
tilt = tilt * aperture

screen = tip*4. + tilt*3.

plt.figure(2)
plt.imshow(screen * aperture, cmap='jet')
plt.colorbar()
plt.title('Aperture phase')

speckle = np.abs(np.fft.fftshift(np.fft.fft2(aperture * np.exp(1j*screen))))**2

plt.figure(3)
plt.imshow(speckle[N//2-32:N//2+32,N//2-32:N//2+32], aspect='auto',
    extent=(N//2-32,N//2+32,N//2-32,N//2+32), interpolation='none', cmap='gray')
plt.colorbar()
plt.title('Intensity')

nloops = 50  
focal_magnitude = np.sqrt(speckle)

focal_plane = focal_magnitude * np.exp(1j*np.zeros((N, N)))

errors_aperture = np.zeros(nloops)
errors_focal = np.zeros(nloops)

for loop in np.arange(nloops):

    print(loop)

    aperture_plane = np.fft.ifft2(np.fft.ifftshift(focal_plane))
    aperture_plane = aperture_plane * aperture
    errors_aperture[loop] = np.sum((np.abs(aperture_plane)-aperture)**2)
    focal_plane = np.fft.fftshift(np.fft.fft2(aperture_plane))
    errors_focal[loop] = np.sum((np.abs(focal_plane) - focal_magnitude)**2)
    focal_plane = focal_magnitude * np.exp(1j*np.angle(focal_plane))


plt.figure(4)
plt.imshow(np.angle(aperture_plane) * aperture, cmap='jet')
plt.title('Phase aperture plane')
plt.colorbar()

plt.figure(5)
plt.imshow(np.abs(aperture_plane) * aperture, cmap='gray')
plt.title('Magnitude aperture plane')
plt.colorbar()

plt.figure(6)
plt.imshow(np.angle(focal_plane), cmap='jet')
plt.title('Phase focal plane')
plt.colorbar()

plt.figure(7)
plt.imshow(np.abs(focal_plane)[N//2-32:N//2+32,N//2-32:N//2+32], aspect='auto',
    extent=(N//2-32,N//2+32,N//2-32,N//2+32), interpolation='none', cmap='gray')
plt.title('Magnitude focal plane')
plt.colorbar()

plt.figure(8)
plt.plot(np.log(errors_aperture))
plt.xlabel('Iteration')
plt.ylabel('Log Error')
plt.title('Error reduction - Aperture plane')

plt.figure(9)
plt.plot(np.log(errors_focal))
plt.xlabel('Iteration')
plt.ylabel('Log Error')
plt.title('Error reduction - Focal plane')