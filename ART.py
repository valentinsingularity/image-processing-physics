import numpy as np
import matplotlib.pyplot as plt
import tomolib_solution as tomolib

sinogram = np.load('phantom_sino256.npy')

plt.figure(1)
plt.clf()
plt.imshow(sinogram, cmap='gray', interpolation='none')
plt.colorbar()
plt.xlabel('Projection angles')
plt.ylabel('Spatial axis')

phantom = np.load('phantom_tomo256.npy')

plt.figure(2)
plt.clf()
plt.imshow(phantom, cmap='gray', interpolation='none')
plt.colorbar()
plt.title('Phantom')

Niter = 10

N, Nangles = sinogram.shape

theta_list = 360. * np.arange(Nangles)/Nangles

initial_tomo = np.zeros((N, N))

x, y = np.ogrid[-1:1:1j*N, -1:1:1j*N]
mask = (x*x + y*y <= 1.)
mask_proj = np.sum(mask, axis=0)
renorm = np.zeros(N)
renorm[mask_proj != 0] = 1. / mask_proj[mask_proj != 0]

tomo = initial_tomo.copy()
# tomo = tomolib.fbp(sinogram)
error = []

for i in range(Niter):
    err = 0
    plt.figure(3)
    plt.clf()
    plt.imshow(tomo, cmap='gray', interpolation='none')
    plt.colorbar()
    plt.title('ART reconstruction - after %i iterations' % i)
    plt.pause(0.1)

    for index, th in enumerate(theta_list):

        proj = tomolib.forwardproject(tomo, th)

        diff = sinogram[:, index] - proj

        err += (diff**2).sum()

        bpj = tomolib.backproject(renorm * diff, th)

        tomo += bpj

    print('Iteration ' + str(i) + ' completed.')
    print('Error: ' + str(err))
    error.append(err)

plt.figure(3)
plt.clf()
plt.imshow(tomo, cmap='gray')
plt.colorbar()
plt.title('ART reconstruction - 10 iterations')

plt.figure(4)
plt.clf()
plt.plot(error, label='ART')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()
