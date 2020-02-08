import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

img = plt.imread('bears.jpg') / 255.
sh = img.shape

red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]
print(sh, red.shape, green.shape, blue.shape)

plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('color')
plt.subplot(2, 2, 2)
plt.imshow(red, cmap='gray')
plt.title('red channel')
plt.subplot(2, 2, 3)
plt.imshow(green, cmap='gray')
plt.title('green channel')
plt.subplot(2, 2, 4)
plt.imshow(blue, cmap='gray')
plt.title('blue channel')

red_hist = np.histogram(red, bins=50, range=(0, 1))
green_hist = np.histogram(green, bins=50, range=(0, 1))
blue_hist = np.histogram(blue, bins=50, range=(0, 1))

red_bins = red_hist[1]
central_bins = (red_bins[1:] + red_bins[:-1]) / 2.

plt.figure(2)
plt.title('histograms of 3 color channels')
plt.plot(central_bins, blue_hist[0], label='blue')
plt.plot(central_bins, green_hist[0], label='green')
plt.plot(central_bins, red_hist[0], label='red')
plt.grid()
plt.legend()

img_noisy = img + 0.1 * np.random.standard_normal(sh)
img_noisy[img_noisy < 0.] = 0.
img_noisy[img_noisy > 1.] = 1.

plt.figure(3)
plt.title('noisy image')
plt.imshow(img_noisy, vmin=0, vmax=1.)

red_hist_noisy = np.histogram(img_noisy[..., 0], bins=50, range=(0, 1))
green_hist_noisy = np.histogram(img_noisy[..., 1], bins=50, range=(0, 1))
blue_hist_noisy = np.histogram(img_noisy[..., 2], bins=50, range=(0, 1))

plt.figure(4)
plt.title('histograms of 3 noisy color channels')
plt.plot(central_bins, blue_hist_noisy[0], label='blue')
plt.plot(central_bins, green_hist_noisy[0], label='green')
plt.plot(central_bins, red_hist_noisy[0], label='red')
plt.grid()
plt.legend()

sigma = (1, 1, 0)
img_filtered = nd.filters.gaussian_filter(img_noisy, sigma=sigma)

plt.figure(5)
plt.title('filtered image')
plt.imshow(img_filtered, vmin=0, vmax=1.)

red_hist_filtered = np.histogram(img_filtered[..., 0], bins=50, range=(0, 1))
green_hist_filtered = np.histogram(img_filtered[..., 1], bins=50, range=(0, 1))
blue_hist_filtered = np.histogram(img_filtered[..., 2], bins=50, range=(0, 1))

plt.figure(6)
plt.title('histograms of 3 filtered color channels')
plt.plot(central_bins, blue_hist_filtered[0], label='blue')
plt.plot(central_bins, green_hist_filtered[0], label='green')
plt.plot(central_bins, red_hist_filtered[0], label='red')
plt.grid()
plt.legend()
