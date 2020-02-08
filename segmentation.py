import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

img = plt.imread('stars.jpg')
img = np.sum(img, axis=-1, dtype=float)
img = img / img.max()

plt.figure(1)
plt.title('img')
plt.imshow(img, cmap='gray', interpolation='none', vmin=0., vmax=1.)
plt.colorbar()

threshold = 0.1
img_bin = img > threshold

plt.figure(2)
plt.title('img_bin')
plt.imshow(img_bin, cmap='gray', interpolation='none')

s1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
img_bin1 = nd.binary_closing(img_bin, structure=s1)

plt.figure(3)
plt.title('img_bin1')
plt.imshow(img_bin1, cmap='gray', interpolation='none')


s2 = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
img_bin2 = nd.binary_opening(img_bin1, structure=s2)

plt.figure(4)
plt.title('img_bin2')
plt.imshow(img_bin2, cmap='gray', interpolation='none')


s3 = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
img_bin3 = nd.binary_opening(img_bin2, structure=s3)

plt.figure(5)
plt.title('img_bin3')
plt.imshow(img_bin3, cmap='gray', interpolation='none')


plt.figure(6)
plt.imshow(img_bin.astype(int) + img_bin1.astype(int) + img_bin2.astype(int) +
           img_bin3.astype(int), cmap='jet', interpolation='none')
plt.colorbar()

img_lbld, num_stars = nd.label(img_bin3)

plt.figure(7)
plt.imshow(img_lbld, cmap='jet', interpolation='none')
plt.colorbar()



slice_list = nd.find_objects(img_lbld)

starnum = 105

plt.figure(8)
plt.title("star %i" % starnum)
plt.imshow(img_lbld[slice_list[starnum-1]], cmap='gray', interpolation='none')

star_list = [img_lbld[slc] > 0 for slc in slice_list]
mass_list = [star.sum() for star in star_list]
mass_list_sorted = np.sort(mass_list)

plt.figure(9)
plt.title("sizes of stars")
plt.hist(mass_list_sorted[:-1], bins=170, range=(0, 170), align='left')
