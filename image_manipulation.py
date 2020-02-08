import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

img = plt.imread('tree.jpg')
img_red = img[:, :, 0]

plt.figure(1)
plt.imshow(img_red, cmap='gray')
plt.colorbar()

img_crop = img_red[360:480, 370:490].copy()
img_crop_inv = img_crop.max() - img_crop

plt.figure(2)
plt.imshow(img_crop_inv, cmap='gray')
plt.colorbar()

threshold = 90
img_binary = img_red < threshold

plt.figure(3)
plt.imshow(img_binary, cmap='gray')

line_tree = img_red[:, 425]

plt.figure(4)
plt.plot(line_tree)

img_seg = np.zeros((400, 400))
cs = img_crop.shape
ss = img_seg.shape

img_seg[ss[0]//2 - cs[0]//2:ss[0]//2 + cs[0]//2,
        ss[1]//2 - cs[1]//2:ss[1]//2 + cs[1]//2] = img_crop

plt.figure(5)
plt.imshow(img_seg, cmap='gray')

img_rot = nd.rotate(img_seg, 45, reshape=False)

plt.figure(6)
plt.imshow(img_rot, cmap='gray')
