import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

img = plt.imread('tree.jpg') / 255.
img = np.mean(img, axis=2)
sh = np.shape(img)

step = 1
Nsteps = 20
angle = 360 / Nsteps

img_cropped = img[300:600, 300:600]
img_order0 = img_cropped
img_order1 = img_cropped
img_order2 = img_cropped
img_order3 = img_cropped
img_order5 = img_cropped

plt.figure(1)

while step <= Nsteps:

    print('rotation No ' + str(step) + ' angle ' + str(step * angle))

    img_order0 = nd.rotate(img_order0, angle, order=0, reshape=False)
    img_order1 = nd.rotate(img_order1, angle, order=1, reshape=False)
    img_order2 = nd.rotate(img_order2, angle, order=2, reshape=False)
    img_order3 = nd.rotate(img_order3, angle, order=3, reshape=False)
    img_order5 = nd.rotate(img_order5, angle, order=5, reshape=False)

    plt.ion()
    plt.subplot(231)
    plt.imshow(img_order0, cmap='gray', interpolation='none')
    plt.title('nearest neighbour')
    plt.subplot(232)
    plt.imshow(img_order1, cmap='gray', interpolation='none')
    plt.title('bilinear')
    plt.subplot(233)
    plt.imshow(img_order2, cmap='gray', interpolation='none')
    plt.title('biquadratic')
    plt.subplot(234)
    plt.imshow(img_order3, cmap='gray', interpolation='none')
    plt.title('bicubic')
    plt.subplot(235)
    plt.imshow(img_order5, cmap='gray', interpolation='none')
    plt.title('5th order')

    plt.pause(.1)

    step += 1


plt.subplot(236)
plt.imshow(img_cropped, cmap='gray', interpolation='none')
plt.title('original')
