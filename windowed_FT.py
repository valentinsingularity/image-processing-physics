import numpy as np
import matplotlib.pyplot as plt


def calc_spectrogram(signal, width, sigma):

    signal = np.hstack([np.zeros(width//2), signal, np.zeros(width//2)])

    spectrogram = np.zeros((len(signal), len(signal)))

    for position in np.arange(width//2, len(signal) - width//2):
	
        windowed_signal = signal[position - width//2:position + width//2]

        gaussian = np.exp(
            -((np.linspace(0, width, width) - width//2)**2) / (2. * sigma**2))
        windowed_signal = windowed_signal * gaussian

        padded_window = np.zeros((len(signal)))

        padded_window[0:width] = windowed_signal

        local_spectrum = np.fft.fftshift(np.fft.fft(padded_window))

        spectrogram[:, position] = np.abs(local_spectrum)**2

    return spectrogram[:, width//2:len(signal) - width//2]


x = np.linspace(0, 1., 1000)
x2 = np.linspace(0, 4., 4000)
p1 = 1. / 80
p2 = 1. / 160
signal1 = np.cos(2 * np.pi / p1 * x)
signal2 = np.cos(2 * np.pi / p2 * x)
signal3 = (signal1 + signal2) / 2
signal4 = np.cos(2 * np.pi / p1 * x**2)

signal = np.hstack([signal1, signal2, signal3, signal4])

width = 50
sigma = width / 5.
spec = calc_spectrogram(signal, width, sigma)

spec = spec[0:spec.shape[0]//2, :]

plt.figure(1, figsize=(14, 4))
plt.plot(x2, signal)

plt.figure(2, figsize=(14, 4))
plt.imshow(spec, aspect='auto', extent=(0, 1, 0, 500), cmap='gray')
plt.title('spectrogram')
plt.ylabel('frequency')
plt.xlabel('time/space')
plt.show()
