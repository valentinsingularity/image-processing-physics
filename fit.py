import numpy as np
import matplotlib.pyplot as plt

exp_data = np.load('data.npy')

x = exp_data[:, 0]

measurements = exp_data[:, 1]

plt.figure(1)
plt.plot(x, measurements, 'o')
plt.title('Measured data points')
plt.xlabel('x')
plt.ylabel('measurement')

mat_linear = x
mat_const = np.ones(len(x))

f = 1
mat_sine = np.sin(f*x)

A = np.vstack((mat_const, mat_linear, mat_sine))
A = A.T

coeff = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), measurements)
coeff2 = np.linalg.lstsq(A, measurements)[0]

x_fit = np.linspace(x.min(), x.max(), 300)
y_fit = coeff[0] + coeff[1]*x_fit + coeff[2]*np.sin(f*x_fit)

plt.figure(2)
plt.title('Measurememts & Least Squares fit')
plt.plot(x, measurements, 'o', label='Measurements')
plt.plot(x_fit, y_fit, label='Least Squares fit')
plt.legend()
plt.xlabel('x')
plt.ylabel('measurement')
plt.show()
