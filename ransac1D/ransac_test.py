#!/usr/bin/python
# %load_ext autoreload
# %autoreload 2

import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

from ransac_polyfit import ransac_polyfit
import time

START = 0
END = 1 * np.pi
NSAMPLES = 10000
NORDER = 2

# Hand made np.polyval implementation.
def horner_scheme(coefs, start=0, end=1, ns=100):
  # Model's complexity.
  order = len(coefs)

  t = np.linspace(start, end, ns)
  y = np.zeros(ns)

  for n in range(order):
    i = order - n - 1
    y = y + (t ** n) * coefs[i]

  return y

#################################
##  Fitting into curved data.  ##
#################################

# Generate data to fit.
t = np.linspace(START, END, NSAMPLES)

y1 = np.sin(t)

y2 = np.sin(2*t**2)

y = y1 + 0.1 * y2

# Break it into short intervals.

# Fit LSM.
start = time.time()
model_lsm = np.polyfit(t, y, NORDER)
end = time.time()
print("LSM = ", end - start)
print(time.time(), time.clock())

y_fit_LSM = horner_scheme(model_lsm, START, END, NSAMPLES)

# Fit RanSaC.
start = time.time()
# bestfit = ransac_polyfit(t,y)
bestfit, besterr = ransac_polyfit(t, y, order=NORDER, n=12, k=20, t=0.2, d=10, f=0.5)
end = time.time()
print("RanSaC = ", end - start, "Err = ", besterr)
print(time.time(), time.clock())

y_fit = bestfit[0] * t**2 + bestfit[1] * t + bestfit[2]

# Approximate them by polynomials using RanSaC polyfit.

# Show input.
plt.subplot(3,1,1), plt.plot(t,y)
plt.title('Curved sin(t)'), plt.xticks([]), plt.yticks([])
# Show output.
plt.subplot(3,1,2), plt.plot(t,y, color = 'b'), plt.plot(t,y_fit_LSM, color = 'r')
plt.legend(('x=sin(t) + 0.1*sin(2*t**2)','Polyfit'), loc = 'upper left')
plt.title('Polyfit'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,3), plt.plot(t,y, color = 'b'), plt.plot(t,y_fit, color = 'r')
plt.legend(('x=sin(t) + 0.1*sin(2*t**2)','RanSaC + polyfit'), loc = 'upper left')
plt.title('RanSaC + polyfit'), plt.xticks([]), plt.yticks([])
plt.show()

#################################
##  Fitting into noisy data.   ##
#################################
Crand = 0.25;
noise_pos =  Crand * np.random.rand(len(t))
noise_neg = -Crand * np.random.rand(len(t))
noise = noise_pos + noise_neg

x = np.sin(t)
x_noisy = x + noise
x_noisy2 = x + 5 * noise_neg * (noise_pos > 0.9 * Crand) #+ 5 * noise_pos * (noise_neg < -0.9 * Crand)

# Fit LSM.
start = time.time()
model_noisy_lsm = np.polyfit(t, x_noisy2, NORDER)
end = time.time()
print("LSM = ", end - start)
print(time.time(), time.clock())

x_fit_LSM = horner_scheme(model_noisy_lsm, START, END, NSAMPLES)

# Fit RanSaC
start = time.time()
bestfit_noisy, besterr = ransac_polyfit(t, x_noisy2, order=NORDER, n=12, k=20, t=0.2, d=10, f=0.5)
end = time.time()
print("RanSaC = ", end - start, "Err = ", besterr)
print(time.time(), time.clock())

y_fit_noisy = bestfit[0] * t**2 + bestfit[1] * t + bestfit[2]

# Show input.
plt.subplot(3,1,1), plt.plot(t,x_noisy2)
plt.title('Noisy sin(t)'), plt.xticks([]), plt.yticks([])
# Show output.
plt.subplot(3,1,2), plt.plot(t,x_noisy2, color = 'b'), plt.plot(t,x_fit_LSM, color = 'r')
plt.legend(('x=sin(t) - noise','Polyfit'), loc = 'upper left')
plt.title('Polyfit'), plt.xticks([]), plt.yticks([])

plt.subplot(3,1,3), plt.plot(t,x_noisy2, color = 'b'), plt.plot(t,y_fit_noisy, color = 'r')
plt.legend(('x=sin(t) - noise','RanSaC + polyfit'), loc = 'upper left')
plt.title('RanSaC + polyfit'), plt.xticks([]), plt.yticks([])
plt.show()

