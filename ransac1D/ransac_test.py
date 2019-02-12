#!/usr/bin/python
# %load_ext autoreload
# %autoreload 2

import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

from ransac_polyfit import ransac_polyfit

# Generate data to fit.
t = np.linspace(0,np.pi,1000)

y1 = np.sin(t)

y2 = np.sin(2*t**2)

y = y1 + 0.1 * y2

# Break it into short intervals.

# bestfit = ransac_polyfit(t,y)
bestfit = ransac_polyfit(t, y, order=2, n=20, k=100, t=0.2, d=10, f=0.5)

y_fit = bestfit[0] * t**2 + bestfit[1] * t + bestfit[2]

# Approximate them by polynomials using RanSaC polyfit.

# Show input.
plt.subplot(2,1,1), plt.plot(t,y)
plt.title('Curved sin(t)'), plt.xticks([]), plt.yticks([])
# Show output.
plt.subplot(2,1,2), plt.plot(t,y, color = 'b'), plt.plot(t,y_fit, color = 'r')
plt.legend(('x=sin(t) + 0.1*sin(2*t**2)','RanSaC + polyfit'), loc = 'upper left')
plt.title('RanSaC + polyfit'), plt.xticks([]), plt.yticks([])
plt.show()

#################################
##  Fitting into noisy data     #
#################################
Crand = 0.25;
noise_pos =  Crand * np.random.rand(len(t))
noise_neg = -Crand * np.random.rand(len(t))
noise = noise_pos + noise_neg

x = np.sin(t)
x_noisy = x + noise
x_noisy2 = x + 5 * noise_neg * (noise_pos > 0.9 * Crand) #+ 5 * noise_pos * (noise_neg < -0.9 * Crand)

bestfit_noisy = ransac_polyfit(t, x_noisy2, order=2, n=20, k=100, t=0.2, d=10, f=0.5)

y_fit_noisy = bestfit[0] * t**2 + bestfit[1] * t + bestfit[2]

# Show input.
plt.subplot(2,1,1), plt.plot(t,x_noisy2)
plt.title('Noisy sin(t)'), plt.xticks([]), plt.yticks([])
# Show output.
plt.subplot(2,1,2), plt.plot(t,x_noisy2, color = 'b'), plt.plot(t,y_fit_noisy, color = 'r')
plt.legend(('x=sin(t) - noise','RanSaC + polyfit'), loc = 'upper left')
plt.title('RanSaC + polyfit'), plt.xticks([]), plt.yticks([])
plt.show()

