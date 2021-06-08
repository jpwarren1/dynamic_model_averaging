import numpy as np
from dma.kalman_filter import kf_main_loop as ml
from dma.model_averaging import scoring as sc
import matplotlib.pyplot as plt

np.random.seed(3050)
n = 600
# Create a series for y which has 3 different states
x = np.random.randn(n,3)*np.array([0.1,0.2,0.3])
# we'll use static theta
theta = np.random.rand(3)

y = np.zeros(n)
for ii in range(len(y)):
    if ii < 200:
        y[ii] = np.dot(theta[0:1],x[ii, 0:1]) + 0.02*np.random.randn(1)
    elif (ii > 200) and (ii <400):
        y[ii] = np.dot(theta[0:2],x[ii, 0:2]) + 0.1*np.random.randn(1)
    else:
         y[ii] = np.dot(theta,x[ii, :]) + 0.05*np.random.randn(1)

plt.plot(y)
plt.savefig('y_ms.png')
plt.close()

pred_1, l_1, theta_1, covar_1 = ml.recurisve_kalman_filter(x[:,0:1],y,0.5,0.85)
pred_2, l_2, theta_2, covar_2 = ml.recurisve_kalman_filter(x[:,0:2],y,0.5,0.85)
pred_3, l_3, theta_3, covar_3 = ml.recurisve_kalman_filter(x,y,0.5,0.85)

likelihoods = np.array((l_1,l_2,l_3)).T
likelihoods = np.exp(likelihoods)

probabilities = sc.calculate_recurive_scores(likelihoods, 0.99)

plt.plot(probabilities)
plt.savefig('probabilities.png')
plt.close()


