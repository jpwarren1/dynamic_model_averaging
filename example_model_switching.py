import numpy as np
from dma.kalman_filter import kf_main_loop as ml
from dma.model_averaging import scoring as sc
import matplotlib.pyplot as plt

np.random.seed(9050)
n = 300
theta = np.zeros((n,3))
theta[0,:] = np.array([0.7,-0.4,0.3]) + np.array([0.01,0.02,0.03])*np.random.randn(3)
for ii in range(1, len(theta)):
    theta[ii] = theta[ii-1] + np.array([0.01,0.02,0.03])*np.random.randn(3)

x = np.random.rand(n,3)

y = np.zeros(n)
for ii in range(len(y)):
    if ii < 100:
        y[ii] = np.dot(theta[ii,0:1],x[ii, 0:1]) + 0.02*np.random.randn(1)
    elif (ii > 100) and (ii <200):
        y[ii] = np.dot(theta[ii,0:2],x[ii, 0:2]) + 0.02*np.random.randn(1)
    else:
         y[ii] = np.dot(theta[ii,:],x[ii, :]) + 0.02*np.random.randn(1)

plt.plot(y)
plt.savefig('y_ms1.png')
plt.close()

pred_1, l_1, theta_1, covar_1 = ml.recurisve_kalman_filter(x[:,0:1],y,0.5,0.75)
pred_2, l_2, theta_2, covar_2 = ml.recurisve_kalman_filter(x[:,0:2],y,0.5,0.75)
pred_3, l_3, theta_3, covar_3 = ml.recurisve_kalman_filter(x,y,0.5,0.75)

likelihoods = np.array((l_1,l_2,l_3)).T
likelihoods = np.exp(likelihoods)

probabilities = sc.calculate_recurive_scores(likelihoods, 0.99)
labels = ['Model 1', 'Model 2', 'Model 3']
colours = ['#B187DB', '#DB646F', '#DBCE8F']

plt.plot(probabilities[:,0], label = 'Model 1', color='#B187DB')
plt.plot(probabilities[:,1], label = 'Model 2', color='#DB646F')
plt.plot(probabilities[:,2], label = 'Model 2', color='#DBCE8F')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('probabilities1.png', bbox_inches='tight')
plt.close()