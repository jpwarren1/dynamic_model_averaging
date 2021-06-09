import pandas as pd
import numpy as np
from dma.kalman_filter import kf_main_loop as ml
import matplotlib.pyplot as plt

np.random.seed(9050)
n = 300
theta = np.zeros((n,3))
theta[0,:] = np.array([0.7,-0.4,0.3]) + np.array([0.01,0.02,0.03])*np.random.randn(3)
for ii in range(1, len(theta)):
    theta[ii] = theta[ii-1] + np.array([0.01,0.02,0.03])*np.random.randn(3)

plt.plot(theta)
plt.savefig('theta_test.png')
plt.close()

# create x as just a random process
x = np.random.rand(n,3)
plt.plot(x)
plt.savefig('x_check.png')
plt.close()

# create y from theta and x
y = np.zeros(n)
for ii in range(n):
    y[ii] = np.dot(theta[ii], x[ii]) + 0.02*np.random.randn(1)

plt.plot(y)
plt.savefig('y_check.png')
plt.close()

# run through the filter
# Examples of alpha
pred_90, l_90, theta_90, covar_90 = ml.recurisve_kalman_filter(x,y,0.7,0.9)
pred_75, l_75, theta_75, covar_75 = ml.recurisve_kalman_filter(x,y,0.7,0.75)

# examples of Kappa
pred_kap99, l_kap99, theta_kap99, covar_kap99 = ml.recurisve_kalman_filter(x,y,0.99,0.75)
pred_kap50, l_kap50, theta_kap50, covar_kap50 = ml.recurisve_kalman_filter(x,y,0.5,0.75)
pred_kap50, l_kap10, theta_kap10, covar_kap10 = ml.recurisve_kalman_filter(x,y,0.1,0.75)
# Create empty data points to add extra label to the charts
empty = np.zeros(n)
# convert the theta's into dataframes
theta_90 = pd.DataFrame(theta_90)
theta_75 = pd.DataFrame(theta_75)
theta_true = pd.DataFrame(theta)
#Plot theta against the filtered versions
fig, ax = plt.subplots(1,2, sharey='row', figsize=(25,15))
ax[0].plot(theta_true.iloc[:,2], label = 'true', color='#69DB67', linewidth=4)
ax[0].plot(theta_90.iloc[:,2], label = 'lambda = 0.9', color= '#B187DB', linewidth=4)
ax[0].plot(theta_75.iloc[:,2], label = 'lambda = 0.75', color='#DB646F', linewidth=4)
ax[0].plot(empty, label = 'kappa = 0.7', color='w', alpha=0.0)
ax[0].legend(prop={'size': 20})
ax[0].tick_params(axis='both', labelsize=20)
ax[0].set_title('The impact of lambda', fontsize=25)


# convert the theta's into dataframes
theta_kap99 = pd.DataFrame(theta_kap99)
theta_kap50 = pd.DataFrame(theta_kap50)
theta_kap10 = pd.DataFrame(theta_kap10)

ax[1].plot(theta_true.iloc[:,2], label = 'true', color='#69DB67', linewidth=4)
ax[1].plot(theta_kap99.iloc[:,2], label = 'kappa = 0.99', color= '#B187DB', linewidth=4)
ax[1].plot(theta_kap50.iloc[:,2], label = 'kappa = 0.50', color='#DB646F', linewidth=4)
ax[1].plot(theta_kap10.iloc[:,2], label = 'kappa = 0.10', color='#DBCE8F', linewidth=4)
ax[1].plot(empty, label = 'lambda = 0.75', color='w', alpha=0.0)
ax[1].legend(prop={'size': 20})
ax[1].tick_params(axis='both', labelsize=20)
ax[1].set_title('The impact of kappa', fontsize=25)
plt.savefig('TVP_parameters.png')
plt.close()

x_zeros = x
x_zeros[:200, 2] = np.random.rand(200)
pred_z, l_z, theta_z, covar_z = ml.recurisve_kalman_filter(x_zeros,y,0.7,0.75)

theta_z = pd.DataFrame(theta_z)

plt.figure(figsize=(9,7))
plt.plot(theta_75.iloc[:,2], label = 'All observations', color='#DB646F')
plt.plot(theta_z.iloc[:,2], label = 'Missing Observations', color='#B187DB')
plt.plot(empty, label = 'lambda = 0.75', color='w', alpha=0.0)
plt.plot(empty, label = 'kappa = 0.7', color='w', alpha=0.0)

plt.legend()
plt.savefig('missing_obs.png')
plt.close()
