from os import error
import numpy as np
import pandas as pd
from dma.kalman_filter import tvp_kf_equations as tvp, kf_main_loop as ml
import matplotlib.pyplot as plt

np.random.seed(1050)

n = 1000
# Create three features
x = np.random.rand(n,3)
# Create some time varying coefficients
theta_true = np.zeros((n,3))
theta_true[0,:] = np.random.rand(3)
error_scaler = np.array([0.01, 0.035, 0.05])
for ii in range(1, len(theta_true)):
    theta_true[ii,:] = theta_true[ii-1,:] + error_scaler*np.random.randn(3)

plt.plot(theta_true)
# I'm running on wsl2 and haven't set up the renderer
plt.savefig('coefficients.png')
plt.close()

y = np.zeros(n)
for ii in range(len(y)):
    y[ii] = np.dot(x[ii],theta_true[ii]) + 0.02*np.random.randn(1)

plt.plot(y)
plt.savefig('y.png')
plt.close()

y_ols = np.expand_dims(y, axis = 1)
theta_ols = np.zeros((n,3))
pred_ols = np.zeros(n)
for ii in range(len(y)):
    theta_ols[ii,:] = np.linalg.lstsq(x[:ii+1,:], y_ols[:ii+1,:], rcond = None)[0].flatten()
    pred_ols[ii] = y[ii] - np.dot(x[ii], theta_ols[ii])

rolling_window = np.zeros((n,3))
pred_ols_roll = np.zeros(n)
for ii in range(50,len(y)):
    rolling_window[ii,:] = np.linalg.lstsq(x[ii-20:ii+1,:], y_ols[ii-20:ii+1,:], rcond = None)[0].flatten()
    pred_ols_roll[ii] = y[ii] - np.dot(x[ii], theta_ols[ii])


pred_90, l_90, theta_90, covar_90 = ml.recurisve_kalman_filter(x,y,0.5,0.9)
pred_75, l_75, theta_75, covar_75 = ml.recurisve_kalman_filter(x,y,0.5,0.75)


theta_90_df = pd.DataFrame(theta_90)
theta_75_df = pd.DataFrame(theta_75)
theta_ols = pd.DataFrame(theta_ols)
theta_true = pd.DataFrame(theta_true)
rolling_window = pd.DataFrame(rolling_window)
pred_90 = pd.DataFrame(pred_90)

# Ignore the first 50 observations
fig, ax = plt.subplots(1,3, sharey='row', figsize=(15,6))
ax[0].plot(theta_true.iloc[50:,0], label = 'true', color='#69DB67', zorder=15)
ax[0].plot(theta_90_df.iloc[50:,0], label = 'lambda = 0.9', color= '#B187DB', zorder=10)
ax[0].plot(theta_75_df.iloc[50:,0], label = 'lambda = 0.75', color='#DB646F', alpha=0.5, zorder=5)
ax[0].plot(theta_ols.iloc[50:,0], label = 'ols', color='#DBCE8F', zorder=0)

ax[1].plot(theta_true.iloc[50:,1], label = 'true', color='#69DB67', zorder=15)
ax[1].plot(theta_90_df.iloc[50:,1], label = 'lambda = 0.9', color='#B187DB', zorder=10)
ax[1].plot(theta_75_df.iloc[50:,1], label = 'lambda = 0.75', color='#DB646F', alpha=0.5, zorder=5)
ax[1].plot(theta_ols.iloc[50:,1], label = 'ols', color='#DBCE8F', zorder=0)

ax[2].plot(theta_true.iloc[50:,2], label = 'true', color='#69DB67', zorder=15)
ax[2].plot(theta_90_df.iloc[50:,2], label = 'lambda = 0.9', color='#B187DB', zorder=10)
ax[2].plot(theta_75_df.iloc[50:,2], label = 'lambda = 0.75', color='#DB646F', alpha=0.5, zorder=5)
ax[2].plot(theta_ols.iloc[50:,2], label = 'ols', color='#DBCE8F', zorder=0)
plt.legend(loc='best', fontsize='x-small')
plt.savefig('true_estimated.png')


plt.close()

plt.plot(theta_true.iloc[50:,0], label = 'true', color='#69DB67')
plt.plot(theta_90_df.iloc[50:,0], label = 'lambda = 0.9', color='#B187DB', zorder=10)
plt.plot(rolling_window.iloc[50:,0], label = 'rolling window', color='#DB646F', alpha=0.5, zorder=5)
plt.legend()
plt.savefig('rolling_window.png')
plt.close()

error_90 = y - pred_90.values.flatten()
error_ols = y - pred_ols
plt.plot(error_90, label='error_90', color='#B187DB')
plt.plot(error_ols, label='error_ols', color='#DB646F')
plt.legend()
plt.savefig('error_comp.png')
plt.close()

print('sum of abs error, 90 lambda', np.sum(abs(error_90[50:])))
print('sum of abs error, ols', np.sum(abs(error_ols[50:])))

# How to deal with missing variables?
x_zero = x
x_zero[0:800,2] = np.random.rand(800)
pred_check, l_check, theta_check, covar_check = ml.recurisve_kalman_filter(x_zero,y,0.5,0.75)
a = pd.DataFrame(theta_check)
plt.plot(a.iloc[:,2])
plt.plot(theta_75_df.iloc[:,2])
plt.savefig('check.png')
plt.close()
