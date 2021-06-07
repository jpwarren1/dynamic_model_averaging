import numpy as np
from dma.kalman_filter import tvp_kf_equations as tvp, create_initialisations as cr

def recurisve_kalman_filter(x, y, e_persistence, forgetting_factor,  **kwargs):
    """

    """

    tt = len(y)
    if not 'initialisation' in input.keys():
        x, y, theta, covar, h = cr.get_init_values(x, y, **kwargs)
    # initialise parameter matrix
    # initialise as list so we can use for recording
    pred = [] # List for the 1 step ahead prediction
    theta = [theta] # list for the parameter values
    covar = [covar] # list for the covariance values
    l = [] # list for the likelihood
    h = np.eye(len(y[0])) # Place holder, we at least know this is positive definite

    # initialise closure which accept hyperparameters
    m_error_calc = tvp.calculate_h(e_persistence)
    covar_calc = tvp.predict_var_covar_state(forgetting_factor)
    lik_calc = tvp.calculate_likelihood(len(y[0]))

    for ii in range(tt):
        # Predict 
        pred.append(tvp.predict_state(x[ii:ii+1,:], theta[-1])) 
        covar_cond = covar_calc(covar[-1])

        # Evaluate the prediction
        error = tvp.evaluate_prediction(y[ii], pred[-1])
        h = m_error_calc(error, h)
        error_variance = tvp.evaluate_pred_variance(x[ii:ii+1,:], covar_cond, h)

        # calculate the likelihood
        l.append(lik_calc(error_variance, error))

        # update 
        # indexing on x to make sure it is a 2d array
        kalman_gain = tvp.calculate_gain(x[ii:ii+1,:], covar_cond, error_variance)
        theta.append(tvp.update_state(theta[-1], kalman_gain, error))
        covar.append(tvp.update_state_var(covar_cond, x[ii:ii+1,:], kalman_gain))
    
    return pred, l, theta, covar
