import numpy as np
from dma.kalman_filter import tvp_kf_equations as tvp, create_initialisations as cr

def recurisve_kalman_filter(x, y, e_persistence, forgetting_factor,  **kwargs):
    """

    """

    if not 'initialisations' in kwargs.keys():
        x, y, theta, covar, h = cr.get_init_values(x, y, **kwargs)
        # initialise parameter matrix
        # initialise as list so we can use for recording
        theta = [theta] # list for the parameter values
        covar = [covar] # list for the covariance values
    else:
        theta = [kwargs['initialisations']['theta']]
        covar = [kwargs['initialisations']['covar']]
        h     = [kwargs['initialisations']['h']]

    if len(y.shape) == 1 or y.shape[1]==1:
        pred, l, theta, covar = univariate_recursion(x,y, theta, covar, h,
                                                     e_persistence, forgetting_factor)
    else:
        pred, l, theta, covar = multivariate_recursion(x,y, theta, covar, h, 
                                                       e_persistence, forgetting_factor)


    return pred, l, theta, covar

def univariate_recursion(x,y, theta, covar, h, e_persistence, forgetting_factor):
    """
    Only difference between this and the multivariate version is the indexing of the x matrix
    we wish to keep the two dimensional shape of x for matrix conformability so we index ii:ii+1 
    """
    tt = len(y)
    l = [] # list for the likelihood
    pred = [] # List for the 1 step ahead prediction
    # initialise closure which accept hyperparameters
    m_error_calc = tvp.calculate_h(e_persistence)
    covar_calc = tvp.predict_var_covar_state(forgetting_factor)
    lik_calc = tvp.calculate_likelihood(1)

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


def multivariate_recursion(x,y, theta, covar, h, e_persistence, forgetting_factor):
    """
    Computes the Kalman recursion for the multivariate case, x is a list of lagged oberservations
    of y manipulated to be in the seemingly unrelated regression (SUR) form.
    i.e. x = kron(eye(3), x) for each t
    """
    tt = len(y)
    l = [] # list for the likelihood
    pred = [] # List for the 1 step ahead prediction
    # initialise closure which accept hyperparameters
    m_error_calc = tvp.calculate_h(e_persistence)
    covar_calc = tvp.predict_var_covar_state(forgetting_factor)
    lik_calc = tvp.calculate_likelihood(len(y[0]))

    for ii in range(tt):
        # Predict 
        pred.append(tvp.predict_state(x[-1], theta[-1])) 
        covar_cond = covar_calc(covar[-1])

        # Evaluate the prediction
        error = tvp.evaluate_prediction(y[ii], pred[-1])
        h = m_error_calc(error, h)
        error_variance = tvp.evaluate_pred_variance(x[-1], covar_cond, h)

        # calculate the likelihood
        l.append(lik_calc(error_variance, error))

        # update 
        # indexing on x to make sure it is a 2d array
        kalman_gain = tvp.calculate_gain(x[-1], covar_cond, error_variance)
        theta.append(tvp.update_state(theta[-1], kalman_gain, error))
        covar.append(tvp.update_state_var(covar_cond, x[ii:ii+1,:], kalman_gain))
    
    return pred, l, theta, covar
