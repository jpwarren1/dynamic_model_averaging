import numpy as np
from dma.kalman_filter import tvp_kf_equations as tvp

def recurisve_kalman_filter(x, y, forgetting_factor, e_persistence):
    """

    """

    tt = len(y)
    # initialise parameter matrix
    # initialise as list so we can use for recording
    pred = []
    theta = [initialise_theta(x[0,:], y[0])]
    covar = [initialise_covar(x)]

    # initialise closure which accept hyperparameters
    m_error_calc = tvp.calculate_h(e_persistence)
    covar_calc = tvp.predict_var_covar_state(forgetting_factor)
    lik_calc = tvp.calculate_likelihood(len(y[0]))

    for ii in range(tt):
        # Predict 
        pred.append(tvp.predict_state(x[ii], theta[-1])) 
        covar_cond = covar_calc(covar[-1])

        # Evaluate the prediction
        error = tvp.evaluate_prediction(y, pred)
        h = m_error_calc(error, h)
        error_variance = tvp.evaluate_pred_variance(x, covar_cond, h)

        # calculate the likelihood
        l = tvp.calculate_likelihood(error_variance, error)

        # update
        kalman_gain = tvp.calculate_gain(x[ii], covar_cond, error_variance)
        theta = tvp.update_state(theta, kalman_gain, error)
        covar = tvp.update_state_var(covar_cond, x, kalman_gain)
    
    return 

def initialise_theta(x, y):
    """

    """
    x = check_shape(x)
    y = check_shape(y)

    theta = np.linalg.lstsq(x,y, rcond=None)[0]
    theta.flatten()

    return theta

def initialise_covar(x):
    """

    """
    x = check_shape(x)
    covar = np.outer(x.T,x)

    return covar

def check_shape(array):
    """
    """
    if len(array.shape)==1:
        array = np.expand_dims(array, axis = 0)
        return array
    elif len(array.shape)==2:
        return array
    else:
        raise ValueError('Vectors should be 1 or 2 dimensional {} was provide'.format(array.shape))
    
    return
