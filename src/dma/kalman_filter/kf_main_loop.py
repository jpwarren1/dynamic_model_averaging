import numpy as np
from dma.kalman_filter import tvp_kf_equations as tvp

def recurisve_kalman_filter(x, y, forgetting_factor, e_persistence):
    """

    """

    tt = len(y)
    # initialise parameter matrix
    theta = initialise_theta(x[0,:], y[0])


    # initialise closure which accept hyperparameters
    m_error_calc = tvp.calculate_h(e_persistence)
    covar_calc = tvp.predict_var_covar_state(forgetting_factor)
    lik_calc = tvp.calculate_likelihood(len(y[0]))


    return

def initialise_theta(x, y):
    """

    """
    x = check_shape(x)
    y = check_shape(y)

    theta = np.linalg.lstsq(x,y, rcond=None)[0]
    theta.flatten()

    return theta

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
