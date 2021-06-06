import numpy as np
from dma.kalman_filter import tvp_kf_equations as tvp

def recurisve_kalman_filter(x, y):
    """

    """

    tt = len(y)
    # initialise parameter matrix

    return

def initialise_theta(x, y):
    """

    """
    x = check_shape(x)
    y = check_shape(y)


    return 

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
