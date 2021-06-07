import numpy as np
import warnings

def get_init_values(x, y, **kwargs):
    """
    We'll assume that the x and y matrices are already conformable and have been lagged appropriately
    albeit this will change the makeup of the state vector later
    inputs: 
            x: feature matrix TxM, x should not include the intercept
            y: target matrix Txn
            **kwargs:
            lags: int, for the case where y.shape[1] > 1
    """
    if not 'n' in kwargs.keys():
        # if n is not supplied set n to 10 or (num_features^2) + 5
        if len(y.shape) == 1 or y.shape[1]==1: 
            n = max(10, x.shape[1]+5)
        else:
            n = max(10, x.shape[1]**2+5)


    if len(y.shape) == 1 or y.shape[1]==1:
        pass
    else:
        # check that x and y are conformable for a VAR specification
        num_features = x.shape[1]
        assert x.shape[1]%y.shape[1], "x shape {} is not conformable with y \
             shape {} for a VAR model".format(x.shape[1], y.shape[1])
    
    if 'add_intercept' in kwargs.keys() and kwargs['add_intercept']:
        x = add_intercept(x)


    y_sample = y[:n]
    x_sample = x[:n]
    
    # Run regressions and get theta, covar and h
    theta = theta_calc(x_sample, y_sample)
    covar = covar_calc(x_sample)

    h = h_calc(x_sample, y_sample, theta)

    if len(covar) == 1:
        pass
    else:
        covar = check_positive_definite(covar, 'covar')
    
    if not isinstance(h, np.ndarray):
        h = np.array([h])
    elif len(h) == 1:
        pass
    else:
        covar = check_positive_definite(h, 'prediction error covariance')
    
    # Vectorise theta
    theta = theta.flatten()
    # Prepare x values for the kalman filter
    if len(y.shape) == 1 or y.shape[1]==1:
        pass
    else:
        x = prepare_x(x, num_features)

    return x, y, theta, covar, h

def prepare_x(x, num_features):
    """
    If we are running a VAR x needs to be in SUR form
    """
    x_sur = [np.kron(np.eye(num_features),x[ii]) for ii in range(len(x))]
    return x_sur

def check_positive_definite(a, matrix_name):
    """
    Assume already symmetric
    """
    if not np.all(np.linalg.eigvals(a)>0):
        warnings.warn('--- {} --- is not semipositive definite'.format(matrix_name))
        a = np.diag(np.diag(a))
        if not np.all(np.linalg.eigvals(a)>0):
            raise ValueError('diagonal only of {} matrix is not positive definite, \
                                please check for multicolinearity in x data'.format(matrix_name))
        else:
            pass
    else:
        pass    

    return a

def add_intercept(x):
    """
    Add a column of ones two feature matrix
    """
    x = np.c_[np.ones(len(x)), x]

    return x

def theta_calc(x,y):
    """
    Lstq estimation of theta
    """
    theta = np.linalg.lstsq(x, y, rcond = None)[0].T

    return theta

def covar_calc(x):
    """
    Calculation of the variance/covariance matrix of the state
    """
    covar = np.dot(x.T, x)
    return covar

def h_calc(x, y, theta):
    """
    Calculate the prediction error variance
    """
    error = y - np.dot(theta, x.T).T
    h = np.dot(error.T,error)

    return h