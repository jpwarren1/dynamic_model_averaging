import numpy as np

def predict_state(x, theta):
    """

    """
    y_hat = np.dot(x, theta)

    return y_hat

def predict_var_covar_state(hyperparameter):
    """

    """
    def predict(covar_t_1):
        """

        """
        covar_t = (1/hyperparameter)*covar_t_1

        return covar_t
    
    return predict

def evaluate_prediction(y, yhat):
    """

    """
    error = y - yhat

    return error

def calculate_h(hyperparameter):
    """

    """
    def calculate(error, h_old):
        """

        """
        # The error arrays will be 1 dimensional, to avoid any problems with the
        # multiplication we will use the numpy outer function
        h = hyperparameter*h_old + (1-hyperparameter)*np.outer(error,error)

        return h
    
    return calculate

def evaluate_pred_variance(x, covar, h):
    """

    """
    pred_variance = np.dot(np.dot(x,covar),x.T) + h

    return pred_variance

def calculate_gain(x, covar, pred_variance):
    """

    """
    inv_pred_variance = np.linalg.inv(pred_variance)
    kalman_gain = np.dot(np.dot(covar,x.T), inv_pred_variance)
    return kalman_gain

def update_state(theta,kalman_gain, error):
    """

    """
    theta = theta + np.dot(kalman_gain,error)
    return theta

def update_state_var(covar, x, kalman_gain):
    """

    """
    covar = covar - np.dot(kalman_gain, np.dot(x,covar))
    return covar

def calculate_likelihood():
    """

    """
    def calculate(pred_variance, error, t):

        inv_pred_variance = np.linalg.inv(pred_variance)
        likelihood = t*constant - 1/2*np.log(np.linalg.det(pred_variance)) - 1/2* np.dot(np.dot(
                     error, inv_pred_variance),error)      
        return likelihood

    constant = -1/2*np.log((2*np.pi))
    return calculate
