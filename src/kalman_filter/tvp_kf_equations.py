import numpy as np

def predict_state(x, theta):
    """

    """
    y_hat = np.dot(theta, x)

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
    def calculate(error, h_old, t):
        """

        """
        sum_h = (hyperparameter**t)*error + h_old
        h  = np.sqrt(1-hyperparameter*sum_h)

        return h
    
    return calculate

def evaluate_pred_variance(x, covar, h):
    """

    """
    inv_pred_variance = np.linag.inv(np.dot(np.dot(x,covar),x.T) + h)

    return inv_pred_variance

def calculate_gain(x, covar, inv_pred_variance):
    """

    """
    kalman_gain = np.dot(np.dot(covar,x), inv_pred_variance)
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

