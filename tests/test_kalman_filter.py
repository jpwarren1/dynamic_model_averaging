import pytest
import numpy as np
from dma.kalman_filter import tvp_kf_equations as tvp

@pytest.fixture(scope='function')
def kronecker_x():
    """
    Set up for a 3 by 3 var
    """
    x = np.ones(3)
    x = np.kron(np.eye(3),x)

    return x

@pytest.fixture(scope='function')
def coeff_flatten():
    """
    3 by 3 var, flattened
    """
    theta = np.eye(3)
    theta = theta.flatten()

    return theta

@pytest.fixture(scope='function')
def x_univariate():
    """
    Multiple features a single target
    """
    x = np.array([1,2,3])
    return x

@pytest.fixture(scope='function')
def theta_univariate():
    """
    Multiple features a single target
    """
    theta = np.array([1,2,3])
    return theta

@pytest.fixture(scope='function')
def covar_multi():
    covar = np.eye(3)
    covar = np.kron(np.eye(3), covar)
    return covar

def test_predict_state_var(kronecker_x, coeff_flatten):

    out = tvp.predict_state(kronecker_x, coeff_flatten)

    # assert that we get a 2 dim structure out
    assert (out == np.ones(len(out))).all()

    return

def test_predict_state_uni(x_univariate, theta_univariate):
    out = tvp.predict_state(x_univariate, theta_univariate)

    # assert that we get a 2 dim structure out
    assert out == 14

    return

def test_predict_covar_univariate():

    predict_covar_state = tvp.predict_var_covar_state(0.9)
    out = predict_covar_state(1)

    assert out == 1/0.9

    return

def test_predict_covar_multivariate(covar_multi):
    """
    Just for completeness tbh
    """

    check = np.eye(3)*1/0.9
    check = np.kron(np.eye(3), check)
    predict_covar_state = tvp.predict_var_covar_state(0.9)
    out = predict_covar_state(covar_multi)

    assert (out == check).all()

    return

def test_evaluate_prediction():
    """

    """
    y = np.array([2])
    y_hat = 1
    out = tvp.evaluate_prediction(y, y_hat)

    assert (out == 1).all()

    return

def test_calculate_h():
    """

    """
    calc = tvp.calculate_h(0.9)
    h_old = np.array([0])
    error = np.array([1])
    out = calc(error, h_old)

    np.testing.assert_array_almost_equal(out, np.array([[0.1]]))

    return

def test_calculate_h_multivariate():
    """
    """
    calc = tvp.calculate_h(0.9)
    h_old = np.zeros((3,3))
    error = np.ones((3,1))
    out = calc(error, h_old)

    np.testing.assert_array_almost_equal(out, np.ones((3,3))*0.1)

    return

def test_evaluate_pred_variance():
    """
    """
    x = np.array([[3]])
    covar = np.array([[2]])
    h = 7
    out = tvp.evaluate_pred_variance(x, covar, h)

    assert (out == 1/25).all()

    return

def test_evaluate_pred_variance_multivariate(kronecker_x, covar_multi):
    """
    """
    h = np.zeros((3,3))

    out = tvp.evaluate_pred_variance(kronecker_x, covar_multi, h)

    np.testing.assert_array_almost_equal(out, np.eye(3)*1/3)

    return

def test_calculate_gain():
    """
    """
    x = np.array([[3]])
    covar = np.array([[2]])
    inv_pred_variance = np.array([[3]])
    out = tvp.calculate_gain(x, covar, inv_pred_variance)

    assert (out == 18).all()

    return

def test_calculate_gain(kronecker_x, covar_multi):
    """
    """
    h = np.zeros((3,3))
    check = np.kron(np.eye(3),np.ones(3)*1/3).T

    inv_pred_variance = tvp.evaluate_pred_variance(kronecker_x, covar_multi, h)
    out = tvp.calculate_gain(kronecker_x, covar_multi, inv_pred_variance)

    np.testing.assert_array_almost_equal(out, check)

    return

def test_update_state_univariate():
    #tvp.update_state(theta,kalman_gain, error)
    kalman_gain = np.array([1])
    error = np.array([0.1])
    theta = 0.9
    out = tvp.update_state(theta,kalman_gain, error)

    assert out == 1

    return

def test_update_state_multivariate(kronecker_x, covar_multi):
    h = np.zeros((3,3))
    theta = np.ones(9)*(2/3)
    error = np.ones(3)

    inv_pred_variance = tvp.evaluate_pred_variance(kronecker_x, covar_multi, h)
    kalman_gain = tvp.calculate_gain(kronecker_x, covar_multi, inv_pred_variance)
    out = tvp.update_state(theta, kalman_gain, error)

    np.testing.assert_array_almost_equal(out, np.ones(9))

    return

def test_update_state_covar_uni():
    kalman_gain = np.array([0.5])
    x = np.array([1])
    covar = np.array(0.25)
    out = tvp.update_state_var(covar,x,kalman_gain)

    assert out == 0.125

    return

def test_update_state_covar_multivariate(kronecker_x,covar_multi):
    kalman_gain = np.zeros((9,3))
    kalman_gain[0,0] = 1
    kalman_gain[3,1] = 1
    kalman_gain[6,2] = 1
    check = np.eye(3)
    check[0,0] = 0
    check[0,1:] = -1
    check = np.kron(np.eye(3), check)

    out = tvp.update_state_var(covar_multi, kronecker_x, kalman_gain)

    np.testing.assert_array_almost_equal(out, check)

    return

