import pytest
from math import isclose
import numpy as np
from dma.kalman_filter import create_initialisations as cr

@pytest.fixture(scope = 'function')
def x():
    x = np.random.rand(100,3)

    return x

@pytest.fixture(scope = 'function')
def y_uni():
    y = np.zeros(100) + np.array([0.5])

    return y
    
@pytest.fixture(scope = 'function')
def y_multi():
    y = np.zeros((100,3)) + np.array([1, 0, -1])

    return y

def test_add_intercept(x):
    x = cr.add_intercept(x)

    assert (x[:,0] == 1).all()

    return

def test_theta_calc_uni(x, y_uni):
    x = cr.add_intercept(x)
    theta = cr.theta_calc(x,y_uni)
    
    isclose(theta[0], 0.5, abs_tol=1e-4)

    return

def test_theta_calc_multi(x, y_multi):
    x = cr.add_intercept(x)
    theta = cr.theta_calc(x,y_multi)

    np.testing.assert_array_almost_equal(theta[:,0], np.array([1, 0, -1]))

    return

def test_covar_calc(x):
    """
    Relatively meaningless test
    """
    covar = cr.covar_calc(x)
    np.testing.assert_almost_equal(covar, np.dot(x.T,x))

    return

def test_h_calc_uni(x, y_uni):
    x = cr.add_intercept(x)
    theta = cr.theta_calc(x,y_uni)
    out = cr.h_calc(x, y_uni, theta)

    assert isclose(out, 0,abs_tol=1e-4) 

    return

def test_h_calc_multi(x, y_multi):
    x = cr.add_intercept(x)
    theta = cr.theta_calc(x,y_multi)
    out = cr.h_calc(x, y_multi, theta)

    np.testing.assert_almost_equal(out, np.zeros((3,3)))

    return

@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_check_positive_definite():
    c_test = np.eye(3)
    out = cr.check_positive_definite(c_test, 'good')
    np.testing.assert_array_equal(out,c_test)

    return

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_check_positive_definite_error():
    # A matrix of zeros is no positive definite
    c_test = np.zeros((3,3))
    with pytest.raises(ValueError):
        out = cr.check_positive_definite(c_test, 'bad')
    
    return

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_check_positive_definite_error():
    # This matrix shouldn't be pos-def, but the diagonal should
    c_test = np.ones((3,3))
    out = cr.check_positive_definite(c_test, 'diag')
    np.testing.assert_array_equal(out, np.eye(3))

    return

def test_prepare_x(x):
    x = cr.add_intercept(x)
    x = cr.prepare_x(x, num_features=3)
    shapes = [True for row in x if row.shape == (3,12)]
    assert all(shapes)

    return

def test_SUR():
    """
    Check that the SUR calculation is correct
    """
    x = np.array([[2,3,4]])
    y = np.array([[1,2,3]])
    x = cr.add_intercept(x)
    theta = cr.theta_calc(x,y)
    normal = np.dot(theta, x.T).T
    normal = normal.flatten()

    x = cr.prepare_x(x, 3)
    x = x[0]
    theta = theta.flatten()
    sur = np.dot(x,theta)

    np.testing.assert_almost_equal(normal, sur)
    return

def test_get_init_values_uni(x, y_uni):
    """
    Testing the wrapping function
    """
    x_check = np.c_[np.ones(len(x)), x]
    add_intercept = True
    x, y, theta, covar, h = cr.get_init_values(x,y_uni,add_intercept=add_intercept)


    # Assert x has the intercept added
    # Assert y is unchanged
    # Assert theta is a vector with 4 elements
    # Assert h is a array with 1 element

    assert np.all(x == x_check)
    assert all(y == y_uni)
    assert theta.shape == (4,)
    assert h.shape == (1,)

    return


