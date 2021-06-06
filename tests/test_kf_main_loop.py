import pytest
import numpy as np
from dma.kalman_filter import kf_main_loop as ml

def test_check_shape_array():
    """
    """
    array = np.random.rand(4)
    out = ml.check_shape(array)

    assert out.shape == (1,4) 

    return

def test_check_shape_value():
    """
    """
    array = np.random.rand(1)
    out = ml.check_shape(array)

    assert out.shape == (1,1)

    return

def test_check_shape_error():
    """
    """
    array = np.random.rand(4)
    array = np.expand_dims(np.expand_dims(array, axis = 0), axis = 0)

    with pytest.raises(ValueError):
        out = ml.check_shape(array)



