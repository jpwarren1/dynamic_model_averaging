import pytest
import numpy as np
import math
from dma.model_averaging import scoring as sc

def test_calculate_recursive_scores():
    likelihoods = np.array([[1,2,3]])
    probabilities = sc.calculate_recurive_scores(likelihoods, 0.9)

    assert math.isclose(sum(probabilities[-1,:]), 1, abs_tol=1e-4)
    return