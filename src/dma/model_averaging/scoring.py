import numpy as np

def calculate_recurive_scores(likelihoods, forgetting_factor):
    """

    """
    # Constant to stop machine 0
    jitter = 0.0001
    # Get the number of models
    n = likelihoods.shape[1]
    
    probabilities = np.zeros((len(likelihoods)+1, n))
    probabilities[0,:] = 1/n

    for ii in range(1,len(probabilities)):
        step = (probabilities[ii-1,:]**forgetting_factor + jitter)
        probabilities[ii,:] = step*likelihoods[ii-1,:]/sum(step*likelihoods[ii-1,:])
    
    return probabilities



    