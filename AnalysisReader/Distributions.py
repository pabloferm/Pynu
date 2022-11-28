# Provides parameters of non-Gaussian probability distributions

def Beta(m,s):
    alpha = ((1-m)/s**2-1/m)*m**2
    beta = alpha*(1/m-1)
    return alpha,beta
