import numpy as np
from scipy.stats import poisson, nbinom

def zipoisson_prob(mu, pi, x):
    if x == 0:
        return pi + (1-pi)*poisson.cdf(x, mu)
    else:
        return (1-pi)*poisson.cdf(x, mu)
    

def zipoisson_prob_log(mu, pi, x):
    return np.log(zipoisson_prob(mu, pi, x))


def zinb_prob(r, p, pi, x):
    if x == 0:
        return pi + (1-pi)*nbinom.cdf(x, r, p)
    else:
        return (1-pi)*nbinom.cdf(x, r, p)
