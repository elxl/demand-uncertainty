import torch
import numpy as np
from scipy.stats import poisson, nbinom, norm

def zipoisson_pmf(x, mu, pi):
    if x == 0:
        return pi + (1-pi)*poisson.pmf(x, mu)
    else:
        return (1-pi)*poisson.pmf(x, mu)
    
def zipoisson_pmf_torch(x, mu, pi):
    # Create a Poisson distribution with the specified mean (lambda)
    poisson_dist = torch.distributions.Poisson(mu)
    log_pmf = poisson_dist.log_prob(x)
    pmf = torch.exp(log_pmf)
    if x == 0:
        pmf = pi + (1-pi)*pmf
    else:
        pmf = (1-pi)*pmf    

    return pmf

def zipoisson_prob_log(x, mu, pi):
    return torch.log(zipoisson_pmf_torch(x, mu, pi))


def zipoisson_interval_single(z, mu, pi, k_max=100):
    lb = (1 - z)/2
    ub = (1 + z)/2
    lower_bound, upper_bound = None, None
    cumulative_prob = 0.0

    for k in range(k_max + 1):
        prob = zipoisson_pmf(k, mu, pi)
        cumulative_prob += prob

        if cumulative_prob >= lb and lower_bound is None:
            lower_bound = k

        if cumulative_prob >= ub and upper_bound is None:
            upper_bound = k
            break

    return lower_bound, upper_bound


def zipoisson_interval(z, mu, pi, k_max=100):
    data = np.concatenate((mu[np.newaxis,:], pi[np.newaxis,:]))
    lb, ub = np.apply_along_axis(lambda x:zipoisson_interval_single(z,x[0],x[1]),0,data)
    return lb, ub


def zinb_prob(x, r, p, pi):
    if x == 0:
        return pi + (1-pi)*nbinom.pmf(x, r, p)
    else:
        return (1-pi)*nbinom.pmf(x, r, p)

def nb_ppf(z, a, b, loc, scale):
    pa = norm.cdf(a, loc, scale)
    pb = norm.cdf(b, loc, scale)

    return norm.ppf(pa+z*(pb-pa))*scale + loc

def nb_interval(z, a, b, loc, scale):
    lb = (1-z)/2
    ub = (1+z)/2
    return nb_ppf(lb, a, b, loc, scale), nb_ppf(ub, a, b, loc, scale)