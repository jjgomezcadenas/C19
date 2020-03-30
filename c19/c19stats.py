import numpy as np
from   typing  import Tuple, List
from numpy import sqrt

NN = np.nan
from . types  import Number, Array, Str, Range

from scipy.stats import nbinom
from scipy.stats import weibull_min
from scipy.stats import norm
from scipy.stats import skewnorm
from scipy.stats import lognorm


def c19_nbinom_transform(r0, k):
    """Transforms the definition used in C19 analysis to standard"""
    n = k
    p = (1 + r0/k)**(-1)
    return n, p


def c19_nbinom_pdf(r0, k, x):
    """PDF of the negative binomial used in c19 studies"""
    n, p = c19_nbinom_transform(r0, k)
    return nbinom.pmf(x, n, p)


def c19_nbinom_rvs(r0, k, size=10):
    """Generates random variates"""
    n, p = c19_nbinom_transform(r0, k)
    r= nbinom.rvs(n, p, size=size)
    return r


def weib_pdf(x,n,a):
    """Weibull distribution PDF with shape a and scale n"""
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


def c19_weib_rvs(mu, rms, size=10):
    """Generates random variates"""
    r = mu * np.random.weibull(rms, size=size)
    return r


def normal_pdf(x,mu,rms):
    return norm.pdf(x, loc=mu, scale=rms)


def normal_rvs(mu, sigma, size=10):
    """Generates random variates"""
    return norm.rvs(loc=mu, scale=sigma, size=size)


def sknormal_pdf(x,mu,rms, a):
    return skewnorm.pdf(x, a, loc=mu, scale=rms)


def sknormal_rvs(mu, sigma, a, size=0):
    """Generates random variates"""
    if size == 0:
        return skewnorm.rvs(a, loc=mu, scale=sigma)
    else:
        return skewnorm.rvs(a, loc=mu, scale=sigma, size=size)


def lognorm_pdf(x, mu, sigma):
    """lognorm distribution"""
    return lognorm.pdf(x, sigma, scale=np.exp(mu))


def hdt(x, zmeanHDT = 13, zmedianHDT = 9.1):
    """Hospitalization to death truncated"""

    muHDT    = np.log(zmedianHDT)
    sigmaHDT = np.sqrt(2*(np.log(zmeanHDT) - muHDT))
    return lognorm.pdf(x, sigmaHDT, scale=zmedianHDT)


def cCFR(df):
    """Computes the confirmed Case Fatality Rate"""
    def scale_cfr(df):
        """Computes mut, scaling the CFR"""
        case_incidence  = df['cases'].values
        cumulative_known_t = 0  # cumulative cases with known outcome at time tt
        # Sum over cases up to time t
        for i in np.arange(len(case_incidence)):
            known_i = 0 # number of cases with known outcome at time i
            for j in np.arange(i):
                known_j =  case_incidence[i - j]*hdt(j)
                known_i += known_j
            cumulative_known_t += known_i

        return cumulative_known_t

    bt = df['deaths'].sum() / df['cases'].sum()  #naive CFR
    pt = df['deaths'].sum() / scale_cfr(df) #cCFR
    return bt, pt


def total_cases(df, cCFRBaseline = 1.38):
    bt, pt = cCFR(df)
    underreporting_estimate = cCFRBaseline / (pt*100)
    total_cases = df.cases.sum() / underreporting_estimate
    return underreporting_estimate, total_cases
