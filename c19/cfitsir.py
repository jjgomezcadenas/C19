import numpy       as np
import scipy.stats as stats
from scipy.integrate import odeint

from   c19.types import SIR
import c19.basic_models as cbm
import c19.cfit         as cfit
from numpy.linalg import inv


# Total population, N.
N0 = 20000
# Initial number of infected and recovered individuals, I0 and R0.
i0, r0 = 1, 0 # 1 infected, 0-recuperated
y0 = cbm.set_sir_initial_conditions(N0, i0, r0)
# time in half-day (100 days)
ts = np.linspace(0, 100, 200)

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
R0 = 3.
gamma0 = 1./7.   # inverse of infection time
beta0 = R0 * gamma0

# alpha = (R0-1) * gamma

def _binedges(ts):
    dt = 0.5*(ts[-1] - ts[-2])
    tbins = np.array([ts[i] - 0.5*(ts[i+1] - ts[i]) for i in range(len(ts)-1)] + [ts[-1] - dt, ts[-1] +dt])
    #print(len(tbins))
    return tbins

def sir(N, beta, gamma, y0 = y0, ts = ts):
    #N, beta, gamma = sir_constrains(N, beta, gamma)
    ret = odeint(cbm.sir_deriv, y0, ts, args=(N, beta, gamma))
    S, I, R = ret.T
    sir = SIR(N, S, I, R, ts, R0, beta, gamma)
    return sir

def sir_rv(sir):
    tbins  = _binedges(sir.t)
    irv = stats.rv_histogram((sir.I, tbins))
    return irv

def sir_llike(times, N, beta, gamma, y0 = y0, ts = ts):
    isir = sir(N, beta, gamma, y0, ts)
    irv  = sir_rv(isir)
    return irv.logpdf(times)

def sir_fun(times, N, beta, gamma, y0 = y0, ts = ts):
    isir = sir(N, beta, gamma, y0, ts)
    irv  = sir_rv(isir)
    dt   = isir.t[1] - isir.t[0]
    nn   = float(np.sum(isir.I)) * dt
    return nn * irv.pdf(times)

#--- MLike tools

def sir_mle(cis, crs, ts, N, beta, gamma):

    isir  = sir(N, beta, gamma)
    tbins = _binedged(isir.t)
    hrvi  = stats.rv_histogram((tbins, cis))
    hrvi  = stats.rv_histogram((tbins, cis))


#---- Kalman Filter tools

def _hrandom(xs, ys, N):
    rvi = stats.rv_histogram((ys, xs))
    xis = rvi.rvs(size = int(N)) ## TODO: generate poisson Ni?
    cis, _  = np.histogram(xis, xs)
    return cis

def _poisson(ns):
    def _delta_ni(i):
        if (i == 0): return ns[0]
        return abs(ns[i] - ns[i-1])
    dns  = [_delta_ni(i) for i in range(len(ns))]
    udns = [np.random.poisson(dni) for dni in dns]
    cns  = [ni + dni for ni, dni in zip(ns, udns)]
    return np.array(cns)

def sir_experiment(sir):
    """ Generate an experiment with a sir model
    """
    #tbins = _binedges(sir.t)
    #cis = _hrandom(tbins, sir.I, np.sum(sir.I))
    #crs = _hrandom(tbins, sir.R, np.sum(sir.R))
    cis = _poisson(sir.I)
    crs = _poisson(sir.R)
    return cis, crs, sir.t


def _sir_kmeasurements(cis, crs, ts, N = N0, fi = 2.):

    def _mi(ci, ri):
        return np.array((ci, ri))

    def _umi(ci, ri):
        u = np.identity(2)
        u[0, 0] = fi * np.max((2.4, np.sqrt(ci)))
        u[1, 1] = fi * np.max((2.4, np.sqrt(ri)))
        return u

    def _hi(ci, ri, dti):
        h = np.zeros(4).reshape(2,2)
        h[0, 0] = ci * dti * (N-(ci+ri))/N
        h[0, 1] = - ci * dti
        h[1, 1] =   ci * dti
        return h

    dt = ts[1] - ts[0]
    ms  = [_mi (ci, ri)    for ci, ri in zip(cis, crs)]
    ums = [_umi(ci, ri)    for ci, ri in zip(cis, crs)]
    hs  = [_hi(ci, ri, dt) for ci, ri in zip(cis, crs)]

    size = len(cis)
    def _delta_mi(i):
        if (i == 0): return ms[0]
        return ms[i] - ms[i-1]
    dms = [_delta_mi(i) for i in range(size)]
    return ms, dms, ums, hs


def _kfilter(xp, uxp, m, um, h):
    prod_ = np.matmul
    ide = np.identity(2)
    res = m - prod_(h, xp.T)
    #print('res ', res)
    k   = np.matmul(prod_(uxp, h.T), inv(prod_(h, prod_(uxp, h.T)) + um))
    #print('k ', k)
    x   = xp + prod_(k, res.T)
    #print('x ', x)
    ux = prod_((ide - prod_(k, h)), uxp)
    #print('cov ', cov)
    return x, ux, res, k


def sir_kfilter(cis, ris, ts, N = N0, beta = beta0, gamma = gamma0, full_output = False,
                fi = 2., sigma = 100.):
    size = len(ts)
    x0, ux0 = np.array((beta, gamma)), sigma*np.identity(2)
    xs  = [x0          for ii in range(size)]
    uxs = [ux0         for ii in range(size)]
    res = [np.zeros(2) for ii in range(size)]
    ms, dms, ums, hs = _sir_kmeasurements(cis, ris, ts, N, fi)
    for i in range(size-1):
        xp, uxp = x0, ux0
        if (i >= 1):
            xp, uxp =  xs[i-1], uxs[i-1]
        xi, uxi, resi, _ = _kfilter(xp, uxp, dms[i], ums[i], hs[i])
        xs[i], uxs[i], res[i] = xi, uxi, resi
    result = (xs, uxs) if full_output is False else (xs, uxs, ms, dms, ums, hs, res)
    return result
