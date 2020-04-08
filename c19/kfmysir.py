import numpy as np
from numpy.linalg import inv

#
# def mysir_hi(ci, dt, N):
#     ni, nr, nd = ci[0], ci[1], ci[2]
#     h = np.zeros(size * size).reshape(size, size)
#     si = np.max(N - ni - nr - nd, 0)
#     h[0, 0], h[0, 1], h[0, 2] = si/N, -1., -1.
#     h[1, 1]                   = 1.
#     h[2, 2]                   = 1.
#     h = h * dt * ni
#     return h
#
# models = {'mysir', mysir_hi}
#
# def mysir_rvs(N, x0, ci0, model = 'mysir', t0 = 0., dt = 0.5, nsamples = 200):
#     # generate
#     #x0     = np.array((beta, gamma, rho))
#     #ci0    = np.array((ni, 0., 0.))
#     print('N ' , N)
#     print('x0' , x0)
#     ts     = [t0 + i * dt for i in range(nsamples)]
#     cis    = [ci0,]
#     nis    = list(cis)
#     hi_    = models[model]
#     #unis   = [np.identity(size),]
#     for i in range(1, nsamples):
#         cip = cis[i-1]
#         #print('cip', cip)
#         hi  = hi_(cip, dt, N)
#         #print('hi ', hi)
#         dci = np.matmul(hi, x0.T)
#         #print('dci ', dci)
#         ci  = cip + dci
#         #print('ci ', ci)
#         nip = nis[i-1]
#         dni = (dci/np.abs(dci)) * np.random.poisson(np.abs(dci))
#         ni  = nip + dni
#         #print(dci)
#         #print(dni)
#         cis .append(ci)
#         nis .append(ni)
#     return ts, cis, nis

#
# def _rvs_delta(xs, min_val = 1., type = int):
#     dxs = _delta(xs)
#     dxs = np.maximum(np.abs(dxs), min_val)
#     rxs = xs + (dxs / np.abs(dxs)) * np.random.poisson(dxs)
#     rxs = np.array(rxs, dtype = type)
#     return rxs
#
# def _udelta(dxs, minval = 2.4):
#     return np.maximum(np.sqrt(abs(dxs), min_val))
#
#
# def kf_measurements(ts, cs, ufactor = 2.4, N0 = 1.):
#     size = len(cs)
#
#     dms = [_delta(ci) for ci in cs]
#     #print(dms)
#     dms = [np.array(dmi) for dmi in zip(*dms)]
#
#     def _udmi(dmi):
#         x = np.maximum(np.sqrt(np.abs( N0 * dmi)), ufactor) / N0
#         u = np.identity(size) * (x * x)
#         return u
#
#     udms = [_udmi(dmi) for dmi in dms]
#
#     return dms, udms

#
#  Generic Kalman Filter
#


def _kfi(xp, uxp, m, um, h):
    prod_ = np.matmul
    ide = np.identity(len(xp))
    res = m - prod_(h, xp.T)
    #print('res ', res)
    k   = np.matmul(prod_(uxp, h.T), inv(prod_(h, prod_(uxp, h.T)) + um))
    #print('k ', k)
    x   = xp + prod_(k, res.T)
    #print('x ', x)
    ux = prod_((ide - prod_(k, h)), uxp)
    #print('cov ', cov)
    return x, ux, res, k


def _kfs(ms, ums, hs, x0, sigma = 100.):
    nsample, msize, ssize = len(ms), len(ms[0]), len(x0)
    ux0 = np.identity(ssize)*sigma*sigma
    xs  = [x0                            for ii in range(nsample)]
    uxs = [np.identity(ssize)            for ii in range(nsample)]
    res = [np.zeros(msize)               for ii in range(nsample)]
    for i in range(nsample):
        xp, uxp = x0, ux0
        if (i >= 1):
            xp, uxp =  xs[i-1], uxs[i-1]
        xi, uxi, resi, _ = _kfi(xp, uxp, ms[i], ums[i], hs[i])
        xs[i], uxs[i], res[i] = xi, uxi, resi
    return xs, uxs, res

def _rvs(cs):
    nsample, size = len(cs), len(cs[0])
    n0 = np.random.poisson(np.abs(cs[0]))
    ns = [n0,]
    for i in range(1, nsample):
        dci = cs[i] - cs[i-1]
        nip = ns[i-1]
        sig = np.ones(size)
        sig[dci < 0 ] = -1.
        dni = sig * np.random.poisson(np.abs(dci))
        ni  = nip + dni
        ns .append(ni)
    return ns

def _hrvs(N, x0, ci0, hi_, t0 = 0, dt = 0.5, nsamples = 200):
    size = len(ci0)
    ts     = [t0 + i * dt for i in range(nsamples)]
    cis    = [ci0,]
    for i in range(1, nsamples):
             cip = cis[i-1]
    #         #print('cip', cip)
             hi  = hi_(cip, dt, N)
    #         #print('hi ', hi)
             dci = np.matmul(hi, x0.T)
    #         #print('dci ', dci)
             ci  = cip + dci
             cis .append(ci)
    nis = _rvs(cis)
    return ts, cis, nis

#
# Specific KF
#


def _delta(xs, type = float):
    dxs = np.copy(xs)
    #dxs[1:] = dxs[1:] - xs[:-1]
    #dxs = np.array(dxs, dtype = type)
    dxs = dxs[1:] - dxs[:-1]
    return dxs

def delta_ms(cs, ufactor = 2., umin = 2.4):
    size = len(cs[0])
    ms  = _delta(cs)
    ums = [np.identity(size) * ufactor * np.maximum(np.sqrt(np.abs(ci)), umin) for ci in cs]
    return ms, ums

def mysir_hi(ci, dt, N):
    size = 3
    ni, nr, nd = ci[0], ci[1], ci[2]
    h = np.zeros(size * size).reshape(size, size)
    si = N - np.sum(ci)
    h[0, 0], h[0, 1], h[0, 2] = si/N, -1., -1.
    h[1, 1]                   = 1.
    h[2, 2]                   = 1.
    h = h * dt * ni
    return h

def hs_(ts, cs, N, hi_):
    dts = _delta(ts)
    hs = [hi_(ci, dt, N) for dt, ci in zip(dts, cs[:-1])]
    return hs

def mysir_rvs(N, x0, **kargs):
    ci = np.array((1, 0, 0))
    return _hrvs(N, x0, ci, mysir_hi, **kargs)

def mysir_kf(ts, cs, x0, N, full_output = False, **kargs):
    ms, ums      = delta_ms(cs)
    hs           = hs_(ts, cs, N, mysir_hi)
    xs, uxs, res = _kfs(ms, ums, hs, x0, N, **kargs)
    result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
    return result

#
#  SIR Model
#

def sir_hi(ci, dt, N):
    size = 2
    ni, nr = ci[0], ci[1]
    h = np.zeros(size * size).reshape(size, size)
    si = N - np.sum(ci)
    h[0, 0], h[0, 1] = si/N, -1.
    h[1, 1]          = 1.
    h = h * dt * ni
    return h

def sir_rvs(N, x0, **kargs):
    ci = np.array((1, 0))
    return _hrvs(N, x0, ci, sir_hi, **kargs)

def sir_kf(ts, cs, x0, N, full_output = False, **kargs):
    ms, ums      = delta_ms(cs)
    hs           = hs_(ts, cs, N, sir_hi)
    xs, uxs, res = _kfs(ms, ums, hs, x0, N, **kargs)
    result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
    return result

#
#  SEIR model
#

def seir_hi(ci, dt, N):
    size = 3
    ne, ni, nr = ci[0], ci[1], ci[2]
    si  = N - np.sum(ci)
    h = np.zeros(size * size).reshape(size, size)
    h[0, 0], h[0, 2] =  ni * si/N, -ne
    h[1, 1], h[1, 2] = -ni       ,  ne
    h[2, 1]          =  ni
    h = h * dt
    return h

def seir_rvs(N, x0, ci = (1, 1, 0), **kargs):
    ci = np.array(ci)
    return _hrvs(N, x0, ci, seir_hi, **kargs)


def seir_kf(ts, cs, x0, N, full_output = False, **kargs):
    ms, ums      = delta_ms(cs)
    hs           = hs_(ts, cs, N, seir_hi)
    xs, uxs, res = _kfs(ms, ums, hs, x0, N, **kargs)
    result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
    return result

#
# SEIR2 model
#

def seir2_hi(ci, dt, N):
    ne, ni, nr, nd = ci[0], ci[1], ci[2], ci[3]
    si  = N - np.sum(ci)
    h = np.zeros(4 * 5).reshape(4, 5)
    h[0, 0], h[0, 2] =  ni * si/N, -ne
    h[1, 1], h[1, 2] = -ni       ,  ne
    h[2, 1], h[2, 3] =  ni       , -ni
    h[3, 3], h[3, 4] =  ni       , -nd
    h = h * dt
    return h


def seir2_rvs(N, x0, ci = (1, 1, 0, 0), **kargs):
    ci = np.array(ci)
    return _hrvs(N, x0, ci, seir2_hi, **kargs)


def seir2_kf(ts, cs, x0, N, full_output = False, **kargs):
    ms, ums      = delta_ms(cs)
    hs           = hs_(ts, cs, N, seir2_hi)
    xs, uxs, res = _kfs(ms, ums, hs, x0, N, **kargs)
    result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
    return result



#
# SIR2 no-exposed model
#

def sir2_hi(ci, dt, N):
    ni, nr, nd = ci[0], ci[1], ci[2]
    si  = N - np.sum(ci)
    h = np.zeros(3 * 4).reshape(3, 4)
    h[0, 0], h[0, 1] =  ni * si/N, -ni
    h[1, 1], h[1, 2] =  ni       , -ni
    h[2, 2], h[2, 3] =  ni       , -nd
    h = h * dt
    return h


def sir2_rvs(N, x0, ci = (1, 0, 0), **kargs):
    ci = np.array(ci)
    return _hrvs(N, x0, ci, sir2_hi, **kargs)


def sir2_kf(ts, cs, x0, N, full_output = False, **kargs):
    ms, ums      = delta_ms(cs)
    hs           = hs_(ts, cs, N, sir2_hi)
    xs, uxs, res = _kfs(ms, ums, hs, x0, N, **kargs)
    result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
    return result

#
# SIR2 completed with death
#


def sir2c_hi(ci, dt, N):
    ni, nr, nd, nm = ci[0], ci[1], ci[2], ci[3]
    si  = N - np.sum(ci)
    h = np.zeros(4 * 4).reshape(4, 4)
    h[0, 0], h[0, 1] =  ni * si/N, -ni
    h[1, 1], h[1, 2] =  ni       , -ni
    h[2, 2], h[2, 3] =  ni       , -nd
    h[3, 3]          =  nd
    h = h * dt
    return h


def sir2c_rvs(N, x0, ci = (1, 0, 0, 0), **kargs):
    ci = np.array(ci)
    return _hrvs(N, x0, ci, sir2c_hi, **kargs)


def sir2c_kf(ts, cs, x0, N, full_output = False, **kargs):
    ms, ums      = delta_ms(cs)
    hs           = hs_(ts, cs, N, sir2c_hi)
    xs, uxs, res = _kfs(ms, ums, hs, x0, N, **kargs)
    result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
    return result

#---------

#
# def kf_hssir_ms(ts, cs, N):
#
#
# def kf_kfilter(ts, nis, x0, N, mifun, hifun, sigma = 100., ufactor = 1., full_output = False):
#     nsample, size,  msize = len(ts), len(x0), len(nis)
#     dt = ts[1] - ts[0]
#     ux0 = np.identity(size)*sigma*sigma
#     xs  = [x0                            for ii in range(nsample)]
#     uxs = [np.identity(size)             for ii in range(nsample)]
#     res = [np.zeros(size)                for ii in range(nsample)]
#     #hmatrix = hmatrix[model]
#     ms  = _delta(nis)
#     ums = [np.identity(msize) * np.abs(ms) * ufactor]
#     hi_ = models[model]
#     hs  = [hi_(ni, dt, N) for ni in nis]
#     for i in range(nsample):
#         xp, uxp = x0, ux0
#         if (i >= 1):
#             xp, uxp =  xs[i-1], uxs[i-1]
#         xi, uxi, resi, _ = _kfilter(xp, uxp, ms[i], ums[i], hs[i])
#         xs[i], uxs[i], res[i] = xi, uxi, resi
#     result = (xs, uxs) if full_output is False else (xs, uxs, ms, ums, res)
#     return result
#
#
# def kf_kfilter(ts, cs, x0, N, model = 'seir2', sigma = 100., ufactor = 2., full_output = False):
#     nsample, size = len(ts), len(x0)
#     ux0 = np.identity(size)*sigma*sigma
#     xs  = [x0                            for ii in range(nsample)]
#     uxs = [np.identity(size)             for ii in range(nsample)]
#     res = [np.zeros(size)                for ii in range(nsample)]
#     #hmatrix = hmatrix[model]
#     ms, ums = kf_measurements(ts, cs)
#     ms  [-1] = cs[-1]
#     for i in range(nsample):
#         ums[-1][4, 4] = mp.sqrt(np.maximum(ms[-1][i], 2.4))
#     hs      = kf_hmatrices_seir2(ts, cs, N)
#     for i in range(nsample):
#         xp, uxp = x0, ux0
#         if (i >= 1):
#             xp, uxp =  xs[i-1], uxs[i-1]
#         xi, uxi, resi, _ = _kfilter(xp, uxp, ms[i], ums[i], hs[i])
#         xs[i], uxs[i], res[i] = xi, uxi, resi
#     result = (xs, uxs) if full_output is False else (xs, uxs, ms, ums, hs, res)
#     return result
