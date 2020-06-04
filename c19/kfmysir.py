import numpy as np
from numpy.linalg import inv

mprod_ = np.matmul


def _kfi(xp, uxp, m, um, h, f = None, q = None):
    #if (f is not None): print('---')
    #else: print ('xxx')
    size = len(xp)
    f = f if f is not None else np.identity(size)
    xi    = mprod_(f, xp.T)
    #print('f xp ', xi)
    q = np.zeros(size * size).reshape(size, size) if q is None else q
    #print(' qi ', q)
    uxi   = mprod_(f, mprod_(uxp, f.T)) + q
    ide = np.identity(size)
    #h2  = h2 if h2 is not None else 0. * h
    res = m - mprod_(h, xi.T) # - mprod_(h2, xi.T)
    #print('xi   ', xi.T)
    #print('h    ', h)
    #print('h xi ', mprod_(h, xi.T))
    #print('res  ', res)
    k   = np.matmul(mprod_(uxi, h.T), inv(mprod_(h, mprod_(uxi, h.T)) + um))
    #print('k ', k)
    x   = xi + mprod_(k, res.T)
    #print(' k res ', mprod_(k, res.T))
    #print('x ', x)
    ux = mprod_((ide - mprod_(k, h)), uxi)
    #print('cov ', cov)
    return x, ux, res, k

def _kfs(ms, ums, hs, x0, ux0 = None, fs = None, qs = None):
    nsample, msize, ssize = len(ms), len(ms[0]), len(x0)
    ux0 = 1e4 * np.identity(ssize)       if ux0 is None else ux0
    xs  = [x0                            for ii in range(nsample)]
    uxs = [np.identity(ssize)            for ii in range(nsample)]
    res = [np.zeros(msize)               for ii in range(nsample)]
    fs  = [np.identity(ssize) for ii in range(nsample)] if fs is None else fs
    qs  = [0.* fi for fi in fs]                         if qs is None else qs
    for i in range(nsample):
        xp, uxp = x0, ux0
        if (i >= 1):
            xp, uxp =  xs[i-1], uxs[i-1]
        xi, uxi, resi, _ = _kfi(xp, uxp, ms[i], ums[i], hs[i], f = fs[i], q = qs[i])
        xs[i], uxs[i], res[i] = xi, uxi, resi
    return xs, uxs, res


#------

def _delta(xs, type = float):
    #dxs[1:] = dxs[1:] - xs[:-1]
    #dxs = np.array(dxs, dtype = type)
    axs = np.array(xs)
    dxs = axs[1:] - axs[:-1]
    return dxs

def _delta_ms(cs, ufactor = 2., umin = 2.4):
    size = len(cs[0])
    ms  = _delta(cs)
    ums = [np.identity(size) * ufactor * np.maximum(np.sqrt(np.abs(ci)), umin) for ci in cs]
    return ms, ums

def _hs(ts, cs, N, hi_):
    dts = _delta(ts)
    hs = [hi_(ci, dt, N) for dt, ci in zip(dts, cs[:-1])]
    return hs

def _delta_kfs(ts, cs, x0, N, hi_, full_output = False):
    ms, ums      = _delta_ms(cs)
    hs           = _hs(ts, cs, N, hi_)
    xs, uxs, res = _kfs(ms, ums, hs, x0)
    result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
    return result

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


#
# Extende
#-----

def sirm_fi(si, xi, dt, N):
    f        = np.identity(4)
    s        = (N - np.sum(si))/N
    ni, nid  = si[0], xi[4]
    f        = np.identity(4)
    f[4, 1], f[4, 4]  = s * ni * dt, nid * dt
    return f

def sirm_hi(si, xi, dt, N):
    h        = np.zeros( 3 * 5).reshape(3, 5)
    s        = (N - np.sum(si))/N
    ni, nid  = si[0], xi[4]
    h[0, 0 ], h[0, 2] = s * ni, nid - ni
    h[1, 2]           =        -nid + ni
    h[2, 3]           =         nid
    return dt * h


def sirm_hv(s0 = (1, 0, 0), x0 = (0.6, 0.06, 0.2, 0.2), dt = 1, N = 1e6, nsamples = 100):
    ts, xs, ss = [], [], []
    for i in range(nsamples):
        ti = i * dt
        sip, xip = s0, x0 if i == 0 else xs[i-1], ss[i-1]
        fi = sirm_fi(sip, xip, dt, N)
        xi = mprod_(fi, x0.T)
        hi = sirm_hi(sip, xip, dt, N)
        si = mprod_(hi, x0.T)
        ss.append(si); xs.append(xi)
    return ts, xs, ss


#
#
#-------

def sirm_hi(ci, dt, N):
    size = 3
    ni, nr, nd = ci[0], ci[1], ci[2]
    h = np.zeros(size * size).reshape(size, size)
    si = N - np.sum(ci)
    h[0, 0], h[0, 1], h[0, 2] = si/N, -1., -1.
    h[1, 1]                   = 1.
    h[2, 2]                   = 1.
    h = h * dt * ni
    return h

def sirm_rvs(N, x0, s0 = np.array((1, 0, 0)), **kargs):
    return _hrvs(N, x0, s0, sirm_hi, **kargs)

def sirm_kf(ts, cs, x0, N, full_output = False, **kargs):
    return _delta_kfs(ts, cs, x0, N, sirm_hi, full_output, **kargs)
    #ms, ums      = delta_ms(cs)
    #hs           = hs_(ts, cs, N, sirm_hi)
    #xs, uxs, res = _kfs(ms, ums, hs, x0, N, **kargs)
    #result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
    #return result

#
# SIM Model
#

def sir_hi(ci, dt, N):
    size = 2
    ni, nr = ci[0], ci[1]
    h = np.zeros(size * size).reshape(size, size)
    si = N - np.sum(ci)
    h[0, 0], h[0, 1] = si/N, -1., -1
    h[1, 2]                   = 1.
    h = h * dt * ni
    return h

def sir_rvs(N, x0, s0 = np.array((1, 0, 0)), **kargs):
    return _hrvs(N, x0, s0, sirm_hi, **kargs)

def sir_kf(ts, cs, x0, N, full_output = False, **kargs):
    return _delta_kfs(ts, cs, x0, N, sir_hi, full_output, **kargs)
#    ms, ums      = _delta_ms(cs)
#    hs           = _hs(ts, cs, N, sir_hi)
#    xs, uxs, res = _kfs(ms, ums, hs, x0, N, **kargs)
#    result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
#    return result

#

#
#  SIR Model
#

# def sir_hi(ci, dt, N):
#     size = 2
#     ni, nr = ci[0], ci[1]
#     h = np.zeros(size * size).reshape(size, size)
#     si = N - np.sum(ci)
#     h[0, 0], h[0, 1] = si/N, -1.
#     h[1, 1]          = 1.
#     h = h * dt * ni
#     return h
#
# def sir_rvs(N, x0, **kargs):
#     ci = np.array((1, 0))
#     return _hrvs(N, x0, ci, sir_hi, **kargs)
#
# def sir_kf(ts, cs, x0, N, full_output = False, **kargs):
#     ms, ums      = delta_ms(cs)
#     hs           = hs_(ts, cs, N, sir_hi)
#     xs, uxs, res = _kfs(ms, ums, hs, x0, N, **kargs)
#     result = (xs, uxs, res, ms, ums, hs) if full_output else (xs, uxs)
#     return result

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

#
#  Chainned KF
#
#

def sirm_xhi(sp, dt, sf):
    ni, nr, nm = sp[0], sp[1], sp[2]
    h  = np.zeros(3 * 3).reshape(3, 3)
    h[0, 0], h[0, 1], h[0, 2] =  ni * sf, -ni, -ni
    h[1, 1]                   =  ni
    h[2, 2]                   =  ni
    h = h * dt
    return h

def sirm_sfi(xi, dt, sf):
    beta, gamma, rho = xi[0], xi[1], xi[2]
    f = np.zeros(3 * 3).reshape(3, 3)
    f[0, 0]  = sf * beta - gamma
    f[1, 0]  = gamma - rho
    f[2, 0]  = rho
    f        = np.identity(3) + f * dt
    return f

def _extrapolate(ts, ss, xs, N, sfi_):

    sps = []
    for i in range(1, len(ss)):
        dt, si, xi = ts[i] - ts[i-1], ss[i-1], xs[i-1]
        sf = (N - np.sum(si))/N
        sfi = sfi_(xi, dt, sf)
        spi = mprod_(sfi, si.T)
        #print('si ', si)
        #print('dt ', dt, 'xi', xi)
        #print('fi ', sfi)
        #print('spi', spi)
        sps.append(spi)
    return sps


def _delta_ckfs(ts, cs, x0, s0, N, xhi_, sfi_, sh = None, sigma = 100, **kargs):
    csize, ssize, xsize = len(cs[0]), len(s0), len(x0)
    #ucs     = np.random.poisson(np.sqrt(cs))
    ms, ums = _delta_ms(cs)
    ssize, xsize = len(s0), len(x0)
    shi = sh if sh is not None else np.identity(ssize)
    #hp  = np.zeros(2 * 3).reshape(2, 3)
    #hp[0, 0], hp[1, 2] = 1, 1
    ux0 = sigma * sigma * np.identity(xsize)
    us0 = sigma * sigma * np.identity(ssize)
    xs, uxs, xrs = [x0,], [ux0,], []
    ss, uss, srs = [s0,], [us0,], []
    #for i in range(1, 20): # len(ms)):
    for i in range(1, len(ms)):
        ci, uci = cs[i] , np.identity(csize) * np.sqrt(np.maximum(1, cs[i]))
        mi, umi = ms[i]  , ums[i-1]
        xp, uxp = xs[i-1], uxs[i-1]
        sp, usp = ss[i-1], uss[i-1]
        # Move first the samples
        dt      = ts[i] - ts[i-1]
        sf      = (N - np.sum(sp))/N
        sfi     = sfi_(xp, dt, sf)
        usp     = usp  * (np.sqrt(2) ** 2)
        si, usi, sri, _ = _kfi(sp, usp, ci, uci, shi, sfi)
        si      = np.maximum(si, 0.)
        si      = np.minimum(si, N)
        # Now the state
        # Use the true values
        # sp      = cp -
        dt      = ts[i] - ts[i-1]
        sf      = (N - np.sum(si))/N
        xhi     = mprod_(shi, xhi_(si, dt, sf))
        xi, uxi, xri, _ = _kfi(xp, uxp, mi, umi, xhi)
        xi      = np.maximum(xi, 0.)
        #print('ci ', ci)
        #print('sp ', sp)
        #print('si-p', mprod_(sfi, sp))
        #print('si  ', si)
        xs.append(xi); uxs.append(uxi); xrs.append(xri)
        ss.append(si); uss.append(usi); srs.append(sri)
    return (xs, uxs, xrs), (ss, uss, srs)
#
#


#
# def model_h(ci, dt, N):
#     h
#     P * h, (1- P)* h
#     return h
#
# def model_rvs()
#
#
# def _model_kfi(sp, xp, uxp, mi, umi):
#     h = model_hi(sp, dt, N):
#
#     si = h * xp
#
#
#
# def model_kf(ts, cs, c0, N, full_output = False, **kargs):
#     nsample = len(cs)
#     ms, ums = delta_ms(cs)
#     for i in range(1, nsample):
#         mi, umi     = ms[i], umi[i]
#         sp, xp, uxp = ss[i], xs[i], uxp[i]
#         si, xi, uxi, res = _model_kfi(xp, uxp, sp, mi, umi)
#
#
# #
# #
# #
#
#
# #---------
#
# #
# # def kf_hssir_ms(ts, cs, N):
# #
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
