
import numpy as np
import pandas as pd

import c19.kfilter as kf
import scipy.stats as stats
import c19.cfit    as cfit

mprod_ = np.matmul
npa    = np.array


import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.dates as mdates

#--- useir internal utilities

def ftheta(t0):
    def _fun(t):
        return np.array(t == t0, dtype = float)
    return _fun

def fpois(ti):
    return stats.poisson(ti).pmf

def fexpon(ti, ndays = 200):
    xp     = stats.expon(scale = ti).pdf
    ts     = np.arange(ndays)
    norma  = np.sum(xp(ts))
    return lambda x : xp(x)/norma

def fgamma(ti, ndays = 200):
    xp     = stats.gamma(ti).pdf
    ts     = np.arange(ndays)
    norma  = np.sum(xp(ts))
    return lambda x: xp(x)/norma

def funiform(ti, dti = 0, ndays = 200):
    if (dti <= 0 or dti > ti): dti = ti
    xp     = stats.uniform(ti - dti, ti + dti).pdf
    ts     = np.arange(ndays)
    norma  = np.sum(xp(ts))
    return lambda x: xp(x)/norma

def ftriang(ti, dti = 0, ndays = 200):
    if (dti <= 0 or dti > ti): dti = ti
    xp     = stats.triang(0.5, ti - dti, ti + dti).pdf
    ts     = np.arange(ndays)
    norma  = np.sum(xp(ts))
    return lambda x: xp(x)/norma

def fweibull(ti, c0 = 1.23, fi = 1., ndays = 200):
    xp     = stats.weibull_min(c0, scale = fi * ti ).pdf
    ts     = np.arange(ndays)
    norma  = np.sum(xp(ts))
    return lambda x: xp(x)/norma

def frho(rho = ''):

    _frho  = ftheta
    if   (rho == 'poisson'): _frho = fpois
    elif (rho == 'expon')  : _frho = fexpon
    elif (rho == 'gamma')  : _frho = fgamma
    elif (rho == 'uniform'): _frho = funiform
    elif (rho == 'triang') : _frho = ftriang
    elif (rho == 'weibull'): _frho = fweibull

    return _frho

def uV(vals, ts, rho):
    # convolution funciton: int_t' v(t') rho(t - t')
    #     vals are the vales of v(t'), and rho(t) is the pdf function
    rs = np.flip(rho(ts))
    v  = np.sum(vals * rs)
    return v

def mdeltas(v):
    """ return Delta V of the V -vector
    """
    m     = np.copy(v)
    m[1:] =  v[1:] - v[:-1]
    return m

def mrvs(dv, umin = 1e-5):
    """ return randon variable of DV vector with poisson mean DV
    """
    m   = stats.poisson(np.maximum(np.abs(dv), umin)).rvs()
    m[dv < 0] = -1 * m[dv < 0]
    return m

#--- useir models

def uSEIR(n, r0, ti, tr, tm, phim, ndays = 200, rho = 'theta', S0 = None, D0 = None):

    """ uSEIR model
    """

    def uDE(s, i, beta):
        return beta * s * i

    ts = np.arange(ndays)
    phir   = 1 - phim
    beta  = (r0/tr)

    S, DE, DI0    = np.zeros(ndays), np.zeros(ndays), np.zeros(ndays)
    E, R, M , I   = np.zeros(ndays), np.zeros(ndays), np.zeros(ndays), np.zeros(ndays)

    if (S0 is not None): print('S0 : ', S0)
    if (D0 is not None): print('D0 : ', D0)

    #TODO: convert S0, D0 into arrays to start the pandemic after some days
    S[0], DE[0], DI0[0] = (n, 1, 0) if S0 is None else S0 #(S0[0], S0[1], S0[2])
    R[0], M[0] , I[0]   = (0, 0, 0) if D0 is None else D0 #D0[0], D0[1], D0[2]

    _frho  = frho(rho)

    #print(str(_frho).split()[1])
    frhoi, frhor, frhom, = _frho(ti), _frho(tr), _frho(tm)

    for i in range(1, ndays):
        sp, ip = S[i-1], I[i-1]
        #print('Sp, Ip', sp, ip)
        de     = uDE(sp, ip, beta/n)
        DE [i] = de
        di0    =        uV(DE [:i+1], ts[:i+1], frhoi)
        DI0[i] = di0
        dr     = phir * uV(DI0[:i+1], ts[:i+1], frhor)
        dm     = phim * uV(DI0[:i+1], ts[:i+1], frhom)
        S [i]  = S[i-1] - de
        R [i]  = R[i-1] + dr
        M [i]  = M[i-1] + dm
        E [i]  = E[i-1] + de  - di0
        I [i]  = I[i-1] + di0 - dr - dm

    DR  = mdeltas(R)
    DM  = mdeltas(M)

    return (S, E, I, R, M), (DE, DI0, DR, DM)


def uSEIRq(n, r0, ti, tr, tm, phim, s1, r1, ndays = 200,
               rho = 'theta', S0 = None, D0 = None):
    """ uSEIR quenched model after a fraction S1 of supsected infividuals
              infection factor r0 -> r1 after s1
    """

    def uDE(s, i, beta):
        return beta * s * i

    ts = np.arange(ndays)
    phir   = 1 - phim
    beta  = (r0/tr)

    S, DE, DI0    = np.zeros(ndays), np.zeros(ndays), np.zeros(ndays)
    E, R, M , I   = np.zeros(ndays), np.zeros(ndays), np.zeros(ndays), np.zeros(ndays)

    if (S0 is not None): print('S0 : ', S0)
    if (D0 is not None): print('D0 : ', D0)

    S[0], DE[0], DI0[0] = (n, 1, 0) if S0 is None else S0 #(S0[0], S0[1], S0[2])
    R[0], M[0] , I[0]   = (0, 0, 0) if D0 is None else D0 #D0[0], D0[1], D0[2]

    _frho  = frho(rho)

    #print(str(_frho).split()[1])
    frhoi, frhor, frhom, = _frho(ti), _frho(tr), _frho(tm)

    for i in range(1, ndays):
        sp, ip = S[i-1], I[i-1]
        #print('Sp, Ip', sp, ip)
        if (1-sp/n) >= s1: beta = r1/tr
        de     = uDE(sp, ip, beta/n)
        DE [i] = de
        di0    =        uV(DE [:i+1], ts[:i+1], frhoi)
        DI0[i] = di0
        dr     = phir * uV(DI0[:i+1], ts[:i+1], frhor)
        dm     = phim * uV(DI0[:i+1], ts[:i+1], frhom)
        S [i]  = S[i-1] - de
        R [i]  = R[i-1] + dr
        M [i]  = M[i-1] + dm
        E [i]  = E[i-1] + de  - di0
        I [i]  = I[i-1] + di0 - dr - dm

    DR  = mdeltas(R)
    DM  = mdeltas(M)

    return (S, E, I, R, M), (DE, DI0, DR, DM)


# def uSEIRq(n, r0, ti, tr, tm, phim, t1, r1, ndays = 200,
#                rho = 'theta', S0 = None, D0 = None):
#
#     def uDE(s, i, beta):
#         return beta * s * i
#
#     ts = np.arange(ndays)
#     phir   = 1 - phim
#     beta  = (r0/tr)
#
#     S, DE, DI0    = np.zeros(ndays), np.zeros(ndays), np.zeros(ndays)
#     E, R, M , I   = np.zeros(ndays), np.zeros(ndays), np.zeros(ndays), np.zeros(ndays)
#
#     if (S0 is not None): print('S0 : ', S0)
#     if (D0 is not None): print('D0 : ', D0)
#
#     S[0], DE[0], DI0[0] = (n, 1, 0) if S0 is None else S0 #(S0[0], S0[1], S0[2])
#     R[0], M[0] , I[0]   = (0, 0, 0) if D0 is None else D0 #D0[0], D0[1], D0[2]
#
#     _frho  = frho(rho)
#
#     #print(str(_frho).split()[1])
#     frhoi, frhor, frhom, = _frho(ti), _frho(tr), _frho(tm)
#
#     for i in range(1, ndays):
#         sp, ip = S[i-1], I[i-1]
#         #print('Sp, Ip', sp, ip)
#         if (i >= t1): beta = r1/tr
#         de     = uDE(sp, ip, beta/n)
#         DE [i] = de
#         di0    =        uV(DE [:i+1], ts[:i+1], frhoi)
#         DI0[i] = di0
#         dr     = phir * uV(DI0[:i+1], ts[:i+1], frhor)
#         dm     = phim * uV(DI0[:i+1], ts[:i+1], frhom)
#         S [i]  = S[i-1] - de
#         R [i]  = R[i-1] + dr
#         M [i]  = M[i-1] + dm
#         E [i]  = E[i-1] + de  - di0
#         I [i]  = I[i-1] + di0 - dr - dm
#
#     DR  = mdeltas(R)
#     DM  = mdeltas(M)
#
#     return (S, E, I, R, M), (DE, DI0, DR, DM)
#
# # Data generation

#def mrvs(dv, umin = 1e-5, error = True):
#    m   = stats.poisson(np.maximum(np.abs(dv), umin)).rvs()
#    m[dv < 0] = -1 * m[dv < 0]
#    return m

# def errors(ds, d0 = 2.4):
#     return np.sqrt(np.maximum(ds, d0))

# def useir_rvdata(DI0, DR, DM):
#     """ Generate randon data from uSEIR (new infection, recovered and death)
#     """
#
#     dios  = mrvs(DI0)
#     drs   = mrvs(DR)
#     dms   = mrvs(DM)
#
#     return (dios, drs, dms)
#
    #-----

# def meas(dis, drs, dms, fi = 1., vmin = 2.4):
#     def _um(di, dr, dm):
#         u = np.identity(3)
#         u[0, 0] = fi * np.maximum(np.abs(di), vmin)
#         u[1, 1] = fi * np.maximum(np.abs(dr), vmin)
#         u[2, 2] = fi * np.maximum(np.abs(dm), vmin)
#         return u
#
#     ms  = [npa((di, dr, dm)) for di, dr, dm in zip(dis, drs, dms)]
#     ums = [_um( di, dr, dm)  for di, dr, dm in zip(dis, drs, dms)]
#     return ms, ums





# Projection hmatrix

#--- uSEIR KF utilites

def ninfecting(ts, dios, rhor):
    """ computes the number of infecting individuals given the new infections, dios,
    and the pdf of infected removal, rhor. The ts is the arrat of times to consider
    return the number of infected and the delta of removals
    """
    nsize = len(ts)
    drs = npa([uV(dios[:i+1], ts[:i+1], rhor) for i in range(nsize)])
    dis = dios - drs
    nis = npa([np.sum(dis[:i+1]) for i in range(nsize)])
    return nis, drs

def fhmatrix(frhoi, frhor, frhom):

    size = 3

    def _fun(dis, nis, ts):

        #nis, _ = nis_(dis, ts, frhor, frhom, phim)
        nsize = len(dis)
        #fis   = sis[:nsize]
        h = np.zeros(size * size).reshape(size, size)
        h[0, 0] = uV(nis[:-1], ts[:-1], frhoi)
        #h[0, 0] = uV(nis[:], ts[:], frhoi)
        h[1, 1] = uV(dis[:], ts[:], frhor)
        h[2, 2] = uV(dis[:], ts[:], frhom)
        return h
    return _fun


def hmatrices(ts, dios, nis, frhoi, frhor, frhom):
    ndays = len(dios)
    fh   = fhmatrix(frhoi, frhor, frhom)
    hs   = [fh(dios[:i+1], nis[:i+1], ts[:i+1]) for i in range(0, ndays)]
    #hs   = [fh(dios[:i], nis[:i], ts[:i]) for i in range(0, ndays)]
    #hs = [hs[0],] + hs
    #print(ndays, len(hs))
    return hs

# def betas(ts, xdios, rhor, rhoi):
#     xnis, xxds = ninfecting(ts, xdios, rhor)
#     nes   = npa([uV(xnis[0:i], ts[0:i], rhoi) for i in range(len(ts))])
#     betas = np.zeros(len(xdios))
#     xsel = nes > 0
#     betas[xsel] = xdios[xsel]/nes[xsel]
#     return betas
#
# def nis_(ts, dios, rhor, rhom, phim):
#     nsize = len(ts)
#     drs = npa([uV(dios[:i+1], ts[:i+1], rhor) for i in range(nsize)])
#     #drs = npa([0.,] + drs)
#     dms = npa([uV(dios[:i+1], ts[:i+1], rhom) for i in range(nsize)])
#     #dms = npa([0.,] + dms)
#     #print(len(dios), len(drs), len(dms))
#     dis = dios - (1-phim) * drs - phim * dms
#     nis = npa([np.sum(dis[:i+1]) for i in range(nsize)])
#     return nis, (dis, (1- phim) * drs, (phim)* dms)

# def fhmatrix(frhoi, frhor, frhom):
#
#     size = 3
#
#     def _fun(dis, nis, ts):
#
#         #nis, _ = nis_(dis, ts, frhor, frhom, phim)
#         nsize = len(dis)
#         #fis   = sis[:nsize]
#         h = np.zeros(size * size).reshape(size, size)
#         h[0, 0] = uV(nis[:-1], ts[:-1], frhoi)
#         #h[0, 0] = uV(nis[:], ts[:], frhoi)
#         h[1, 1] = uV(dis[:], ts[:], frhor)
#         h[2, 2] = uV(dis[:], ts[:], frhom)
#         return h
#     return _fun


# def fhmatrix(frhoi, frhor, frhom):
#
#     size = 3
#    # rhoi = frho(ti)
#    # rhor = frho(tr)
#    # rhom = frho(tm)
#
#     def _fun(dis, ts):
#         h = np.zeros(size * size).reshape(size, size)
#         h[0, 0] = uV(dis, ts, frhoi)
#         h[1, 1] = uV(dis, ts, frhor)
#         h[2, 2] = uV(dis, ts, frhom)
#         return h
#     return _fun


# def hmatrices(ts, dios, nis, frhoi, frhor, frhom):
#     ndays = len(dios)
#     fh   = fhmatrix(frhoi, frhor, frhom)
#     hs   = [fh(dios[:i+1], nis[:i+1], ts[:i+1]) for i in range(0, ndays)]
#     #hs   = [fh(dios[:i], nis[:i], ts[:i]) for i in range(0, ndays)]
#     #hs = [hs[0],] + hs
#     #print(ndays, len(hs))
#     return hs
#
# def plt_hmatrices(ts, hs):
#
#     plt.figure(figsize = (8, 6))
#     plt.plot([hi[0, 0] for hi in hs], label = 'infected')
#     plt.plot([hi[1, 1] for hi in hs], label = 'recovered')
#     plt.plot([hi[2, 2] for hi in hs], label = 'death')
#     plt.legend(); plt.grid(); plt.yscale('log'), plt.title('H-matrix elements')
#     return
#
# def plt_hmatrices2(ts, hs, ms):
#
#     hs = [npa([hi[i, i] for hi in hs]) for i in range(3)]
#     xs = [npa([mi[i]    for mi in ms]) for i in range(3)]
#
#     plt.figure(figsize = (8, 6))
#     plt.plot(hs[0], label = 'infected')
#     plt.plot(xs[0], ls = '', marker = 'o', label = 'infected')
#     plt.plot(hs[1], label = 'recovered')
#     plt.plot(xs[1], ls = '', marker = 'o', label = 'recovered')
#     plt.plot(hs[2], label = 'death')
#     plt.plot(xs[2], ls = '', marker = 'o', label = 'death')
#     plt.legend(); plt.grid(); plt.yscale('log');
#     plt.title('H-matrix elements')
#
#     plt.figure(figsize = (8, 6))
#     plt.plot(xs[0]/np.maximum(hs[0], 1.), label = 'infected')
#     plt.plot(xs[1]/np.maximum(hs[1], 1.), label = 'recovered')
#     plt.plot(xs[2]/np.maximum(hs[2], 1.), label = 'death')
#     plt.legend(); plt.grid(); plt.yscale('log');
#     plt.title('meas/prediction ratio');
#
#     return

#---- uSEIR KF main elements

def kfmeas(ds, uds = (None, None, None), vmin = 2.4, scale = False):
    dios, drs, dms = ds
    drs   = np.copy(dios) if drs is None else drs
    dms   = np.copy(dios) if dms is None else dms
    udios, udrs, udms = uds
    udios = np.maximum(dios, vmin) if udios is None else np.maximum(udios, vmin)
    udrs  = np.maximum(drs , vmin) if udrs  is None else np.maximum(udrs , vmin)
    udms  = np.maximum(dms , vmin) if udms  is None else np.maximum(udms , vmin)

    nscale = 1. if scale is False else np.sum(dios)
    ide = np.identity(3) * nscale * nscale
    ms  = [      npa((di, dr, dm))/nscale   for di, dr, dm in zip(dios , drs , dms)]
    ums = [ide * npa((1./di, 1./dr, 1./dm)) for di, dr, dm in zip(udios, udrs, udms)]

    return ms, ums


def kfhmatrices(ts, dios, times, sis = None, frho = fgamma, scale = False):
    nscale = np.sum(dios) if scale is True else 1.
    ti, tr, tm = times
    #nis, _  = nis_(ts, dios, frho(tr), frho(tm), phim)
    nis , _ = ninfecting(ts, dios/nscale, frho(tr))
    sis     = 1. if sis is None else sis
    hs      = hmatrices(ts,  dios/nscale,  sis * nis, frho(ti), frho(tr), frho(tm))
    return hs


def useir_kfs(ds, times, q0 = 0., uds = None, x0 = None, ux0 = None,
              sis = None, scale = False):
    """
    ds   : dios, drs, dms # new infected, recovered, deaths per day
    times: ti  ,  tr, tm  # time infection, recovery, death
    q0   :                # errors between steps
    x0   :                # guess parameters
    ux0  :                # guess uncentanties matrix
    phim : 0.1            # fraction of deaths
    sis  : 1.             # array of susceptibles (to fit beta-const)
    """
    dios, drs, dms = ds
    ti, tr, tm     = times
    nsize          = len(dios)

    uds            = (None, None, None) if uds is None else uds
    ms, ums        = kfmeas(ds, uds, scale = scale)
    ts             = np.arange(nsize)

    hs             = kfhmatrices(ts, dios, times, sis = sis, scale = scale)

    fs = [np.identity(3)      for i in range(nsize)]

    #if (len(q0) != nsize):
    q0 = [q0 for i in range(nsize)]
    qs = [qi * np.identity(3) for qi in q0]

    ks = [kf.KFnode(mi, umi, hi, fi, qi) for mi, umi, hi, fi, qi in zip(ms, ums, hs, fs, qs)]

    x0   = x0  if x0  is not None else npa((1., 0.8, 0.2))
    ux0  = ux0 if ux0 is not None else 1. * np.identity(3)
    ks = kf.kfilter(ks, x0, ux0)
    xm, uxm = [ki.xm for ki in ks], [ki.uxm for ki in ks]

    ks = kf.kfsmooth(ks)
    xs, uxs = [ki.xs for ki in ks], [ki.uxs for ki in ks]

    return (xs, uxs), (xm, uxm)


#------
#     KFS with data
#-------

#
# def useir_kfs_comomo(dates, cases, ucases, times,
#                     dates_blind = None, q0 = 1.):
#
#     nsize          = len(dates)
#     #t0             = int(tm)
#     #dios           = np.zeros(nsize)
#     #dios[:-t0]     = cases[t0:]
#     ti, td, tm     = times
#     ds             = (cases, cases, cases)
#     uds            = (ucases, ucases, ucases)
#
#     q0      = q0 * np.ones(nsize)
#
#     dates_blind = ('2030-01-01', '2030-12-31') if dates_blind is None else dates_blind
#     date0, date1 = dates_blind
#     sel    = (dates >= np.datetime64(date0)) & (dates <= np.datetime64(date1))
#     q0[sel] = 1.
#
#     kfres = useir_kfs(ds, times, uds = uds, q0 = q0, scale = True)
#
#     def _rs(xs, uxs):
#         rs  = td * npa([xi[0]             for xi in xs])
#         urs = td * npa([np.sqrt(xi[0, 0]) for xi in uxs])
#         return rs, urs
#
#     return _rs(*kfres[0]), _rs(*kfres[1])
#
# #-------
#
# def useir_kf(ms, ums, hs, x0, ux0, qs = None):
#     ndays = len(ms)
#     if (qs is None):
#         qs    = [np.identity(3) * 0. for i in range(ndays)]
#     xs, uxs, res = kf._kfs(ms, ums, hs, x0, ux0, qs = qs)
#     return xs, uxs, res
# #
# # def plt_useir_kf(ts, xs, uxs, res):
# #
# #     def _plot(rs, pr, pm, title):
# #         plt.figure(figsize = (8, 6))
# #         plt.title(title)
#         plt.grid();
#         plt.plot(ts, rs, c = 'black', label = r'$\beta$')
#         plt.ylabel('R')
#         plt.gca().legend(loc = 2);
#
#         ax2 = plt.gca().twinx()
#         ax2.grid(True)
#         ax2.plot(ts, pr, label = r'$\Phi_R$')
#         ax2.plot(ts, pm, label = r'$\Phi_M$')
#         ax2.set_ylabel(r'$\Phi$');
#         ax2.legend();
#
#     rs = [xi[0] for xi in xs]
#     pr = [xi[1] for xi in xs]
#     pm = [xi[2] for xi in xs]
#     _plot(rs, pr, pm, 'parameters')
#
#     rs = [np.sqrt(xi[0, 0]) for xi in uxs]
#     pr = [np.sqrt(xi[1, 1]) for xi in uxs]
#     pm = [np.sqrt(xi[2, 2]) for xi in uxs]
#     _plot(rs, pr, pm, 'uncertainties')
#
#     rs = [xi[0] for xi in res]
#     pr = [xi[1] for xi in res]
#     pm = [xi[2] for xi in res]
#     _plot(rs, pr, pm, 'residuals')
#
#     return
#

#----- useir LL fit


def _rv(dms):
    tbins  = np.arange(len(dms) + 1)
    #tbins  = _binedges(sir.t)
    irv = stats.rv_histogram((dms, tbins))
    return irv

def rvs(pars, ufun, size = 0):
    dms    = ufun(pars)
    rv     = _rv(dms)
    n0     = np.sum(dms)
    nbins  = len(dms)
    ni     = stats.poisson(n0).rvs(size = 1)
    size   = ni if size == 0 else size
    times  = rv.rvs(size = size)
    ys, xs = np.histogram(times, nbins+1, (0, nbins+1))
    res    = times, (xs[:-1], ys)
    return res

def fmodel(pars, ufun):
    dms    = ufun(pars)
    rv     = _rv(dms)
    ni     = np.sum(dms)
    def _fun(x):
        return ni * rv.pdf(x)
    return _fun

def ll(xs, ys, ufun, pars = None):

    # data is a table (days, individuals)
    #xs, ys = data
    nx     = len(xs)
    n0     = np.sum(ys)

    def _fun(pars):
        dms = ufun(pars)
        rv  = _rv(dms)
        ll  = npa([-2 * yi * rv.logpdf(xi) for xi, yi in zip (xs, ys)])
        ll  = np.nan_to_num(ll)
        ni  = np.sum(dms)
        lp  = -2 * stats.poisson(ni).logpmf(n0) / nx
        lp = 0.
        return ll + lp

    res = _fun if pars is None else _fun(pars)
    return res

def res(xs, ys, ufun, pars = None, yerr = None, sqr = True):

    # data is a table (days, individuals)
    #xs, ys = data
    yerr   = np.maximum(np.sqrt(ys), 1.) if yerr is None else yerr
    n      = np.sum(ys)

    def _fun(pars):
        dms = ufun(pars)
        rv  = _rv(dms)
        ni  = np.sum(dms)
        ds  = npa([yi - ni * rv.pdf(xi) for xi, yi in zip(xs, ys)])
        ds  = npa([ds/ye                for ds, ye in zip(ds, yerr)])
        if (sqr): ds = ds * ds
        return ds

    res = _fun if pars is None else _fun(pars)
    return res


def chi2(xs, ys, pars, ufun, yerr = None):
    return np.sum(res(xs, ys, pars, ufun = ufun, yerr = yerr))


def mll(xs, ys, pars, ufun):
    return np.sum(ll(xs, ys, pars, ufun = ufun))


#------ fit-models

rho = 'weibull'
ndays = 200

def _t0(pars, ufun):
    # this deplaces the absolute time of the pandemic
    # TODO : make an interpolation of the pandemic and move dt0 - float not itn!
    dt0, tpars = int(pars[0]), pars[1:]
    dms = ufun(tpars)
    if (dt0 <= 0): return dms
    dms[:-dt0] = dms[dt0:]
    dms[-dt0:] = 0.
    return dms

def _t0(pars, ufun):
    # this deplaces the absolute time of the pandemic
    # TODO : make an interpolation of the pandemic and move dt0 - float not itn!
    dt0, tpars = pars[0], pars[1:]
    dms = ufun(tpars)
    ts  = np.arange(len(dms))
    dmsp = np.interp(ts + dt0, ts, dms)
    return dmsp

def dms_useir(pars, ndays = ndays, rho = rho):

    beta, ti, tr, tm, n, phim = pars
    r0      = beta * tr

    ns, ds = uSEIR(n, r0, ti, tr, tm, phim, ndays = ndays, rho = rho)
    return ds[3]


def dms_useirq(pars, fname = 'weibull'):

    beta, gamma, ti, tr, tm, n, phim, s1 = pars

    #factor = 0.041
    r0, r1   = beta * tr, gamma * tr
    tm       = tr
    # TODO pass the rest of arguments

    ndays           = 200
    srho            = fname

    #print(n, r0, ti, tr, tm, phim, s1, r1, ndays, rho)
    ns, ds = uSEIRq(n, r0, ti, tr, tm, phim, s1, r1, ndays = ndays, rho = rho)
    return ds[3]

def dms_t0useir(pars, ndays = ndays, rho = rho):
    return _t0(pars, dms_useir)


def dms_t0useirq(pars, ndays = ndays, rho = rho):
    return _t0(pars, dms_useirq)

def dms_fit(ts, cases, ufun, pars, pmask,
            ucases = None, ffit = 'chi2'):

    _fun   = ll if ffit == 'mll' else res
    fun    = _fun(ts, cases, ufun)

    xres   = cfit.minimize(pars, fun, mask = pmask, method = 'Nelder-Mead')

    xfun = lambda pars: np.sum(fun(pars))
    return xres, xfun(xres), xfun



#
# def _useirext(pars, fname = 'gamma'):
#
#     beta, tr, ti, n, phim = pars
#     #r0, tr  = pars
#     tm      = tr
#     r0      = beta * tm
#
#     ndays           = 200
#     rho             = fname
#
#     #print(n, r0, ti, tr, tm, phim, rho, ndays)
#     ns, ds = uSEIR(n, r0, ti, tr, tm, phim, ndays = ndays, rho = rho)
#     return ds[3]
#
#
#
# def _useirvarextmfix(pars, fname = 'gamma'):
#
#     beta, gamma, tr, ti, tm, n, phim, s1 = pars
#
#     tm = tr
#     #factor = 0.041
#     r0, r1   = beta * tr, gamma * tr
#     #tm       = tr
#     # TODO pass the rest of arguments
#
#     ndays           = 200
#     srho            = fname
#
#     ns, ds = uSEIR_Rvar(n, r0, ti, tr, tm, phim, s1, r1, ndays = ndays, rho = srho)
#     return ds[3]
#
#
#
# def _useirqr(pars, fname = 'gamma'):
#
#     beta, gamma, tr, ti, n, phim, s1 = pars
#
#     #factor = 0.041
#     r0, r1   = beta * tr, gamma * tr
#     tm       = tr
#     # TODO pass the rest of arguments
#
#     ndays    = 200
#     srho     = fname
#
#     ns, ds = uSEIR_Rvar(n, r0, ti, tr, tm, phim, s1, r1, ndays = ndays, rho = srho)
#     return ds[3]
#
#
#
# def _useirqm(pars, fname = 'gamma'):
#
#     beta, gamma, tr, ti, tm, n, phim, s1 = pars
#
#     #factor = 0.041
#     r0, r1   = beta * tr, gamma * tr
#     #tm       = tr
#     # TODO pass the rest of arguments
#
#     ndays           = 200
#     srho            = fname
#
#     ns, ds = uSEIR_Rvar(n, r0, ti, tr, tm, phim, s1, r1, ndays = ndays, rho = srho)
#     return ds[3]
#
#
# def _useirvarext(pars, fname = 'gamma'):
#
#     beta, gamma, tr, ti, tm, n, phim, s1 = pars
#
#     #factor = 0.041
#     r0, r1   = beta * tr, gamma * tr
#     #tm       = tr
#     # TODO pass the rest of arguments
#
#     ndays           = 200
#     srho            = fname
#
#     ns, ds = uSEIR_Rvar(n, r0, ti, tr, tm, phim, s1, r1, ndays = ndays, rho = srho)
#     return ds[3]
#
# def _useir(pars, args = None):
#
#     r0, tr  = pars
#     tm      = tr
#
#     # TODO pass the rest of arguments
#     N               = 1e6
#     R0, TI, TR, TM  = 3., 5., 5., 5.
#     PhiM            = 0.01
#     ndays           = 200
#     rho             = fname
#
#     r0, tr  = pars
#     tm      = tr
#     #print(N, r0, TI, tr, tm, PhiM, rho, ndays)
#     ns, ds = uSEIR(N, r0, TI, tr, tm, PhiM, ndays = ndays, rho = rho)
#     return ds[3]
#
# def _useirvar(pars, args = None):
#     factor = 1. # 0.041
#     # TODO pass the rest of arguments
#     n, ndays        = 3e6 * factor, 200
#     s1              = 0.05
#     r0, r1          = 3., 0.8
#     ti, tr, tm      = 5, 5, 5
#     phim            = 0.03
#     srho            = fname
#
#     r0, r1, tr  = pars
#     tm          = tr
#     ns, ds = uSEIR_Rvar(n, r0, ti, tr, tm, phim, s1, r1, ndays = ndays, rho = srho)
#     return ds[3]
#
#
# def _t0(pars, ufun = _useir):
#     dt0, tpars = int(pars[0]), pars[1:]
#     dms = ufun(tpars)
#     if (dt0 <= 0): return dms
#     dms[:-dt0] = dms[dt0:]
#     dms[-dt0:] = 0.
#     return dms
#
#
# def _rv(dms):
#     tbins  = np.arange(len(dms) + 1)
#     #tbins  = _binedges(sir.t)
#     irv = stats.rv_histogram((dms, tbins))
#     return irv
#
#
# def rvs(pars, size = 0, ufun = _useir):
#     dms    = ufun(pars)
#     rv     = _rv(dms)
#     n0     = np.sum(dms)
#     nbins  = len(dms)
#     ni     = stats.poisson(n0).rvs(size = 1)
#     size   = ni if size == 0 else size
#     times  = rv.rvs(size = size)
#     ys, xs = np.histogram(times, nbins+1, (0, nbins+1))
#     res    = times, (xs[:-1], ys)
#     return res
#
# def fmodel(pars, ufun = _useir):
#     dms    = ufun(pars)
#     rv     = _rv(dms)
#     ni     = np.sum(dms)
#     def _fun(x):
#         return ni * rv.pdf(x)
#     return _fun
#
# def mll(data, pars = None, ufun = _useir):
#
#     # data is a table (days, individuals)
#     xs, ys = data
#     nx     = len(xs)
#     n0     = np.sum(ys)
#
#     def _fun(pars):
#         dms = ufun(pars)
#         rv  = _rv(dms)
#         ll  = npa([-2 * yi * rv.logpdf(xi) for xi, yi in zip (xs, ys)])
#         ll  = np.nan_to_num(ll)
#         ni  = np.sum(dms)
#         lp  = -2 * stats.poisson(ni).logpmf(n0) / nx
#         lp = 0.
#         return ll + lp
#
#     res = _fun if pars is None else _fun(pars)
#     return res
#
# def res(data, pars = None, ufun = _useir, sqr = True):
#
#     # data is a table (days, individuals)
#     xs, ys = data
#     yerr   = np.maximum(np.sqrt(ys), 1.)
#     n      = np.sum(ys)
#
#     def _fun(pars):
#         dms = ufun(pars)
#         rv  = _rv(dms)
#         ni  = np.sum(dms)
#         ds  = npa([yi - ni * rv.pdf(xi) for xi, yi in zip(xs, ys)])
#         ds  = npa([ds/ye                for ds, ye in zip(ds, yerr)])
#         if (sqr): ds = ds * ds
#         return ds
#
#     res = _fun if pars is None else _fun(pars)
#     return res
#
#
# def chi2(data, pars, ufun = _useir):
#     return np.sum(res(data, pars, ufun = ufun))
#
# def mle(data, pars, ufun = _useir):
#     return np.sum(mll(data, pars, ufun = ufun))
#
# #----------
