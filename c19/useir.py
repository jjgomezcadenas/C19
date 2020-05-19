
import numpy as np
import pandas as pd

import c19.kfmysir as kf
import scipy.stats as stats

mprod_ = np.matmul
npa    = np.array


import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.dates as mdates

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

# def funiform(ti, dti = 2, ndays = 200):
#     if (dti == 0 or dti > ti): dti = ti
#     xp     = stats.uniform(ti - dti, ti + dti).pdf
#     ts     = np.arange(ndays)
#     norma  = np.sum(xp(ts))
#     return lambda x: xp(x)/norma
#
# def ftriang(ti, dti = 2, ndays = 200):
#     if (dti == 0 or dti > ti): dti = ti
#     xp     = stats.triang(0.5, ti - dti, ti + dti).pdf
#     ts     = np.arange(ndays)
#     norma  = np.sum(xp(ts))
#     return lambda x: xp(x)/norma

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

def frho(rho = ''):

    _frho  = ftheta
    if   (rho == 'poisson'): _frho = fpois
    elif (rho == 'expon')  : _frho = fexpon
    elif (rho == 'gamma')  : _frho = fgamma
    elif (rho == 'uniform'): _frho = funiform
    elif (rho == 'triang') : _frho = ftriang

    return _frho

def uV(vals, ts, rho):
    # convolution funciton: int_t' v(t') rho(t - t')
    #     vals are the vales of v(t'), and rho(t) is the pdf function
    rs = np.flip(rho(ts))
    v  = np.sum(vals * rs)
    return v


def uSEIR(n, r0, ti, tr, tm, phim, ndays = 200, rho = 'theta', S0 = None, D0 = None):

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

    print(str(_frho).split()[1])
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


def uSEIR_Rvar(n, r0, ti, tr, tm, phim, s1, r1, ndays = 200,
               rho = 'theta', S0 = None, D0 = None):

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

    print(str(_frho).split()[1])
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


def plt_uSEIR(ts, seir, dseir, title = '', yscale = 'log'):
    S, E, I, R, M   = seir
    DE, DI0, DR, DM = dseir

    plt.figure(figsize = (8, 6))
    plt.plot(ts, S, label = 'susceptible')
    plt.plot(ts, E, label = 'exposed')
    plt.plot(ts, I, label = 'infected')
    plt.plot(ts, R, label = 'recovered')
    plt.plot(ts, M, label = 'death')
    plt.ylim((1, 1.5 * np.max(S)))
    plt.grid(which = 'both'); plt.legend(); plt.title(title); plt.yscale(yscale)

    plt.figure(figsize = (8, 6))
    plt.plot(ts, DE , label = r'$\Delta E$')
    plt.plot(ts, DI0, label = r'$\Delta I_0$')
    plt.plot(ts, DM, label = r'$\Delta M$')
    plt.plot(ts, DR, label = r'$\Delta R$')
    plt.ylim((1., 1.5 * np.max(DI0)))
    plt.grid(which = 'both'); plt.legend(); plt.title(title); plt.yscale(yscale)

    return

# Data preparation

def mdeltas(v):
    m     = np.copy(v)
    m[1:] =  v[1:] - v[:-1]
    return m

def mrvs(dv, umin = 1e-5, error = True):
    m   = stats.poisson(np.maximum(np.abs(dv), umin)).rvs()
    m[dv < 0] = -1 * m[dv < 0]
    return m

def errors(ds, d0 = 2.4):
    return np.sqrt(np.maximum(ds, d0))

def meas(dis, drs, dms, fi = 1.):
    def _um(di, dr, dm):
        u = np.identity(3)
        u[0, 0] = fi * np.maximum(np.abs(di), 2.4)
        u[1, 1] = fi * np.maximum(np.abs(dr), 2.4)
        u[2, 2] = fi * np.maximum(np.abs(dm), 2.4)
        return u

    ms  = [npa((di, dr, dm)) for di, dr, dm in zip(dis, drs, dms)]
    ums = [_um( di, dr, dm)  for di, dr, dm in zip(dis, drs, dms)]
    return ms, ums

def plt_meas(ts, ms, ums, yscale = 'log'):

    def _plot(ts, ris, rrs, rms, title ):
        plt.figure(figsize = (8, 6))
        plt.plot(ts, ris , ls = '--', marker = 'o', label = 'infected')
        plt.plot(ts, rrs , ls = '--', marker = 'o', label = 'recovered')
        plt.plot(ts, rms , ls = '--', marker = 'o', label = 'death')
        plt.grid(which = 'both'); plt.yscale(yscale); plt.legend()
        plt.xlabel('days'); plt.ylabel('individuals')
        return

    ris = [xi[0] for xi in ms]
    rrs = [xi[1] for xi in ms]
    rms = [xi[2] for xi in ms]
    _plot(ts, ris, rrs, rms, 'measurements')

    ris = [np.sqrt(xi[0, 0]) for xi in ums]
    rrs = [np.sqrt(xi[1, 1]) for xi in ums]
    rms = [np.sqrt(xi[2, 2]) for xi in ums]
    _plot(ts, ris, rrs, rms, 'uncertainties')
    return

# Data generation

def useir_rvdata(DI0, DR, DM):

    dios  = mrvs(DI0)
    drs   = mrvs(DR)
    dms   = mrvs(DM)

    return (dios, drs, dms)

def plt_useir_rvdata(ts, DS, ds):

    DI0 , DR, DM   = DS
    dios, drs, dms = ds

    plt.figure(figsize = (8, 6))
    plt.plot(ts[:], DI0, label = 'infected')
    plt.plot(ts[:], DR , label = 'recovered')
    plt.plot(ts[:], DM , label = 'death')

    plt.plot(ts[:], dios, label = 'infected', ls = '', marker = 'o')
    plt.plot(ts[:], drs , label = 'recovered', ls = '', marker = 'o')
    plt.plot(ts[:], dms , label = 'death'    , ls = '', marker = 'o')
    plt.grid(which = 'both'); plt.legend();
    return


# Projection hmatrix

def ninfecting(ts, dios, rhor):
    nsize = len(ts)
    drs = npa([uV(dios[:i+1], ts[:i+1], rhor) for i in range(nsize)])
    dis = dios - drs
    nis = npa([np.sum(dis[:i+1]) for i in range(nsize)])
    return nis, drs

def betas(ts, xdios, rhor, rhoi):
    xnis, xxds = ninfecting(ts, xdios, rhor)
    nes   = npa([uV(xnis[0:i], ts[0:i], rhoi) for i in range(len(ts))])
    betas = np.zeros(len(xdios))
    xsel = nes > 0
    betas[xsel] = xdios[xsel]/nes[xsel]
    return betas

def nis_(ts, dios, rhor, rhom, phim):
    nsize = len(ts)
    drs = npa([uV(dios[:i+1], ts[:i+1], rhor) for i in range(nsize)])
    #drs = npa([0.,] + drs)
    dms = npa([uV(dios[:i+1], ts[:i+1], rhom) for i in range(nsize)])
    #dms = npa([0.,] + dms)
    #print(len(dios), len(drs), len(dms))
    dis = dios - (1-phim) * drs - phim * dms
    nis = npa([np.sum(dis[:i+1]) for i in range(nsize)])
    return nis, (dis, (1- phim) * drs, (phim)* dms)

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


def hmatrices(ts, dios, nis, frhoi, frhor, frhom):
    ndays = len(dios)
    fh   = fhmatrix(frhoi, frhor, frhom)
    hs   = [fh(dios[:i+1], nis[:i+1], ts[:i+1]) for i in range(0, ndays)]
    #hs   = [fh(dios[:i], nis[:i], ts[:i]) for i in range(0, ndays)]
    #hs = [hs[0],] + hs
    #print(ndays, len(hs))
    return hs

def plt_hmatrices(ts, hs):

    plt.figure(figsize = (8, 6))
    plt.plot([hi[0, 0] for hi in hs], label = 'infected')
    plt.plot([hi[1, 1] for hi in hs], label = 'recovered')
    plt.plot([hi[2, 2] for hi in hs], label = 'death')
    plt.legend(); plt.grid(); plt.yscale('log'), plt.title('H-matrix elements')
    return

def plt_hmatrices2(ts, hs, ms):

    hs = [npa([hi[i, i] for hi in hs]) for i in range(3)]
    xs = [npa([mi[i]    for mi in ms]) for i in range(3)]

    plt.figure(figsize = (8, 6))
    plt.plot(hs[0], label = 'infected')
    plt.plot(xs[0], ls = '', marker = 'o', label = 'infected')
    plt.plot(hs[1], label = 'recovered')
    plt.plot(xs[1], ls = '', marker = 'o', label = 'recovered')
    plt.plot(hs[2], label = 'death')
    plt.plot(xs[2], ls = '', marker = 'o', label = 'death')
    plt.legend(); plt.grid(); plt.yscale('log');
    plt.title('H-matrix elements')

    plt.figure(figsize = (8, 6))
    plt.plot(xs[0]/np.maximum(hs[0], 1.), label = 'infected')
    plt.plot(xs[1]/np.maximum(hs[1], 1.), label = 'recovered')
    plt.plot(xs[2]/np.maximum(hs[2], 1.), label = 'death')
    plt.legend(); plt.grid(); plt.yscale('log');
    plt.title('meas/prediction ratio');

    return

#---- KF


def useir_kf(ms, ums, hs, x0, ux0, qs = None):
    ndays = len(ms)
    if (qs is None):
        qs    = [np.identity(3) * 0. for i in range(ndays)]
    xs, uxs, res = kf._kfs(ms, ums, hs, x0, ux0, qs = qs)
    return xs, uxs, res

def plt_useir_kf(ts, xs, uxs, res):

    def _plot(rs, pr, pm, title):
        plt.figure(figsize = (8, 6))
        plt.title(title)
        plt.grid();
        plt.plot(ts, rs, c = 'black', label = r'$\beta$')
        plt.ylabel('R')
        plt.gca().legend(loc = 2);

        ax2 = plt.gca().twinx()
        ax2.grid(True)
        ax2.plot(ts, pr, label = r'$\Phi_R$')
        ax2.plot(ts, pm, label = r'$\Phi_M$')
        ax2.set_ylabel(r'$\Phi$');
        ax2.legend();

    rs = [xi[0] for xi in xs]
    pr = [xi[1] for xi in xs]
    pm = [xi[2] for xi in xs]
    _plot(rs, pr, pm, 'parameters')

    rs = [np.sqrt(xi[0, 0]) for xi in uxs]
    pr = [np.sqrt(xi[1, 1]) for xi in uxs]
    pm = [np.sqrt(xi[2, 2]) for xi in uxs]
    _plot(rs, pr, pm, 'uncertainties')

    rs = [xi[0] for xi in res]
    pr = [xi[1] for xi in res]
    pm = [xi[2] for xi in res]
    _plot(rs, pr, pm, 'residuals')

    return
