
import numpy as np
import pandas as pd

import c19.kfmysir as kfmsir
import scipy.stats as stats

mprod_ = np.matmul
npa    = np.array


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

def funiform(ti, dti = 2, ndays = 200):
    if (dti == 0 or dti > ti): dti = ti
    xp     = stats.uniform(ti - dti, ti + dti).pdf
    ts     = np.arange(ndays)
    norma  = np.sum(xp(ts))
    return lambda x: xp(x)/norma

def ftriang(ti, dti = 2, ndays = 200):
    if (dti == 0 or dti > ti): dti = ti
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
    rs = np.flip(rho(ts))
    v  = np.sum(vals * rs)
    return v


def uSEIR(n, r0, ti, tr, tm, phim, ndays = 200, rho = 'theta'):

    def uDE(s, i, beta):
        return beta * s * i

    ts = np.arange(ndays)
    phir   = 1 - phim
    beta  = (r0/tr)

    S, DE, DI0    = np.zeros(ndays), np.zeros(ndays), np.zeros(ndays)
    E, R, M , I   = np.zeros(ndays), np.zeros(ndays), np.zeros(ndays), np.zeros(ndays)

    S[0], DE[0], DI0[0] = n, 1, 0
    R[0], M[0] , I[0]   = 0, 0, 0

    _frho  = frho(rho)

    print(_frho)
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
        E [i]  = E[i-1] + de - di0
        I [i]  = I[i-1] + di0 - dr - dm

    return (S, E, I, R, M)

def huseir(vs, ts, k, t0, frho = ftheta, dt0 = 1):
    frhot = frhot(t0)
    m = k * uV(vs, ts, frhot)

    tmin, tmax = max(t0 - dt0, 0), t0 + dt0
    frhot1 = frho(tmax)
    frhot0 = frho(tmin)
    m1 = k * uV(vs, ts, frho1)
    m0 = k * uV(vs, ts, frho0)
    h = np.array((m/k, (m1-m0)/(tmax - tmin)))
    return m, h
    #print(' i ', i)
