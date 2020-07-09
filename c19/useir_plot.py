import numpy as np
import pandas as pd

from copy import copy

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.dates as mdates

import c19.useir as us
import c19.cfit  as cfit
from   c19.HG_analysis import formatter

npa    = np.array

def plt_uSEIR(ts, seir, dseir, title = '', yscale = 'log', norma = False):
    """ Plot uSEIR categories and increment vs days
    """

    S, E, I, R, M   = seir
    N = np.max(S) if norma == True else 1.
    DE, DI0, DR, DM = dseir

    plt.figure(figsize = (8, 6))
    plt.plot(ts, S/N, label = 'susceptible')
    plt.plot(ts, E/N, label = 'exposed')
    plt.plot(ts, I/N, label = 'infected')
    plt.plot(ts, R/N, label = 'recovered')
    plt.plot(ts, M/N, label = 'death')
    ylim = (1./N, 1.5) if norma else (1., 1.5 * N)
    plt.ylim(ylim)
    plt.xlabel('days'); plt.ylabel('individuals')
    plt.grid(which = 'both'); plt.legend();
    plt.title('uSEIR categories '+ title); plt.yscale(yscale)

    plt.figure(figsize = (8, 6))
    plt.plot(ts, DE/N , label = r'$\Delta E$')
    plt.plot(ts, DI0/N, label = r'$\Delta I_0$')
    plt.plot(ts, DM/N, label = r'$\Delta M$')
    plt.plot(ts, DR/N, label = r'$\Delta R$')
    ylim = (1./N, 1.5 * np.max(DI0)/N) if norma else (1., 1.5 * np.max(DI0))
    plt.ylim(ylim)
    plt.xlabel('days'); plt.ylabel('individuals')
    plt.grid(which = 'both'); plt.legend();
    plt.title('uSEIR categories increase '+ title); plt.yscale(yscale)
    return


# def plt_useir_rvdata(ts, DS, ds):
#
#     DI0 , DR, DM   = DS
#     dios, drs, dms = ds
#
#     plt.figure(figsize = (8, 6))
#     plt.plot(ts[:], DI0, label = 'infected')
#     plt.plot(ts[:], DR , label = 'recovered')
#     plt.plot(ts[:], DM , label = 'death')
#
#     plt.plot(ts[:], dios, label = 'infected', ls = '', marker = 'o')
#     plt.plot(ts[:], drs , label = 'recovered', ls = '', marker = 'o')
#     plt.plot(ts[:], dms , label = 'death'    , ls = '', marker = 'o')
#     plt.grid(which = 'both'); plt.legend();
#     return


#---- KF plots

def plt_kfmeas(ts, ms, ums, yscale = 'log'):

    def _plot(ts, ris, rrs, rms, title):
        plt.figure(figsize = (8, 6))
        plt.plot(ts, ris , ls = '--', marker = 'o', label = 'infected')
        plt.plot(ts, rrs , ls = '--', marker = 'o', label = 'recovered')
        plt.plot(ts, rms , ls = '--', marker = 'o', label = 'death')
        plt.grid(which = 'both'); plt.yscale(yscale); plt.legend()
        plt.xlabel('days'); plt.ylabel('individuals')
        plt.title(title)
        return

    ris = [xi[0] for xi in ms]
    rrs = [xi[1] for xi in ms]
    rms = [xi[2] for xi in ms]
    #_plot(ts, ris, rrs, rms, 'measurements')

    uris = [np.sqrt(1./xi[0, 0]) for xi in ums]
    urrs = [np.sqrt(1./xi[1, 1]) for xi in ums]
    urms = [np.sqrt(1./xi[2, 2]) for xi in ums]

    plt.figure(figsize = (8, 6))
    plt.errorbar(ts, ris , yerr = uris, ls = '', marker = 'o', label = 'infected')
    plt.errorbar(ts, rrs , yerr = urrs, ls = '', marker = 'o', label = 'recovered')
    plt.errorbar(ts, rms , yerr = urms, ls = '', marker = 'o', label = 'death')
    plt.grid(which = 'both'); plt.yscale(yscale); plt.legend()
    plt.xlabel('days'); plt.ylabel('individuals')
    plt.title('kf measurements')

    #_plot(ts, ris, rrs, rms, 'uncertainties')
    return

def plt_kfhmatrices(ts, hs):

    plt.figure(figsize = (8, 6))
    plt.plot([hi[0, 0] for hi in hs], label = 'infected')
    plt.plot([hi[1, 1] for hi in hs], label = 'recovered')
    plt.plot([hi[2, 2] for hi in hs], label = 'death')
    plt.legend(); plt.grid(); plt.yscale('log'), plt.title('H-matrix elements')
    return

def plt_kfhmatrices_check(ts, hs, ms):

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

def plt_kfs(xs, uxs, labels = (r'$\beta$', r'$\Phi_R$', r'$\Phi_M$')):
    nsize = len(xs)
    msize = len(xs[0])
    ts    = np.arange(nsize)

    plt.figure(figsize = (8, 6))
    for k in range(msize):
        rs  = npa([xi[k]              for xi  in xs])
        urs = npa([np.sqrt(uxi[k, k]) for uxi in uxs])
        #print(rs)
        plt.errorbar(ts, rs, yerr = urs, ls = '', marker = 'o', ms = 4,label = labels[k])
    plt.grid(which = 'both');
    plt.legend()


def plt_kfpars_time_evolution(xms, xss):
    xm, uxm = xms
    plt_kfs(xm, uxm); plt.title('filter'); plt.ylim((0., 1.5))
    xs, uxs = xss
    plt_kfs(xs, uxs); plt.title('smooth'); plt.ylim((0., 1.5));


def plt_beta_time_evolution(xss, td, beta0, sn):
    xs, uxs = xss
    ts      = np.arange(len(xs))

    betas = [xi[0] for xi in xs]
    ubetas = [np.sqrt(xi[0, 0]) for xi in uxs]
    rr    = beta0 * sn

    plt.figure(figsize = (8, 6))
    plt.plot(ts + td, beta0 * sn, color = 'black', lw = 2, label = r'$\beta_{True}(t)$')
    plt.errorbar(ts, betas, yerr = ubetas, ls = '', marker = 'o',
                 color = 'blue', alpha = 0.5, label = r"$\beta(t)$");
    plt.grid(); plt.ylim((0., 0.8)); plt.legend(fontsize = 12)

    return betas, ubetas, rr


def plt_betaq_time_evolution(xss, td, beta0, beta1, sn, s1):

    xs, uxs = xss
    ts      = np.arange(len(xs))

    betas  = [xi[0] for xi in xs]
    ubetas = [np.sqrt(xi[0, 0]) for xi in uxs]

    sel = sn > 1 - s1
    rr  = np.zeros(len(xs))
    rr[ sel] = beta0 * sn[ sel]
    rr[~sel] = beta1 * sn[~sel]

    plt.figure(figsize = (8, 6))
    plt.plot(ts + td, rr, lw = 2, color = 'black', label = r'$\beta_{True}(t)$')
    plt.errorbar(ts, betas, yerr = ubetas, ls = '', marker = 'o', ms = 6,
                 color = 'blue', alpha = 0.5, label = r"$\beta(t)$");
    plt.grid(); plt.ylim((0., 0.8)); plt.legend();

    return betas, ubetas, rr


#---- Fit plots

def plt_data_model(xs, ys, pars, ufun, yerr = None):

    if (type(pars) == dict):
        pars = us.kpars_to_pars(pars, ufun)
    #print(pars)

    isdate = False
    ts     = copy(xs)
    if (type(xs[0]) == np.datetime64):
        isdate = True
        ts = us.to_ts(xs)
    #print(ts)

    f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios': [6, 1]})
    #plt.subplot(2, 1, 1)
    yerr   = np.sqrt(ys) if yerr is None else yerr
    ax1.errorbar(xs, ys, yerr = np.sqrt(ys), ls = '', label = 'data',
                 marker = 'o', ms = 4, c = 'black');
    fun    = us.fmodel(pars, ufun)
    ax1.plot(xs, fun(ts), ls = '--', c = 'blue', label = 'model')
    ax1.grid(); ax1.legend();
    if isdate: formatter(ax1);

    #plt.subplot(2, 1, 2)
    fres  = us.res(ts, ys, ufun, sqr = False)
    #ax1 = plt.twinx()
    ax2.bar(xs, fres(pars));
    ax2.set_ylim((-5., 5.));
    ax2.set_ylabel(r"$\sigma$");
    if isdate: formatter(ax2);

    f.tight_layout()
    return

def plt_ffit_scan(fun, pars, vals0, vals1,
                  index0 = 0, index1 = 1,
                  name0 = '', name1 = '', title = '', levels = 20):

    xvals = cfit.scan(pars, fun, index0, vals0)
    plt.figure(figsize = (6 * 2, 5 * 2))
    plt.subplot(2, 2, 3);
    plt.plot(vals0, xvals - np.min(xvals));
    plt.xlabel(name0); plt.title(title); plt.grid();

    xvals = cfit.scan(pars, fun, index1, vals1)
    plt.subplot(2, 2, 2)
    plt.plot(vals1, xvals - np.min(xvals));
    plt.xlabel(name1); plt.title(title); plt.grid();

    xvals = cfit.scan2d(pars, fun, index0, vals0, index1, vals1)
    xx, yy = np.meshgrid(vals0, vals1)
    zz     = xvals - np.min(xvals)
    plt.subplot(2, 2, 1)
    cc = plt.contourf(xx, yy, zz, alpha = 0.5, levels = levels);
    x0, y0 = pars[index0], pars[index1]
    plt.plot(x0, y0, marker = '*', color = 'black', ms = 10);
    plt.xlabel(name0); plt.ylabel(name1); plt.title(title)
    plt.colorbar(cc);
