import numpy as np
import pandas as pd
import datetime

import c19.useir   as us

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.dates as mdates

from numpy import array as npa

npdate = np.datetime64
npday  = lambda days : np.timedelta64(days, 'D')

date0 = np.datetime64('2020-03-15')
#dates = [np.datetime64(d) for d in ('2020-02-15', '2020-03-01', '2020-04-01',
#                                    '2020-04-15', '2020-05-01', '2020-05-15')]

def ddate(days):
    return np.timedelta64(days, 'D')

def formatter(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

def data_ca(df, name, sel = None):

    dfc = df[df.geoId == name]

    N    = dfc.popData2018.values[0]

    dates = dfc.dateRep.values
    nis   = dfc.cases.values
    nrs   = dfc.recovered.values
    nms   = dfc.deaths.values

    # sort by date
    xdata = [(date, ni, nr, nm) for date, ni, nr, nm in zip(dates, nis, nrs, nms)]
    xdata.sort()

    dates = npa([xd[0] for xd in xdata])
    nis   = npa([xd[1] for xd in xdata])
    nrs   = npa([xd[2] for xd in xdata])
    nms   = npa([xd[3] for xd in xdata])

    #sodates = [data]

    def _mdeltas(v):
        m     = np.copy(v)
        m[1:] =  v[1:] - v[:-1]
        return m

    dios = _mdeltas(nis)
    drs  = _mdeltas(nrs)
    dms  = _mdeltas(nms)

    #sel = np.logical_and(np.logical_and((dios > 0), (drs > 0)), (dms > 0))

    #ns = (nis[sel] , nrs[sel], nms[sel])
    #ds = (dios[sel], drs[sel], dms[sel])
    #return dates[sel], days[sel], ns, ds

    ns = (nis , nrs, nms)
    ds = (dios, drs, dms)
    return dates,  ns, ds

def plt_data_ca(dates, ns, ds, yscale = 'log'):

    def _plot(dates, nis, nrs, nms, title, ylabel = 'individuals'):
        plt.figure(figsize = (8, 6))
        plt.plot(dates, nis, ls = '--', marker = 'o', label = 'infected');
        plt.plot(dates, nrs, ls = '--', marker = 'o', label = 'recovered');
        plt.plot(dates, nms, ls = '--', marker = 'o', label = 'deaths');
        plt.title(title)
        plt.legend(); plt.grid(which = 'both'); plt.xlabel('days');
        plt.ylabel(ylabel); formatter(plt.gca()); plt.yscale(yscale);
        return

    nis, nrs, nms = ns
    _plot(dates, nis, nrs, nms, 'integrated')

    dis, drs, dms = ds
    _plot(dates, dis, drs, dms, 'incremental')

    fis = (nis - nrs - nms)/np.maximum(1., nis)
    frs = nrs/np.maximum(1., nis)
    fms = nms/np.maximum(1., nis)
    _plot(dates, fis, frs, fms, 'fraction', '')

    return


def ana_ca(dates, ns, ds, times, frho, phim = 0.):

    ti, tr, tm, td = times
    dios, drs, dms = ds
    ms, ums        = us.meas(dios, drs, dms)

    # Compute the I curve from proxy: either detector or death
    ts              = np.arange(len(dios))
    xdios           = np.copy(dios)
    if (phim > 0.):
        xdios  = np.zeros(len(dios))
        i0 = int(td)
        xdios[:-i0] = dms[i0:]/phim
    xnis, xdis      = us.nis_(ts, xdios, frho(td), frho(td), 0.5)
    hs              = us.hmatrices(ts, xdios, xnis, frho(ti), frho(tr), frho(tm))

    x0           = npa((5., 0.8, 0.2))
    ux0          = np.identity(3) * 1.
    qi           = npa((1., 1., 1.))
    xs, uxs, res = us.useir_kf(ms, ums, hs, x0, ux0, qi)

    return (xs, uxs, res), (xdios, xnis, hs)

def plt_ana_ca(dates, ds, kfres, nisres, times, yscale = 'log'):

    dios, drs, dms = ds
    xs  , uxs, res = kfres
    xdios, nis, hs = nisres
    ti, tr, tm, td = times

    #us.plt_useir_kf(dates, xs, uxs, res)
    bes = npa([xi[0] for xi in xs])
    pms = npa([xi[2] for xi in xs])

    xdates = dates - np.timedelta64(ti + td, 'D')
    #xdates = dates
    #print(len(bes), len(dios), len(dates))
    plt.figure(figsize = (8, 6))
    xlabel = r'$R, T_D $' + str(td) + r'$, \, T_I =$' + str(ti)
    plt.plot(xdates, bes * td, ls = '--', marker = 'o', label = xlabel)
    plt.plot( dates, pms     , ls = '--', marker = 'o', label = r'$\Phi_m$')
    plt.ylim((0.01, 100.)); plt.grid(which = 'both')
    plt.legend(); formatter(plt.gca()); plt.yscale(yscale);

    #ax2 = plt.gca().twinx()
    #ax2.grid(True)
    #ax2.plot(ts, pr, label = r'$\Phi_R$')
    #ax2.plot( dates, pms     , ls = '--', marker = 'o', c = 'r',  label = r'$\Phi_m$')
    #ax2.set_ylabel(r'$\Phi$');
    #ax2.legend();
    plt.title('R')

    plt.figure(figsize = (8, 6))
    xdates = dates - np.timedelta64(td, 'D')
    xlabel = r'$I/T_D, \; T_D$' + str(td)
    plt.plot(xdates, nis/td, lw = 2, label = xlabel);
    plt.plot(dates, xdios, lw = 2, label = r'$\Delta I, proxy$');
    plt.plot(dates , ds[0] , ls = '--', marker = 'o', label = r'$\Delta I_d$');
    #plt.plot(dates , ds[1] , ls = '--', marker = 'o', label = r'$\Delta R$');
    plt.plot(dates , ds[2] , ls = '--', marker = 'o', label = r'$\Delta $M');
    formatter(plt.gca()); plt.grid(which = 'both'); plt.legend(); plt.yscale(yscale);
    plt.title('I/Td')

    return

#---- KF
