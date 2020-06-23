import numpy as np
import pandas as pd
import os
import datetime

import c19.data_functions_c19 as c19data
import c19.plotting as cplt
import c19.io as cio

import c19.useir   as us
import c19.cfit    as cfit

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.dates as mdates

from numpy import array as npa

npdate   = np.datetime64
npdtime  = lambda days : np.timedelta64(days, 'D')

api_key = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqamdvbWV6Y2FkZW5hc0BnbWFpbC5jb20iLCJqdGkiOiI5MDUzYjU1MC0wODkzLTRmYWMtYTNhMC1hNzk2ZDFjMTk0NzIiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTU4NTc1OTIyMiwidXNlcklkIjoiOTA1M2I1NTAtMDg5My00ZmFjLWEzYTAtYTc5NmQxYzE5NDcyIiwicm9sZSI6IiJ9.AiauVzy5kdJStrmK9vxQWpcaTf6Cg7EeVDscrRBX_lU"

date0 = np.datetime64('2020-03-15')

ccaas = {'Madrid'            : ('MA', 'MD'),
         'Castilla y Leon'   : ('CL', 'CL'),
         'Castilla La Mancha': ('CM', 'CM'),
         'Cataluna'          : ('CA', 'CT'),
         'C. Valenciana'     : ('CV', 'VC'),
         'Aragon'            : ('AR', 'AR'),
         'Pais Vasco'        : ('PV', 'PV'),
         'Navarra'           : ('NA', 'NC'),
         'Pais Vasco'        : ('PV', 'PV'),
         'La Rioja'          : ('LR', 'LR'),
         'Cantabria'         : ('CT', 'CT'),
         'Asturias'          : ('AS', 'AS'),
         'Galicia'           : ('GA', 'GA'),
         'Murcia'            : ('MU', 'MU'),
         'Andalucia'         : ('AN', 'AN'),
         'Extremadura'       : ('EX', 'EX'),
         'Baleares'          : ('BA', 'BA'),
         'Canarias'          : ('CN', 'CN')
}


def figure(nx = 1, ny = 1):
    if (len(plt.get_fignums()) > 0): return
    plt.figure(figsize = (nx * 6, ny * 5))
    return

def formatter(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

#--- sanidad data

def dfsanidad():
    dfsa    = c19data.get_data_communities(api_key, update = True).fillna(0)
    return dfsa

def population(dfs = None):
    dfs = dfsanidad() if dfs is None else dfs
    nn = {}
    for key in ccaas.keys():
        cas = ccaas[key][0]
        nn[key] = dfs[dfs.geoId == cas].popData2018.values[0]
    return nn

def dfmomo():
    path = '/Users/hernando/investigacion/bio/c19/data'
    file = os.path.join(path,'momo_17_05_2020.csv')
    print(file)
    dfmo = pd.read_csv(file)
    return dfmo

def dfsanidad_ca(df, ca = 'MA'):
    return df[df.geoId == ca]

def dfsanidad_cadata(df, ca = 'MA'):

    dfc = dfsanidad_ca(df, ca)

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

    def _mdeltas(v):
        m     = np.copy(v)
        m[1:] =  v[1:] - v[:-1]
        return m

    dios = _mdeltas(nis)
    drs  = _mdeltas(nrs)
    dms  = _mdeltas(nms)

    def _repare(dms):
        if (ca == 'GA'):
            sel0 = dates == npdate('2020-04-29')
            sel1 = dates == npdate('2020-04-28')
            dms[sel0] = dms[sel1]
        return dms

    dms = _repare(dms)

    ns = (nis , nrs, nms)
    ds = (dios, drs, dms)
    return dates, ns, ds

def plt_dfsanidad_cadata(dates, ns, ds, yscale = 'log'):

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

#----- momo data

def dfmomo_ca(df, ca = 'MD', cod_sexo='all', cod_gedad='all'):
    c1 = df[df['ambito'] == 'ccaa']
    c2 = c1[c1['cod_ambito'] == ca]
    c3 = c2[c2['cod_sexo'] == cod_sexo]
    c4 = c3[c3['cod_gedad'] == cod_gedad]
    return c4

def dfmomo_cadata(df, ca = 'MD', date0 = '2020-03-01'):
    dfs     = dfmomo_ca(df, ca)

    sdates  = dfs['fecha_defuncion'].values
    deaths  = dfs['defunciones_observadas'].values
    deaths0 = dfs['defunciones_esperadas'].values

    dates   = np.array([np.datetime64(xi, 'D') for xi in sdates])

    sel     = dates >= np.datetime64('2020-03-01')
    xdates  = dates[sel]
    ydeaths = deaths[sel]
    ddeaths = np.maximum(deaths[sel] - deaths0[sel], 0.)
    derrors  = np.sqrt(deaths[sel] + deaths0[sel])

    return xdates, ydeaths, ddeaths, derrors


def plt_dfmomo_cadata(xdates, deaths, xdeaths, xerrors, yscale = 'log'):
    plt.figure(figsize = (8, 6))
    plt.plot(xdates,  deaths, ls = '--', marker = 'o', label = 'total');
    plt.plot(xdates, xdeaths, ls = '--', marker = 'o', label = 'excess');
    plt.grid(which = 'both'); plt.legend(); plt.yscale(yscale)
    formatter(plt.gca())
    return

#--- CCAA comparisons

def plot_cas_momos(dfsa, dfmo, cas = ccaas.keys(), yscale = 'linear'):
    nr, nc = int(len(cas)/2  + len(cas)%2), 2
    plt.figure(figsize = (6 * nc, 5 * nr));
    for i, ca in enumerate(cas):
        cas, cam = ccaas[ca]
        sdates, sns, sds         = dfsanidad_cadata(dfsa, cas)
        mdates, mms, mdms, mudms = dfmomo_cadata   (dfmo, cam)

        plt.subplot(nr, nc, i+1)
        plt.errorbar(sdates, sds[2], yerr = us.errors(sds[2]),
                     ls = ':', marker = 'o', ms = 4, label = 'covid19')
        plt.errorbar(mdates, mms   , yerr = us.errors(mms),
                     ls = ':', marker = 'o', ms = 4, label = 'momo')
        plt.errorbar(mdates, mdms, yerr = mudms, ls = ':', marker = 'o',
                     ms = 4, label = 'momo-excess')
        plt.legend(); plt.grid(which = 'both');
        formatter(plt.gca()); plt.title(ca); plt.yscale(yscale)

#---- KF analysis

def ca_comomo(dfsa, dfmo, ca = 'Madrid', nsig = 2., plot = False):

    cas, cam = ccaas[ca]
    sdates, sns, sds         = dfsanidad_cadata(dfsa, cas)
    mdates, mms, mdms, mudms = dfmomo_cadata   (dfmo, cam)

    sdms = sds[2]
    cms, ucms  = np.zeros(len(mdates)), np.zeros(len(mdates))
    for i, mdate in enumerate(mdates):
        isel = sdates == mdate
        cms[i]  = sdms[isel]
        ucms[i] = np.maximum(np.sqrt(cms[i]), 2.4)
        dd = (mdms[i] - sdms[isel])/mudms[i]
        if (dd > nsig):
            cms[i]  = mdms[i]
            ucms[i] = mudms[i]

    if (plot):
        pargs = {'ls' : '--', 'marker' : 'o', 'ms' : 4}
        plt.errorbar(mdates, cms , yerr = ucms, ls = '-', marker = 'o', label = 'covid-momo');
        plt.plot(mdates, mdms, **pargs, label = 'momo-excess');
        plt.plot(sdates, sdms, **pargs, label = 'covid');
        plt.title(ca)
        formatter(plt.gca()); plt.legend(); plt.grid(which = 'both');

    return mdates, cms, ucms


#--- kf a CA

def kfqs(dates, idates, dt, qi = 1.):
    qis = np.zeros(len(dates))
    itd = np.timedelta64(dt, 'D')
    id0, id1 = idates
    sel = (dates - itd >= npdate(id0)) & (dates -itd <= npdate(id1))
    qis[sel] = qi;
    qs  = [np.identity(3) * qi for qi in qis]
    return qs

def ca_kafi(dfsa, dfmo, ca, times = (5, 10, 7, 5), frho = us.fgamma,
            idates = ('2020-01-01', '2020-06-01'), plot = False):

    ti, tr, tm, td = times
    idt = int(ti + td + tm)

    def _rs(kres):
        xs, uxs, _, _  = kres
        rs  = td * npa([xi[0] for xi in xs])
        urs = td * npa([np.sqrt(xi[0, 0]) for xi in uxs])
        return rs, urs

    xdates, xdms, xudms = ca_comomo(dfsa, dfmo, ca)
    kms, kums           = kfmeas(xdms, xudms)
    qs                  = kfqs(xdates, idates, idt)
    kres, bres          = kafi(xdates, kms, kums, times, frho, qs)
    krs                 = _rs(kres)
    xdates              = xdates - np.timedelta64(idt, 'D')

    ca0              = ccaas[ca][0]
    sdates, sns, sds = dfsanidad_cadata(dfsa, ca0)
    sdms, sudms      = sds[2], us.errors(sds[2])
    skms, skums      = kfmeas(sdms, sudms)
    sqs              = kfqs(sdates, idates, idt)
    skres, sbres     = kafi(sdates, skms, skums, times, frho, sqs)
    skrs             = _rs(skres)
    sdates           = sdates - np.timedelta64(idt, 'D')

    if (plot):
        pargs  = {'ls' : '--', 'marker' : 'o', 'ms' : 4}
        yscale = 'log'
        idate0 = '2020-02-25'
        plt.figure(figsize = (12, 5))

        plt.subplot(1, 2, 1)
        plt.errorbar(xdates, xdms, yerr = xudms, **pargs, label = 'comomo');
        plt.errorbar(sdates, sdms, yerr = sudms, **pargs, label = 'covid');
        plt.grid(which = 'both'); plt.legend(); plt.yscale(yscale)
        formatter(plt.gca()); plt.title(r'$\Delta I_0$ proxy ' + str(ca))

        plt.subplot(1, 2, 2)
        xsel = xdates >= npdate(idate0)
        plt.errorbar(xdates[xsel], krs[0][xsel] , yerr = krs[1][xsel],  **pargs, label = r'$r(t)$ comomo kf')
        plt.plot    (xdates[xsel], td * bres[0][xsel],                  **pargs, label = r'$\hat{r}(t)$ comomo')
        #xsel = sdates >= npdate(idate0)
        #plt.errorbar(sdates[xsel], skrs[0][xsel], yerr = skrs[1][xsel], **pargs, label = r'$r(t)$ covid kf')
        plt.grid(which = 'both'); plt.legend(); plt.yscale(yscale); plt.ylim((0.1, 20.))
        formatter(plt.gca()); plt.title(r'$r(t)$ ' + str(ca))

    return (xdates, krs[0], krs[1]), (sdates, skrs[0], skrs[1])


#--- KF fits

def kfmeas(dms, udms):
    norma  = np.sum(dms)
    xdms   = dms/norma
    xudms  = udms/norma

    kms  = [npa((xi, xi, xi)) for xi in xdms]
    kums = [np.identity(3) * xi * xi for xi in xudms]

    return kms, kums

def kafi(dates, kms, kums, times, frho, qs = 0., x0 = (2., 0.8, 0.2)):

    nsample = len(kms)
    ts = np.arange(len(dates))
    ti, tr, tm, td = times

    xdios = npa([mi[0] for mi in kms])

    xnis , _       = us.ninfecting(ts, xdios, frho(td))
    xbetas         = us.betas(ts, xdios, frho(td), frho(ti))
    hs             = us.hmatrices(ts, xdios, xnis, frho(ti), frho(tr), frho(tm))

    x0           = npa(x0)
    ux0          = np.identity(3) * 10.

    if (type(qs) == float):
        qs = [np.identity(3) * qs for i in range(nsample)]

    xs, uxs, res = us.useir_kf(kms, kums, hs, x0, ux0, qs)
    return (xs, uxs, res, hs), (xbetas, xnis)


def plt_kafi(dates, times, kfres, nisres, yscale = 'log', date0 = '2020-02-15'):

    ti, tr, tm, td  = times

    xs    , uxs, res, hs  = kfres
    xbetas, xnis          = nisres

    plt.figure(figsize = ((8, 6)));
    krs  = td * npa([xi[0] for xi in xs])
    kurs = td * npa([np.sqrt(xi[0, 0]) for xi in uxs])
    idt  = np.timedelta64(int(ti + td + tm), 'D')

    xdates = dates - idt
    sel    = xdates >= npdate(date0)

    pargs = {'ls' : '--', 'marker' : 'o', 'ms' : 4}
    plt.errorbar(xdates[sel], krs[sel], yerr = kurs[sel], **pargs, label = r'$r(t)$ kf')
    plt.plot(xdates[sel], td * xbetas[sel], ls = '--', marker = 'o', label = r'$\hat{r}(t)$')
    plt.grid(which = 'both'); plt.legend(); plt.yscale(yscale)
    formatter(plt.gca());


#-------------

def ana_kf(dates, ds, times, frho, qi = (100., 0.1, 0.1), type = 'proxy'):

    #TODO: errors on ds
    #qi   = npa((1e3, 1., 1.)) if qi is None else qi

    ti, tr, tm, td = times
    xds = (ds, ds, ds) if type == 'momo' else ds
    dios, drs, dms = xds[0], xds[1], xds[2]
    #print(len(dios), dios)
    #print(len(drs), drs)
    #print(len(dms), dms)

    xdios = dios
    if (type == 'proxy'):
        xdios = np.zeros(len(dios))
        i0    = 0 #int(td)
        norma = np.sum(dms)
        #print('norma ', norma)
        xdios[:] = dms[:]/norma
    if (type == 'proxy-reco'):
        xdios = np.zeros(len(dios))
        #i0    = int(td)
        norma = np.sum(drs)
        xdios[:] = drs[:]/norma
    if (type == 'proxy-detected'):
        xdios = np.zeros(len(dios))
        i0    = int(td)
        norma = np.sum(dios)
        xdios[:] = dios[:]/norma
    if (type == 'momo'):
        xdios = np.zeros(len(dios))
        i0    = int(td)
        norma = np.sum(dms)
        xdios[:] = dms[:]/norma

    xds = (xdios, drs, dms)
    return _ana_kf(dates, xds, times, frho, npa(qi))

def _kf_ana(dates, ds, times, frho, qi):

    ti, tr, tm, td  = times
    xdios, drs, dms = ds

    # Compute the I curve from proxy: either detector or death
    ts              = np.arange(len(xdios))

    ms, ums        = us.meas(xdios, drs, dms)
    xnis , _       = us.ninfecting(ts, xdios, frho(td))
    xbetas         = us.betas(ts, xdios, frho(td), frho(ti))
    #xnis, xdis     = us.nis_(ts, xdios, frho(td), frho(td), 0.5)
    hs             = us.hmatrices(ts, xdios, xnis, frho(ti), frho(tr), frho(tm))

    x0           = npa((1., 0.8, 0.2))
    ux0          = np.identity(3) * 1.
    xs, uxs, res = us.useir_kf(ms, ums, hs, x0, ux0, qi)
    return (xs, uxs, res), (xbetas, xdios, xnis, hs)

def plt_kf_ana(dates, times, kfres, nisres, yscale = 'log'):

    ti, tr, tm, td  = times

    xs    , uxs, res, hs  = kfres
    xbetas, xnis          = nisres

    plt.figure(figsize = ((8, 6)));
    kbetas = npa([xi[0] for xi in xs])
    idt    = np.timedelta64(ti + td, 'D')
    plt.plot(dates - idt, td * kbetas, ls = '--', marker = 'o', label = 'kf')
    plt.plot(dates - idt, td * xbetas, ls = '--', marker = 'o', label = 'betas')
    plt.grid(which = 'both'); plt.legend(); plt.yscale(yscale)
    formatter(plt.gca());

#--- KFs - plots

def plot_data_kfs_rs(dates, cases, rs_filter, rs_smooth = None, ucases = None):

    figure()
    yerr = ucases if ucases is not None else np.maximum(2.4, np.sqrt(cases))
    plt.errorbar(dates, cases, yerr = yerr, ls = '', marker = '*', color = 'black', ms = 4);
    ax2 = plt.gca().twinx()
    bs, ubs = rs_filter
    ax2.errorbar(dates, bs , yerr = ubs , ls = '--', marker = 'o', ms = 4, label = 'filter')
    if (rs_smooth is not None):
        bs, ubs = rs_smooth
        ax2.errorbar(dates, bs , yerr = ubs , ls = '--', marker = 'o', ms = 4, label = 'smoother')
    ax2.grid(which = 'both'); plt.legend();
    formatter(plt.gca()); plt.ylim((0., 8.)); #plt.yscale('log')
    return

#--- LLF - fits - functions


def useirqr_fit(dates, cases, kpars = {}, kmask = ('beta', 'gamma', 'tr'), ffit = 'chi2'):

    xpars = [30., 1., 0.3, 3., 5.5, 3e6, 0.01, 0.1]
    xmask = 8 * [False,]
    ixpar = {'t0' : 0, 'beta' : 1, 'gamma' : 2, 'tr' : 3, 'ti' : 4,
             'n'  : 5, 'phi'  : 6, 's1' : 7}

    ts     = np.arange(len(dates))
    xdata  = ts, cases

    umod = lambda pars : us._useirqr(pars, 'weibull')
    ufun = lambda pars : us._t0(pars, umod)
    _fun   = us.mll if ffit == 'mle' else us.res
    fun    = _fun(xdata, ufun = ufun)

    for key in kpars.keys(): xpars[ixpar[key]] = kpars[key]
    for key in kmask       : xmask[ixpar[key]] = True
    res   = cfit.minimize(xpars, fun, mask = xmask, method = 'Nelder-Mead')

    dres = {}
    for key in ixpar.keys(): dres[key] = res[ixpar[key]]

    xfun = np.sum(fun(res))  #/(len(ts) - np.sum(xmask))
    return dres, xfun


def useirqr_mllfit(dates, cases, kpars = {}, kmask = ('beta', 'gamma', 'tr')):

    xpars = [30., 1., 0.3, 3., 5.5, 3e6, 0.01, 0.1]
    xmask = 8 * [False,]
    ixpar = {'t0' : 0, 'beta' : 1, 'gamma' : 2, 'tr' : 3, 'ti' : 4,
             'n'  : 5, 'phi'  : 6, 's1' : 7}

    ts     = np.arange(len(dates))
    xdata  = ts, cases

    umod = lambda pars : us._useirqr(pars, 'weibull')
    ufun = lambda pars : us._t0(pars, umod)
    fun    = us.mle(xdata, ufun = ufun)

    for key in kpars.keys(): xpars[ixpar[key]] = kpars[key]
    for key in kmask       : xmask[ixpar[key]] = True
    res   = cfit.minimize(xpars, fun, mask = xmask, method = 'Nelder-Mead')

    dres = {}
    for key in ixpar.keys(): dres[key] = res[ixpar[key]]

    xmle = np.sum(fun(res))
    return dres, xmle


def useirqr_fmodel(kpars = {}, pars = None):

    xpars = [30., 1., 0.3, 3., 5.5, 3e6, 0.01, 0.1] if pars is None else pars
    ixpar = {'t0' : 0, 'beta' : 1, 'gamma' : 2, 'tr' : 3, 'ti' : 4,
             'n'  : 5, 'phi'  : 6, 's1' : 7}

    for key in kpars.keys(): xpars[ixpar[key]] = kpars[key]

    umod = lambda pars : us._useirqr(pars, 'weibull')
    ufun = lambda pars : us._t0(pars, umod)
    rfun = us.fmodel(xpars, ufun = ufun)
    return rfun


def useirqm_chi2fit(dates, cases, kpars = {}, kmask = ('beta', 'gamma', 'tr')):

    xpars = [30., 1., 0.3, 3., 5.5, 9., 3e6, 0.01, 0.1]
    xmask = 9 * [False,]
    ixpar = {'t0' : 0, 'beta' : 1, 'gamma' : 2, 'tr' : 3, 'ti' : 4,
             'tm' : 5, 'n'    : 6, 'phi'   : 7, 's1' : 8}

    ts     = np.arange(len(dates))
    xdata  = ts, cases

    umod = lambda pars : us._useirqm(pars, 'weibull')
    ufun = lambda pars : us._t0(pars, umod)
    fun    = us.res(xdata, ufun = ufun)

    for key in kpars.keys(): xpars[ixpar[key]] = kpars[key]
    for key in kmask       : xmask[ixpar[key]] = True
    res   = cfit.minimize(xpars, fun, mask = xmask, method = 'Nelder-Mead')

    dres = {}
    for key in ixpar.keys(): dres[key] = res[ixpar[key]]

    xchi2 = np.sum(fun(res)) #/(len(ts) - np.sum(xmask))
    return dres, xchi2


def useirqm_fmodel(kpars = {}, pars = None):

    xpars = [30., 1., 0.3, 3., 5.5, 9., 3e6, 0.01, 0.1] if pars is None else pars
    ixpar = {'t0' : 0, 'beta' : 1, 'gamma' : 2, 'tr' : 3, 'ti' : 4,
             'tm' : 5, 'n'    : 6, 'phi'   : 7, 's1' : 8}

    for key in kpars.keys(): xpars[ixpar[key]] = kpars[key]

    umod = lambda pars : us._useirqm(pars, 'weibull')
    ufun = lambda pars : us._t0(pars, umod)
    rfun = us.fmodel(xpars, ufun = ufun)
    return rfun


def useirqt_chi2fit(dates, cases, kpars = {}, kmask = ('beta', 'gamma', 'tr')):

    xpars = [30., 1., 0.3, 3., 5.5, 9., 3e6, 0.01, 0.1]
    xmask = 9 * [False,]
    ixpar = {'t0' : 0, 'beta' : 1, 'gamma' : 2, 'tr' : 3, 'ti' : 4,
             'tm' : 5, 'n'    : 6, 'phi'   : 7, 'tq' : 8}

    ts     = np.arange(len(dates))
    xdata  = ts, cases

    umod = lambda pars : us._useirq(pars, 'weibull')
    ufun = lambda pars : us._t0(pars, umod)
    fun    = us.res(xdata, ufun = ufun)

    for key in kpars.keys(): xpars[ixpar[key]] = kpars[key]
    for key in kmask       : xmask[ixpar[key]] = True
    res   = cfit.minimize(xpars, fun, mask = xmask, method = 'Nelder-Mead')

    dres = {}
    for key in ixpar.keys(): dres[key] = res[ixpar[key]]

    xchi2 = np.sum(fun(res))/(len(ts) - np.sum(xmask))
    return dres, xchi2


def useirqt_fmodel(kpars = {}, pars = None):

    xpars = [30., 1., 0.3, 3., 5.5, 9., 3e6, 0.01, 0.1] if pars is None else pars
    ixpar = {'t0' : 0, 'beta' : 1, 'gamma' : 2, 'tr' : 3, 'ti' : 4,
             'tm' : 5, 'n'    : 6, 'phi'   : 7, 'tq' : 8}

    for key in kpars.keys(): xpars[ixpar[key]] = kpars[key]

    umod = lambda pars : us._useirq(pars, 'weibull')
    ufun = lambda pars : us._t0(pars, umod)
    rfun = us.fmodel(xpars, ufun = ufun)
    return rfun



def plot_data_model(xs, ys, fmodel, yerr = None, dates = True):
    # TODO: check if cs is dates or not

    yerr = yerr if yerr is not None else np.maximum(np.sqrt(ys), 2.4)

    figure()
    plt.errorbar(xs, ys, yerr = yerr, color = 'black',
                ls = '', marker = 'o', ms = 4, label = 'data');
    ts = xs if dates is False else np.arange(len(xs))
    plt.plot(xs, fmodel(ts), ls = '--', label = 'model')
    plt.grid(); plt.legend();
    if (dates): formatter(plt.gca())
    return


def plot_fit_data(xdata, pars, ufun, ax = None):

    if (ax is None):  plt.figure(figsize = (6, 5))

    xs, ys = xdata

    plt.errorbar(xs, ys, yerr = np.sqrt(ys), color = 'black',
                ls = '', marker = 'o', ms = 4, label = 'data');

    fun = us.fmodel(pars, ufun = ufun)
    plt.plot(xs, fun(xs), ls = '--', label = 'model')

    plt.grid(); plt.legend();
    return

#
# def data_ca(df, name, sel = None):
#
#     dfc = df[df.geoId == name]
#
#     N    = dfc.popData2018.values[0]
#
#     dates = dfc.dateRep.values
#     nis   = dfc.cases.values
#     nrs   = dfc.recovered.values
#     nms   = dfc.deaths.values
#
#     # sort by date
#     xdata = [(date, ni, nr, nm) for date, ni, nr, nm in zip(dates, nis, nrs, nms)]
#     xdata.sort()
#
#     dates = npa([xd[0] for xd in xdata])
#     nis   = npa([xd[1] for xd in xdata])
#     nrs   = npa([xd[2] for xd in xdata])
#     nms   = npa([xd[3] for xd in xdata])
#
#     #sodates = [data]
#
#     def _mdeltas(v):
#         m     = np.copy(v)
#         m[1:] =  v[1:] - v[:-1]
#         return m
#
#     dios = _mdeltas(nis)
#     drs  = _mdeltas(nrs)
#     dms  = _mdeltas(nms)
#
#     #sel = np.logical_and(np.logical_and((dios > 0), (drs > 0)), (dms > 0))
#
#     #ns = (nis[sel] , nrs[sel], nms[sel])
#     #ds = (dios[sel], drs[sel], dms[sel])
#     #return dates[sel], days[sel], ns, ds
#
#     ns = (nis , nrs, nms)
#     ds = (dios, drs, dms)
#     return dates,  ns, ds
#
# def plt_data_ca(dates, ns, ds, yscale = 'log'):
#
#     def _plot(dates, nis, nrs, nms, title, ylabel = 'individuals'):
#         plt.figure(figsize = (8, 6))
#         plt.plot(dates, nis, ls = '--', marker = 'o', label = 'infected');
#         plt.plot(dates, nrs, ls = '--', marker = 'o', label = 'recovered');
#         plt.plot(dates, nms, ls = '--', marker = 'o', label = 'deaths');
#         plt.title(title)
#         plt.legend(); plt.grid(which = 'both'); plt.xlabel('days');
#         plt.ylabel(ylabel); formatter(plt.gca()); plt.yscale(yscale);
#         return
#
#     nis, nrs, nms = ns
#     _plot(dates, nis, nrs, nms, 'integrated')
#
#     dis, drs, dms = ds
#     _plot(dates, dis, drs, dms, 'incremental')
#
#     fis = (nis - nrs - nms)/np.maximum(1., nis)
#     frs = nrs/np.maximum(1., nis)
#     fms = nms/np.maximum(1., nis)
#     _plot(dates, fis, frs, fms, 'fraction', '')
#
#     return
#
#
# def ana_ca(dates, ns, ds, times, frho, phim = 0.):
#
#     ti, tr, tm, td = times
#     dios, drs, dms = ds
#
#     # Compute the I curve from proxy: either detector or death
#     ts              = np.arange(len(dios))
#     xdios           = np.copy(dios)
#     norma = 1.
#     if (phim > 0.):
#         xdios  = np.zeros(len(dios))
#         i0 = int(td)
#         norma       = np.sum(dms)
#         xdios[:-i0] = dms[i0:]/norma
#
#     ms, ums        = us.meas(xdios, drs/norma, dms/norma)
#     xnis, xdis     = us.nis_(ts, xdios, frho(td), frho(td), 0.5)
#     hs              = us.hmatrices(ts, xdios, xnis, frho(ti), frho(tr), frho(tm))
#
#     x0           = npa((0.5, 0.8, 0.2))
#     ux0          = np.identity(3) * 1.
#     qi           = npa((1e3, 1., 1.))
#     xs, uxs, res = us.useir_kf(ms, ums, hs, x0, ux0, qi)
#
#     return (xs, uxs, res), (xdios, xnis, hs)
#
# def plt_ana_ca(dates, ds, kfres, nisres, times, yscale = 'log'):
#
#     dios, drs, dms = ds
#     xs  , uxs, res = kfres
#     xdios, nis, hs = nisres
#     ti, tr, tm, td = times
#
#     #us.plt_useir_kf(dates, xs, uxs, res)
#     bes = npa([xi[0] for xi in xs])
#     pms = npa([xi[2] for xi in xs])
#
#     xdates = dates - np.timedelta64(ti + td, 'D')
#     #xdates = dates
#     #print(len(bes), len(dios), len(dates))
#     plt.figure(figsize = (8, 6))
#     xlabel = r'$R, T_D $' + str(td) + r'$, \, T_I =$' + str(ti)
#     plt.plot(xdates, bes * td, ls = '--', marker = 'o', label = xlabel)
#     plt.plot( dates, pms     , ls = '--', marker = 'o', label = r'$\Phi_m$')
#     plt.ylim((0.01, 100.)); plt.grid(which = 'both')
#     plt.legend(); formatter(plt.gca()); plt.yscale(yscale);
#
#     #ax2 = plt.gca().twinx()
#     #ax2.grid(True)
#     #ax2.plot(ts, pr, label = r'$\Phi_R$')
#     #ax2.plot( dates, pms     , ls = '--', marker = 'o', c = 'r',  label = r'$\Phi_m$')
#     #ax2.set_ylabel(r'$\Phi$');
#     #ax2.legend();
#     plt.title('R')
#
#     plt.figure(figsize = (8, 6))
#     xdates = dates - np.timedelta64(td, 'D')
#     xlabel = r'$I/T_D, \; T_D$' + str(td)
#     plt.plot(xdates, nis/td, lw = 2, label = xlabel);
#     plt.plot(dates, xdios, lw = 2, label = r'$\Delta I, proxy$');
#     plt.plot(dates , ds[0] , ls = '--', marker = 'o', label = r'$\Delta I_d$');
#     #plt.plot(dates , ds[1] , ls = '--', marker = 'o', label = r'$\Delta R$');
#     plt.plot(dates , ds[2] , ls = '--', marker = 'o', label = r'$\Delta $M');
#     formatter(plt.gca()); plt.grid(which = 'both'); plt.legend(); plt.yscale(yscale);
#     plt.title('I/Td')
#
#     return
#
# #---- KF
