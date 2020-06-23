import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.dates as mdates
plt.rcParams["figure.figsize"] = 8, 6
plt.rcParams["font.size"     ] = 12

import os
import sys
import glob
import time
import warnings
import datetime

import numpy as np
import pandas as pd
import matplotlib

import ipywidgets as widgets
from ipywidgets import IntProgress
from IPython.display import display

import c19.data_functions_momo as momodata
import c19.momo_analysis       as c19ma

import c19.useir            as us
import c19.kfilter          as kf
import c19.useir_ana        as usa
import c19.cfit             as cfit

#import c19.momodata         as

import scipy          as sp
import scipy.stats    as stats
import scipy.optimize as optimize

npa     = np.array
npdate  = np.datetime64
npdtime = np.timedelta64

#---- data

path = '/Users/hernando/investigacion/bio/c19/cdata/'
dfv, dfe, dfc = momodata.comomo_from_csv(path=path)

#---  Kpars0

K0 = {}
K0['Madrid'] =  {'t0': 40.6631717631003, 'beta': 1.5661995670268805, 'gamma': 0.18622613570230911,
                 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 6578079.0, 'phi': 0.012575986452685078,
                 's1': 0.10189130199782927}
K0['Castilla y Leon'] = {'t0': 32.66622553582464, 'beta': 1.413767144021723, 'gamma': 0.16803306349835717, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 2409164.0, 'phi': 0.021587067639146522, 's1': 0.03355446594468488}
K0['Castilla La Mancha'] = {'t0': 35.583008607563954, 'beta': 1.505713108477026, 'gamma': 0.20372946998745708, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 2026807.0, 'phi': 0.01427983835697912, 's1': 0.10112184170026027}
K0['Cataluna'] =  {'t0': 37.557781458118015, 'beta': 1.4948226070499842, 'gamma': 0.19756720386818188, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 7600065.0, 'phi': 0.00984638319827012, 's1': 0.08775740900486598}
K0['C. Valenciana'] = {'t0': 34.00043666729813, 'beta': 1.3500258329975496, 'gamma': 0.15235947218901752, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 4963703.0, 'phi': 0.009999940647715888, 's1': 0.024503207404145792}
K0['Aragon'] = {'t0': 27.441067728142166, 'beta': 1.2175582655593709, 'gamma': 0.1617444757352847, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 1308728.0, 'phi': 0.0324295828107219, 's1': 0.012361994039465148}
K0['Pais Vasco'] = {'t0': 28.487651703454375, 'beta': 1.5514355877315564, 'gamma': 0.21376206021150718, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 2199088.0, 'phi': 0.012405935479150098, 's1': 0.023205716935152478}
K0['Navarra'] =  {'t0': 25.730898012862696, 'beta': 1.374293557199548, 'gamma': 0.1360021205509374, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 647554.0, 'phi': 0.018725322617549826, 's1': 0.030756373554041032}
K0['La Rioja'] = {'t0': 32.99989770298113, 'beta': 1.1173163050996746, 'gamma': 0.1259755233477562, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 315675.0, 'phi': 0.009998990480059547, 's1': 0.08745144103411126}
K0['Cantabria'] = {'t0': 13.016169109759588, 'beta': 1.3920264445207466, 'gamma': 0.17268189806112166, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 580229.0, 'phi': 0.02922136664417093, 's1': 0.006201730808334946}
K0['Asturias'] = {'t0': 29.21824683092212, 'beta': 1.350942953822674, 'gamma': 0.2585300851484079, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 1028244.0, 'phi': 0.011458461172496385, 's1': 0.010161481775283424}
K0['Galicia'] = {'t0': 28.62464298804072, 'beta': 1.1191494904129895, 'gamma': 0.15861428056079435, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 2701743.0, 'phi': 0.028397168179308336, 's1': 0.004337577431946362}
K0['Murcia'] = {'t0': 12.48935511660366, 'beta': 2.2023881801480085, 'gamma': 0.23365103604671206, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 1478509.0, 'phi': 0.01605628462265475, 's1': 0.0021878612265094207}
K0['Andalucia'] = {'t0': 29.362184078944743, 'beta': 1.2902677529879232, 'gamma': 0.17046417361737165, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 8384408.0, 'phi': 0.04687241258808541, 's1': 0.0020818074559111814}
K0['Extremadura'] = {'t0': 31.775588594031863, 'beta': 1.2240900379676996, 'gamma': 0.18479754151357647, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 1072863.0, 'phi': 0.016715544903289505, 's1': 0.01927127899745814}
K0['Canarias']  = {'t0': 29.886253597018758, 'beta': 1.2126353053658898, 'gamma': 0.2238995476983161, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 2127685.0, 'phi': 0.014956806605211672, 's1': 0.0018462433393859915}
K0['Baleares'] = {'t0': 30.800723833629, 'beta': 1.200083828684592, 'gamma': 0.203209532709875213, 'tr': 2.7, 'ti': 4.75, 'tm': 9.0, 'n': 1128908.0, 'phi': 0.01542693902616, 's1': 0.1316663797438487}


#--- code

canames = ['Madrid', 'Castilla y Leon', 'Castilla La Mancha',
           'Cataluna', 'C. Valenciana', 'Aragon',
           'Pais Vasco', 'Navarra', 'La Rioja',
           'Cantabria', 'Asturias', 'Galicia',
           'Murcia', 'Andalucia', 'Extremadura',
           'Canarias', 'Baleares']

nn = {'Madrid': 6578079, 'Castilla y Leon': 2409164, 'Castilla La Mancha': 2026807,
      'Cataluna': 7600065, 'C. Valenciana': 4963703, 'Aragon': 1308728, 'Pais Vasco': 2199088,
      'Navarra': 647554, 'La Rioja': 315675, 'Cantabria': 580229, 'Asturias': 1028244,
      'Galicia': 2701743, 'Murcia': 1478509, 'Andalucia': 8384408, 'Extremadura': 1072863,
      'Baleares': 1128908, 'Canarias': 2127685}

TR, TI = 2.7, 4.75

#masks0 =  2  * (('t0', 's1', 'phi'), ('beta', 'gamma'))
#masks0 =  2 * (('t0', 's1', 'phi'), ('s1', 'beta', 'gamma')) + ('t0', 's1', 'phi')
kmasks0 = (('t0', 's1'), ('s1', 'beta'), ('t0', 'gamma'))


def ca_cases(caname):
    """ returns dates, cases of a given autonomous comunity
    """
    dates  = dfv.Date.values
    cases  = dfv[caname].values
    ucases = dfe[caname].values

    if (caname == 'Galicia'):
        sel0 = dates == npdate('2020-04-29')
        sel1 = dates == npdate('2020-04-28')
        cases[sel0] = cases[sel1]

    return dates, cases, ucases


def kpars0(caname, tr = TR, ti = TI):
    """ returns the default parameters for agive autonomous comunity
    """

    n0     = nn[caname]
    phi    = 0.008
    r0, r1 = 4, 0.6
    s1  = 0.05
    t0  = 35
    if (caname in ('Madrid', 'Cataluna', 'Castilla y Leon', 'Castilla La Mancha')):
        s1 = 0.12; t0 = 40
    if (caname in ('Cantabria', 'Asturias', 'Galicia', 'Murcia', 'Andalucia', 'C. Valenciana',
               'Canarias', 'Baleares')):
        s1 = 0.02; t0 = 30

    kpars =  {'t0': t0, 'beta': r0/tr, 'gamma': r1/tr, 'tr': tr, 'ti': TI,
                   'tm': 9., 'n': n0, 'phi': phi, 's1': s1}
    return kpars


def ana_ca(caname, kpars = None, kmasks = None, verbose = False):
    """ a sequence of fits (controlled by the list of masks) data of caname community
        returns a dictoinary with the estimated parameters in each fit in the sequence.
    """

    dates, cases, ucases = ca_cases(caname)
    ts                   = np.arange(len(dates))

    ikpars = kpars0(caname) if kpars  is None else kpars.copy()
    if 'chi2' in ikpars.keys(): ikpars.pop('chi2')
    kmasks = kmasks0        if kmasks is None else kmasks

    dpars = []
    for kmask in kmasks:
        if (verbose):
            print(' fit ', caname, ' masks ', kmask)
            print(' fit ', caname, ' pars0 ', ikpars)
        ikpars, chi2 = usa.useirqm_chi2fit(dates, cases, kpars = ikpars, kmask = kmask)
        ikpars_      = ikpars.copy(); ikpars_['chi2'] = chi2
        if (verbose):
            print(' fit ', caname, ' pars ', ikpars_)
        dpars.append(ikpars_)

    rpars = {}
    for key in dpars[0].keys():
        rpars[key] = [dpar[key] for dpar in dpars]

    return rpars

def plot_ana_ca(caname, kpars):
    dates, cases, ucases = ca_cases(caname)
    ikpars   =  kpars.copy()
    if 'chi2' in ikpars.keys(): ikpars.pop('chi2')
    fmodel      = usa.useirqm_fmodel(kpars = ikpars)
    usa.plot_data_model(dates, cases, fmodel)
    plt.title(caname)
    return


def ana_ccaa(canames, kpars = None, kmasks = None, bar = True, verbose = False):

    tbar = None
    if bar:
        nmax = len(canames)
        tbar = IntProgress(0, max = nmax + 1, description = 'fitting...'); display(tbar)
        if (bar): tbar.value += 1

    dpars = {}
    for caname in canames:
        ikpars = kpars[caname] if kpars is not None else None
        idpars = ana_ca(caname, ikpars, kmasks, verbose = verbose)
        dpars[caname] = idpars
        if (bar): tbar.value += 1
    return dpars

def plot_ana_ccaa(dpars, ncols = 2):

    canames = dpars.keys()

    nrows = int(len(canames)/ncols) + int(len(canames)%ncols)

    plt.figure(figsize = (6 * ncols, 5 * nrows))

    for i, caname in enumerate(canames):
        kpars = {}
        for key in dpars[caname].keys():
            kpars[key] = dpars[caname][key][-1]
        plt.subplot(nrows, ncols, i+1)
        plot_ana_ca(caname, kpars)
    plt.tight_layout()
    return

def plot_ana_ccaa_evo(dpars, canames = None):
    canames = dpars.keys() if canames == None else canames
    keys = ['chi2', 't0', 'beta', 'gamma', 's1', 'phi', 'dchi2']

    nrows, ncols = 4, 2
    plt.figure(figsize = (6 * ncols, 5 * nrows))
    for i, key in enumerate(keys):
        plt.subplot(nrows, ncols, i + 1)
        for caname in canames:
            kkey = key if key != 'dchi2' else 'chi2'
            data = dpars[caname][kkey]
            if (key == 'dchi2'):
                data = data - np.min(data)
            plt.plot(np.arange(len(data)), data, ls = '--', marker = 'o', label = caname)
            #if (key == 'dchi2'): plt.yscale('log')
        plt.title(key); plt.grid(); plt.legend()
    plt.tight_layout()

    return

def get_dpars(filename):

    dfr = pd.read_csv(filename)
    dfr = dfr.rename(columns = {'Unnamed: 0': 'pars'})
    dfr = dfr.set_index('pars')

    def _vals(vals):
        vals = vals.replace('[', ''); vals = vals.replace(']', '')
        vals = [float(val) for val in vals.split(',')]
        return vals

    for icol in dfr.columns.values:
        for irow in dfr.index.values:
            vals = dfr[icol][irow]
            dfr[icol][irow] = _vals(vals)

    return dfr


def plot_dpars(dpars, canames = None, id = -1,
               keys = ['chi2', 't0', 'beta', 'gamma', 'r0', 'r1', 's1', 'phi']):

    canames = dpars.keys() if canames is None else canames

    ncols = 2
    nrows = int(len(keys)/ncols) + int(len(keys)%ncols)

    plt.figure(figsize = (ncols * 6, nrows * 5))

    for i, key in enumerate(keys):
        plt.subplot(nrows, ncols, i + 1)
        kkey = key
        if (key == 'r0'): kkey = 'beta'
        if (key == 'r1'): kkey = 'gamma'
        vals = npa([dpars[caname][kkey][id] for caname in canames])
        if (key in ['r0', 'r1']):
            trs = npa([dpars[caname]['tr'][id] for caname in canames])
            vals = vals * trs
        plt.plot(vals, ls = '', marker = '*')
        plt.title(key); plt.grid();
        plt.xticks(np.arange(len(canames)), canames, rotation = 90);

    plt.tight_layout()
    return


#--- RUN --

confs = {'4complete' : 8 * (('t0', 's1', 'beta', 'gamma', 'phi'),) ,
         '4complete0': 8 * (('t0', 's1', 'beta', 'gamma'),)        ,
         '4default'  : 4 * (('t0', 's1', 'phi'), ('t0', 'beta', 'gamma')),
         '4default1' : 4 * (('t0', 's1', 'beta'), ('t0', 'phi', 'gamma'))}



def run(confi):
    canames1 = canames
    #masks  = 2 * (('t0', 's1', 'beta'), ('t0', 'phi', 'gamma'),)
    #              ('t0', 's1', 'phi'), ('beta', 'gamma')))
    #masks  = 4  * (('t0', 's1', 'phi'), ('t0', 'beta', 'gamma'))
    #masks    = 8  * (('s1', 't0', 'beta', 'gamma'),)
    masks    = confs[confi]
    print('confi : ', confi)
    print('masks : ', masks)
    print('ccaa  : ', canames1)
    dpars    = ana_ccaa(canames1, kmasks = masks, verbose = False, bar = True)
    dfr = pd.DataFrame(dpars)
    dfr.to_csv('ccaa_' + confi + '.csv')
    #plot_ana_ccaa_evo(dpars)
    #plot_ana_ccaa(dpars)



run('4default1')
