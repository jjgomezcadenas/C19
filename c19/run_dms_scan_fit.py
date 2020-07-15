from copy import copy

import numpy as np
import pandas as pd
import time
import c19.kfilter as kf
import scipy.stats as stats
import c19.cfit    as cfit

import c19.useir   as us

from c19.data_functions_HG import hg_get_data_ingresos, hg_get_data_pacientes, hg_get_data_age
from c19.HG_analysis import plot_hgi, plot_nh_nu, recovery_time, exitus_time, exitus_group, exitus, hosp_uci, select_age_by_CA_and_sex

datapath="/Users/hernando/investigacion/bio/c19/data/HospitalesGalicia"
file = "IngresosCovid.csv"
filep = "PacientesCOVID.csv"
ff =f'{datapath}/{file}'
fp =f'{datapath}/{filep}'
dfi = pd.read_csv(ff, sep='|')

def get_cases(df, index = 'ingreso'):
    nps = df.groupby([index]).pacienteID.count()
    sel = nps.index >= np.datetime64('2020-02-15')
    dates, cases = nps.index[sel], nps.values[sel]
    return dates, cases

hgi = hg_get_data_ingresos(ff)
dates, cases = get_cases(hgi)
sel = dates >= np.datetime64('2020-03-01')
dates, cases = dates[sel], cases[sel]

# parameters
kpars0 =  {'t0': 22.79, 'beta': 1.48, 'gamma': 0.226, 'ti': 4.75, 'tr': 2.7,
           'n': 2701743.0, 'phim': 0.0661, 's1': 0.0072}
fmodel = us.dms_t0useirq_tr
pmask  = ('t0', 'phim')
klist  = (('beta', 1.3, 1.5, 5), ('gamma', 0.1, 0.3, 5), ('ti', 4.5, 5.5, 5),
          ('tr', 2.5, 3.5, 5), ('s1', 0.005, 0.01, 3))
#klist  = (('beta', 1.3, 1.5, 1), ('gamma', 0.1, 0.3, 1), ('ti', 4.5, 5.5, 1),
#          ('tr', 2.5, 3.5, 1), ('s1', 0.005, 0.01, 1))

dpars = []
ts    = us.to_ts(dates)
t0 = time.time()
print('run ', klist)
us.dms_scan_fit(klist, dpars, ts, cases, fmodel, kpars0, pmask);
t1 = time.time()
print('processing time ', t1 - t0, ' s')
dpars = pd.DataFrame(dpars)

dpars.to_csv('dms_scan_fit.csv', index = False)
