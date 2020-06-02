"""
Gets C19 and associated weather data from MOMO.

- MOMO data source: https://momo.isciii.es/public/momo/dashboard/momo_dashboard.html#datos

"""
import pandas as pd
import datetime
import os
import urllib
import json
import numpy as np

from . types import dc19, idc19, dmomo, idmomo


YMOM = {'obs':'defunciones_observadas', 'esp':'defunciones_esperadas','esp99':'defunciones_esperadas_q99','esp01':'defunciones_esperadas_q01'}

# URL for obtaining the C19 data.
url_momo_data  = "https://momo.isciii.es/public/momo/data"
url_covid_data = "https://cnecovid.isciii.es/covid19/resources/agregados.csv"

def get_data_momo(datapath="../data/momo_data.csv", update=False):

    # If we're just reading (not updating) the data, just read it from the CSV.
    if(not update):
        if(not os.path.isfile(datapath)):
            print("File",datapath,"does not exist. Run this function with update=True to retrieve the data.")
            return None
        df = pd.read_csv(datapath)
        return df

    # Read in the latitude/longitude dataframe for all countries.
    print(f"Reading momo data from {url_momo_data}")
    momo_file="../data/momo_data.csv"
    urllib.request.urlretrieve (url_momo_data, filename=momo_file)
    if(not os.path.isfile(momo_file)):
        print("ERROR downloading momo data.")
        return None
    print("-- Done")

    # Read in the C19 world data.
    df_momo = pd.read_csv(momo_file)

    return df_momo


def momo_select_spain(dm, cod_sexo='all', cod_gedad='all'):
    c1 = dm.loc[dm['ambito'] == 'nacional']
    c2 = c1.loc[c1['cod_sexo'] == cod_sexo]
    c3 = c2.loc[c2['cod_gedad'] == cod_gedad]
    return c3


def momo_select_ca(dm, ca_code='MD', cod_sexo='all', cod_gedad='all'):
    c1 = dm[dm['ambito'] == 'ccaa']
    c2 = c1[c1['cod_ambito'] == 'MD']
    c3 = c2[c2['cod_sexo'] == cod_sexo]
    c4 = c3[c3['cod_gedad'] == 'all']
    return c4


def momo_select_ccaa(dm, cod_sexo='all', cod_gedad='all'):
    ccaa_code = dmomo.values()

    c1 = dm.loc[dm['ambito'] == 'ccaa']
    c2 = c1.loc[c1['cod_sexo'] == cod_sexo]
    c3 = c2.loc[c2['cod_gedad'] == cod_gedad]
    dfcas ={idmomo[ca]:c3.loc[c3['cod_ambito'] == ca] for ca in ccaa_code}
    return dfcas


def momo_select_date(dm, date='2020-01-01', datef='2020-07-01'):
    dates = dm['fecha_defuncion'].values
    npdates =[np.datetime64(d) for d in dates]
    dm['npdate'] = npdates
    c1 = dm.loc[dm['npdate'] >= np.datetime64(date)]
    c2 = c1.loc[c1['npdate'] < np.datetime64(datef)]
    return c2


def momo_select_date_ccaa(dfs, date='2020-01-01', datef='2020-07-01'):
    return {ca:momo_select_date(df, date, datef) for ca, df in dfs.items()}


def get_mdata(df, ydata='defunciones_observadas', ccaa='Spain'):
    tD = df['npdate'].values
    tS = np.arange(len(tD))
    Y = df[ydata].values
    return tD, tS, Y


def get_mdata_ccaa(dfs, ydata='defunciones_observadas'):
    df = dfs['Madrid']
    tD = df['npdate'].values
    tS = np.arange(len(tD))
    Y = {ca:df[ydata].values for ca, df in dfs.items()}

    return tD, tS, Y


def dict_excess_momo(dYobs, dYesp):
    Y = {}
    for key, Yo in dYobs.items() :
        Ye = dYesp[key]
        Y[key] = Yo - Ye

    return Y


def get_xydata_ccaa(dfs, xdata='date', ydata='cdead'):
    X = {}
    Y = {}
    for ccaa_name, df in dfs.items() :
        if ccaa_name == 'Ceuta':
            continue

        X[ccaa_name] = df[xdata].values
        Y[ccaa_name] = df[ydata].values

    return X, Y


def get_c19_dead(cdeadC19):
    ig = 56  # wrong data for Galicia, Navarra, Riojas
    ina = 43
    ir  = 77
    dDead = {}
    for ca, cdead in cdeadC19.items():
        D = [cdead[0]]
        for i in range(1, len(cdead)):
            D.append(cdead[i] - cdead[i-1])

        if ca == 'Galicia':
            D[ig] = 0
        elif ca == 'Navarra':
            D[ina] = 0
        elif ca == 'La Rioja':
            D[ir] = 0

        dDead[ca] = D
    return dDead


def comomo(tD, dmomo, dc19, nsigma=2):
    CM  = {}
    ECM = {}
    for t, Y in dmomo.items() :
        if t == 'Ceuta':
            continue
        elif t == 'Melilla':
            CM[t]  = dc19[t]
            ECM[t] = dc19[t]
        else:
            Y1 = dc19[t]
            YY = []
            YE = []
            for i in range(len(Y)):
                ym  = Y[i]
                yc  = Y1[i]
                if yc >= 0:
                    eyc = np.sqrt(yc)
                else:
                    print(f'for t = {t}, i = {i}, yc = {yc}')
                    yc = 0
                    eyc = 0

                if ym <= 0:
                    ycm = yc
                elif ym - yc > nsigma * eyc:
                    ycm = ym
                else:
                    ycm = yc
                YY.append(ycm)
                YE.append(eyc)
            CM[t]  = YY
            ECM[t] = YE
        CM['Date'] = tD
        ECM['Date'] = tD
    return CM, ECM


def comomo_dataframe_from_dicts(dvalues, derrors):
    dict_of_df= {}
    dict_of_df['values'] = pd.DataFrame.from_dict(dvalues)
    dict_of_df['errors'] = pd.DataFrame.from_dict(derrors)
    df = pd.concat(dict_of_df, axis=1)
    return df


def comomo_to_csv(dvalues, derrors, path):
    path1 =f'{path}/cmvalues.csv'
    path2 =f'{path}/cmerrors.csv'
    df1 = pd.DataFrame.from_dict(dvalues)
    df2 = pd.DataFrame.from_dict(derrors)
    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)


def comomo_from_csv(path):
    path1 =f'{path}/cmvalues.csv'
    path2 =f'{path}/cmerrors.csv'
    dfv = pd.read_csv(path1)
    dfe = pd.read_csv(path2)
    dates = dfv['Date'].values
    npdates =[np.datetime64(d) for d in dates]
    dfv['Date'] = npdates
    dfe['Date'] = npdates
    dict_of_df= {}
    dict_of_df['values'] = dfv
    dict_of_df['errors'] = dfe
    df = pd.concat(dict_of_df, axis=1)
    return dfv, dfe, df
