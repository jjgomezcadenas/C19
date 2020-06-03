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

url_covid_data = "https://cnecovid.isciii.es/covid19/resources/agregados.csv"

def isc3_get_data_covid(datapath="../data/covid_data.csv", update=False):

    # If we're just reading (not updating) the data, just read it from the CSV.
    if(not update):
        if(not os.path.isfile(datapath)):
            print("File",datapath,"does not exist. Run this function with update=True to retrieve the data.")
            return None
        df = pd.read_csv(datapath, error_bad_lines=False)
        return df

    # Read in the latitude/longitude dataframe for all countries.
    print(f"Reading covid data from {url_covid_data}")
    covid_file="../data/covid_data.csv"
    urllib.request.urlretrieve (url_covid_data, filename=covid_file)
    if(not os.path.isfile(covid_file)):
        print("ERROR downloading covid_file data.")
        return None
    print("-- Done")

    # Read in the C19 world data.
    df_covid = pd.read_csv(covid_file, error_bad_lines=False,
                           parse_dates = [2],
                           infer_datetime_format = True,
                           dayfirst = True,
                           encoding = "ISO-8859-1")

    return df_covid


def isc3_get_data(dc):
    dates = dc['FECHA'].values
    dead  = dc['Fallecidos'].values
    ccaa  = dc['CCAA'].values
    npdates = []
    cdead   = []

    CCAA    = []
    dp, mp, yp = 0, 0, 0
    d0 = 0
    for i, d in enumerate(dates):
        if type(d) != str:
            break
        else:
            day, month, year = d.split("/")
            dp, mp, yp = day, month, year
            dt = datetime.datetime(int(year), int(month), int(day))
            npdates.append(np.datetime64(dt))
            cdead.append(dead[i])
            CCAA.append(ccaa[i])
            d0 = dead[i]
    return pd.DataFrame({"ccaa":CCAA, "date":npdates, "cdead":cdead}).dropna()


def isc3_select_ca_and_date_xdead(dm, ca_code='MD', datei='2020-03-10', datef='2020-06-10'):
    df    = dm.loc[dm.ccaa == ca_code, ('ccaa', 'date','cdead')]
    dfx   = df.loc[df.date >= np.datetime64(datei), ('ccaa', 'date','cdead')]
    dft   = dfx.loc[dfx.date < np.datetime64(datef), ('ccaa', 'date','cdead')]
    dead  = dft['cdead'].values
    xdead = []
    d0 = 0
    for d in dead:
        xdead.append(d - d0)
        d0 = d

    dft['dead'] = xdead
    return dft


def isc3_get_ccaa_data(dm, datei='2020-03-10', datef='2020-06-10'):
    return{ccaa:isc3_select_ca_and_date_xdead(dm, dmomo[ccaa], datei, datef) for ccaa in dmomo.keys()}
