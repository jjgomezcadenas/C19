import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import requests

def read_ecdc():
    d = date.today()
    dls = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx"
    resp = requests.get(dls)
    of = f'../data/COVID-19-geographic-disbtribution-worldwide-{d.isoformat()}.xlsx'
    print(f'writing file {of}')
    output = open(of, 'wb')
    output.write(resp.content)
    output.close()
    df = pd.read_excel(of)
    return df


def ecdc_select_country(df, country='ES', thr=2):
    dfs = df.loc[df['geoId'] == country]
    return dfs.loc[df['cases']>thr]
