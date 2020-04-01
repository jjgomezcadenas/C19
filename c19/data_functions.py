import numpy as np
import http.client
import ast
import pandas as pd
import datetime
import os
import urllib

# C19 data files.
url_c19_files      = "https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019"
c19_file_cases     = "ccaa_covid19_casos.csv"
c19_file_deaths    = "ccaa_covid19_fallecidos.csv"
c19_file_uci       = "ccaa_covid19_uci.csv"
c19_file_hosp      = "ccaa_covid19_hospitalizados.csv"
c19_file_recovered = "ccaa_covid19_altas.csv"

# Define the dictionary associating a weather sensor to each region.
sensor_dict = {
    "Andalucia"         : "5402" , # CORDOBA/AEROPUERTO
    "Aragon"            : "9434" , # ZARAGOZA/AEROPUERTO
    "Asturias"          : "1208H", # GIJON, MUSEL
    "Baleares"          : "B278" , # PALMA DE MALLORCA/SON SAN JUAN
    "Canarias"          : "C029O", # LANZAROTE/AEROPUERTO
    "Cantabria"         : "1111" , # SANTANDER I,CMT
    "Castilla-La Mancha": "4121" , # CIUDAD REAL
    "Castilla y Leon"   : "2422" , # VALLADOLID
    "Cataluna"          : "0016A", # REUS/AEROPUERTO
    "Ceuta"             : "5000C", # CEUTA
    "C. Valenciana"     : "8414A", # VALENCIA/AEROPUERTO
    "Extremadura"       : "3469A", # CACERES
    "Galicia"           : "1428" , # SANTIAGO DE COMPOSTELA/LABACOLLA
    "Madrid"            : "3200" , # MADRID/GETAFE
    "Melilla"           : "6000A", # MELILLA
    "Murcia"            : "7178I", # MURCIA
    "Navarra"           : "9263D", # PAMPLONA/NOAIN
    "Pais Vasco"        : "1024E", # SAN SEBASTIAN,IGUELDO
    "La Rioja"          : "9170"   # LOGRONO/AGONCILLO
}

# Dictionary of miscellaneous information.
# Population data from: Cifras oficiales de población resultantes de la revisión del Padrón municipal a 1 de enero (year 2018)
misc_dict = {
    "Andalucia"         : {"geoId": "AN", "countryterritoryCode": "AND", "popData2018": 8384408},
    "Aragon"            : {"geoId": "AR", "countryterritoryCode": "ARA", "popData2018": 1308728},
    "Asturias"          : {"geoId": "AS", "countryterritoryCode": "AST", "popData2018": 1028244},
    "Baleares"          : {"geoId": "BA", "countryterritoryCode": "BAL", "popData2018": 1128908},
    "Canarias"          : {"geoId": "CN", "countryterritoryCode": "CAN", "popData2018": 2127685},
    "Cantabria"         : {"geoId": "CT", "countryterritoryCode": "CAB", "popData2018": 580229},
    "Castilla-La Mancha": {"geoId": "CM", "countryterritoryCode": "CLM", "popData2018": 2026807},
    "Castilla y Leon"   : {"geoId": "CL", "countryterritoryCode": "CYL", "popData2018": 2409164},
    "Cataluna"          : {"geoId": "CA", "countryterritoryCode": "CAT", "popData2018": 7600065},
    "Ceuta"             : {"geoId": "CE", "countryterritoryCode": "CEU", "popData2018": 85144},
    "C. Valenciana"     : {"geoId": "CV", "countryterritoryCode": "CVA", "popData2018": 4963703},
    "Extremadura"       : {"geoId": "EX", "countryterritoryCode": "EXT", "popData2018": 1072863},
    "Galicia"           : {"geoId": "GA", "countryterritoryCode": "GAL", "popData2018": 2701743},
    "Madrid"            : {"geoId": "MA", "countryterritoryCode": "MAD", "popData2018": 6578079},
    "Melilla"           : {"geoId": "ME", "countryterritoryCode": "MEL", "popData2018": 86384},
    "Murcia"            : {"geoId": "MU", "countryterritoryCode": "MUR", "popData2018": 1478509},
    "Navarra"           : {"geoId": "NA", "countryterritoryCode": "NAV", "popData2018": 647554},
    "Pais Vasco"        : {"geoId": "PV", "countryterritoryCode": "PVA", "popData2018": 2199088},
    "La Rioja"          : {"geoId": "LR", "countryterritoryCode": "RIO", "popData2018": 315675}
}

# Get the weather dataframe for the specified station and date range.
def get_meteo_df(station,date_init,date_final,api_key):

    # Send the initial request.
    conn = http.client.HTTPSConnection("opendata.aemet.es")
    request_str = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{}/fechafin/{}/estacion/{}/?api_key={}".format(date_init,date_final,station,api_key)
    headers = {'cache-control': "no-cache"}
    conn.request("GET", request_str, headers=headers)

    # Interpret the response.
    res_init = conn.getresponse()
    data_init = res_init.read()
    dict_init = ast.literal_eval(data_init.decode("utf-8"))
    url_init = dict_init['datos']
    url_meta = dict_init['metadatos']

    # Send the request for the metadata.
    #print("Requesting metadata from:",url_meta)
    conn.request("GET", url_meta, headers=headers)

    res_meta = conn.getresponse()
    data_meta = res_meta.read()
    dict_meta = data_meta.decode("ISO-8859-1")
    #print(dict_meta)

    # Send the request for the data.
    #print("Requesting data from:",url_init)
    conn.request("GET", url_init, headers=headers)

    # Interpret the response.
    res_final = conn.getresponse()
    data_final = res_final.read()
    dict_data = ast.literal_eval(data_final.decode("ISO-8859-1"))

    return pd.DataFrame(dict_data)

def prepare_meteo_df(df):

    # Check that all required keys exist in the dataframe.
    required_keys = ['fecha', 'prec', 'sol', 'tmax', 'tmed', 'tmin']
    for rk in required_keys:
        if(not (rk in df)):
            print("Warning: dataframe missing",rk)
            return None

    # Extract required elements.
    meteo = df[required_keys].copy()

    # Replace comma with dot.
    meteo[['prec', 'sol', 'tmax', 'tmed', 'tmin']] = meteo[['prec', 'sol', 'tmax', 'tmed', 'tmin']].apply(lambda x: x.str.replace(',','.'))

    # Replace Ip with 0.0.
    meteo[['prec']] = meteo[['prec']].apply(lambda x: x.str.replace('Ip','0.0'))

    # Convert to numerical values.
    meteo[['prec','sol','tmax','tmed','tmin']] = meteo[['prec','sol','tmax','tmed','tmin']].astype('float')

    # Convert dates to datetime objects.
    meteo['fecha'] = pd.to_datetime(meteo['fecha'], format="%Y-%m-%d")

    return meteo

def get_data_communities(api_key, datapath="../data/data_communities.csv", update=False):

    # If we're just reading (not updating) the data, just read it from the CSV.
    if(not update):
        if(not os.path.isfile(datapath)):
            print("File",datapath,"does not exist. Run this function with update=True to retrieve the data.")
            return None
        cdf = pd.read_csv(datapath)
        cdf.drop("Unnamed: 0", axis=1, inplace=True)
        cdf['dateRep'] = pd.to_datetime(cdf['dateRep'], format="%Y-%m-%d")
        return cdf

    # Get the meteo data.
    meteo_regions = {}
    date_init = "2020-02-27T00:00:00UTC"
    date_final = "{}T23:59:59UTC".format(datetime.datetime.today().strftime('%Y-%m-%d'))
    print("Obtaining meteo data...")
    for region,station in sensor_dict.items():
        print(region,station)
        df = get_meteo_df(station,date_init,date_final,api_key)
        meteo = prepare_meteo_df(df)
        meteo_regions[region] = meteo
    print("-- Done")

    # Retrieve the C19 data.
    print("Downloading C19 data...")
    urllib.request.urlretrieve ("{}/{}".format(url_c19_files,c19_file_cases), c19_file_cases)
    urllib.request.urlretrieve ("{}/{}".format(url_c19_files,c19_file_deaths), c19_file_deaths)
    urllib.request.urlretrieve ("{}/{}".format(url_c19_files,c19_file_uci), c19_file_uci)
    urllib.request.urlretrieve ("{}/{}".format(url_c19_files,c19_file_hosp), c19_file_hosp)
    urllib.request.urlretrieve ("{}/{}".format(url_c19_files,c19_file_recovered), c19_file_recovered)
    if(not (os.path.isfile(c19_file_cases) and os.path.isfile(c19_file_deaths) and os.path.isfile(c19_file_uci) and os.path.isfile(c19_file_hosp) and os.path.isfile(c19_file_recovered))):
        print("ERROR downloading C19 data.")
        return None
    print("-- Done")

    # Read in the C19 data.
    cases  = pd.read_csv(c19_file_cases);     os.remove(c19_file_cases)
    ucases = pd.read_csv(c19_file_deaths);    os.remove(c19_file_deaths)
    fcases = pd.read_csv(c19_file_uci);       os.remove(c19_file_uci)
    hcases = pd.read_csv(c19_file_hosp);      os.remove(c19_file_hosp)
    rcases = pd.read_csv(c19_file_recovered); os.remove(c19_file_recovered)

    # Remove all accents from the region names.
    cases['CCAA']  = cases['CCAA'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    ucases['CCAA'] = ucases['CCAA'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    fcases['CCAA'] = fcases['CCAA'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    hcases['CCAA'] = hcases['CCAA'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    rcases['CCAA'] = rcases['CCAA'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    # Set the region name as index.
    cases  = cases.set_index('CCAA')
    ucases = ucases.set_index('CCAA')
    fcases = fcases.set_index('CCAA')
    hcases = hcases.set_index('CCAA')
    rcases = rcases.set_index('CCAA')

    # Add the C19 data to the meteo dataframes.
    df_regions = {}
    print("Combining C19 and meteo data...")
    for region,df in meteo_regions.items():

        # Get a new dataframe of cases with the dates and # of cases as columns.
        cframe = pd.DataFrame({'ncases'        : cases.loc[region][1:].values,
                               'fecha'         : cases.loc[region].keys()[1:].values})
        uframe = pd.DataFrame({'uci'           : ucases.loc[region][1:].values,
                               'fecha'         : ucases.loc[region].keys()[1:].values})
        fframe = pd.DataFrame({'fallecidos'    : fcases.loc[region][1:].values,
                               'fecha'         : fcases.loc[region].keys()[1:].values})
        hframe = pd.DataFrame({'hospitalizados': hcases.loc[region][1:].values,
                               'fecha'         : hcases.loc[region].keys()[1:].values})
        rframe = pd.DataFrame({'altas'         : rcases.loc[region][1:].values,
                               'fecha'         : rcases.loc[region].keys()[1:].values})

        # Change the dates to datetime objects.
        cframe['fecha'] = pd.to_datetime(cframe['fecha'], format="%Y-%m-%d")
        uframe['fecha'] = pd.to_datetime(uframe['fecha'], format="%Y-%m-%d")
        fframe['fecha'] = pd.to_datetime(fframe['fecha'], format="%Y-%m-%d")
        hframe['fecha'] = pd.to_datetime(hframe['fecha'], format="%Y-%m-%d")
        rframe['fecha'] = pd.to_datetime(rframe['fecha'], format="%Y-%m-%d")

        # Merge the dataframes.
        mdf = pd.merge(df,  cframe, on = 'fecha', how='outer')
        mdf = pd.merge(mdf, uframe, on = 'fecha', how='outer')
        mdf = pd.merge(mdf, fframe, on = 'fecha', how='outer')
        mdf = pd.merge(mdf, hframe, on = 'fecha', how='outer')
        mdf = pd.merge(mdf, rframe, on = 'fecha', how='outer')
        df_regions[region] = mdf
    print("-- Done")

    # Merge all the community dataframes.
    cdf = None
    print("Merging into a single dataframe...")
    for key in df_regions.keys():

        # Add the misc information to this dataframe.
        cframe = df_regions[key]
        cframe['countriesAndTerritories'] = key
        cframe['geoId']                   = misc_dict[key]['geoId']
        cframe['countryterritoryCode']    = misc_dict[key]['countryterritoryCode']
        cframe['popData2018']             = misc_dict[key]['popData2018']

        if(cdf is None):
            cdf = cframe
        else:
            cdf = cdf.append(cframe)

    # Reset the index count.
    cdf = cdf.reset_index()

    # Change column names.
    cdf = cdf.rename(columns={"fecha": "dateRep", "ncases": "cases", "fallecidos": "deaths", "hospitalizados": "hospitalized", "altas": "recovered"})

    # Add columns for day, month, and year.
    cdf['day']   = cdf.apply(lambda row: row['dateRep'].date().day, axis=1)
    cdf['month'] = cdf.apply(lambda row: row['dateRep'].date().month, axis=1)
    cdf['year']  = cdf.apply(lambda row: row['dateRep'].date().year, axis=1)
    print("-- Done")

    # Save the final dataframe to file.
    print("Saving the final dataframe to",datapath,"...")
    cdf.to_csv(datapath)
    print("-- Done")

    return cdf
