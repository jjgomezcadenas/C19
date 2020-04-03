"""
Gets C19 and associated weather data for the world.

- C19 data source: https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
- Weather data source: https://power.larc.nasa.gov/
- Latitude/longitude data source: https://developers.google.com/public-data/docs/canonical/countries_csv

Weather data explanation and units (from source):
'IR': Downward Thermal Infrared (Longwave) Radiative Flux (kW-hr/m^2/day)
'sol': All Sky Insolation Incident on a Horizontal Surface (kW-hr/m^2/day)
'prec': Precipitation (mm/day)
'shum': Specific Humidity at 2 Meters (kg kg-1)
'rhum': Relative Humidity at 2 Meters (%)
'tmed': Temperature at 2 Meters (C)
'tmax': Maximum Temperature at 2 Meters (C)
'tmin': Minimum Temperature at 2 Meters (C)

"""
import pandas as pd
import datetime
import os
import urllib
import json
import numpy as np

# URL for obtaining the C19 data.
url_c19data = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"

def get_data_world(datapath="../data/data_world.csv", latlongpath="../data/world_latlong.csv", update=False):

    # If we're just reading (not updating) the data, just read it from the CSV.
    if(not update):
        if(not os.path.isfile(datapath)):
            print("File",datapath,"does not exist. Run this function with update=True to retrieve the data.")
            return None
        cdf = pd.read_csv(datapath)
        cdf.drop("Unnamed: 0", axis=1, inplace=True)
        cdf['dateRep'] = pd.to_datetime(cdf['dateRep'], format="%Y-%m-%d")
        return cdf

    # Read in the latitude/longitude dataframe for all countries.
    print("Reading latitude/longitude data from",latlongpath,"...")
    df_latlong = pd.read_csv(latlongpath)

    # Download the C19 data.
    print("Downloading C19 world data...")
    c19_file_world = "c19data_world.csv"
    urllib.request.urlretrieve ("{}".format(url_c19data), c19_file_world)
    if(not os.path.isfile(c19_file_world)):
        print("ERROR downloading C19 world data.")
        return None
    print("-- Done")

    # Read in the C19 world data.
    df_world = pd.read_csv(c19_file_world); os.remove(c19_file_world)

    # Change the dates to datetime objects.
    df_world['dateRep'] = pd.to_datetime(df_world['dateRep'], format="%d/%m/%Y")

    # Obtain climate data for all countries in the latitude/longitude list.
    yesterday = datetime.datetime.today() - datetime.timedelta(days=1)
    date_init = "20191231"
    date_final = yesterday.strftime('%Y%m%d')
    print("Obtaining climate data from",date_init,"to",date_final,"...")

    cdf = None
    countries_latlong = set(df_latlong.country.values)
    countries = set([(ccode,cname) for ccode,cname in zip(df_world.geoId.values,df_world.countriesAndTerritories.values)])
    icountry = 0
    ncountries = len(countries)
    for ccode,cname in countries:
        icountry += 1

        print("[{}/{}] Country {} ({}) ...".format(icountry,ncountries,cname,ccode))
        if(str(ccode) != 'nan' and ccode in countries_latlong):
            lat  = df_latlong[df_latlong.country == ccode]['latitude'].values[0]
            long = df_latlong[df_latlong.country == ccode]['longitude'].values[0]
            print("- Requesting data for (latitude,longitude) = ({},{})".format(lat,long))
        else:
            print("* Skipping country code",ccode,": not in latitude/longitude dataframe")
            continue

        request_str = "https://power.larc.nasa.gov/cgi-bin/v1/DataAccess.py?&request=execute&identifier=SinglePoint&parameters=PRECTOT,QV2M,RH2M,T2M_MAX,T2M_MIN,T2M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_LW_DWN&startDate={}&endDate={}&userCommunity=SSE&tempAverage=DAILY&outputList=JSON&lat={}&lon={}".format(date_init,date_final,lat,long)
        response = urllib.request.urlopen(request_str)
        data = json.loads(response.read())

        # Create the dataframe.
        df = pd.DataFrame(data['features'][0]['properties']['parameter'])

        # Rename the columns.
        df = df.reset_index().rename(columns={"index": "dateRep", "ALLSKY_SFC_LW_DWN": "IR", "ALLSKY_SFC_SW_DWN": "sol",
                                         "PRECTOT": "prec", "QV2M": "shum", "RH2M": "rhum", "T2M": "tmed",
                                         "T2M_MAX": "tmax", "T2M_MIN": "tmin"})

        # Convert date to datetime objects.
        df['dateRep'] = pd.to_datetime(df['dateRep'], format="%Y%m%d")

        # Add the latitude and longitude.
        df_country = df_world[df_world.geoId == ccode]
        df_country['latitude']  = lat
        df_country['longitude'] = long

        # Merge with the same country code in the world dataframe (merging "left" drops entries with no C19 data).
        df_merge = pd.merge(df_country, df, on = 'dateRep', how='left')

        # Append to the final combined dataframe.
        if(cdf is None):
            cdf = df_merge
        else:
            cdf = cdf.append(df_merge)
    print("-- Done")

    # Save the final dataframe to file.
    print("Saving the final dataframe to",datapath,"...")
    cdf.to_csv(datapath)
    print("-- Done")

    return cdf
