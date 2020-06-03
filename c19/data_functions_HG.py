"""
Data from HG (Hospitales Gallegos)

-
"""
import pandas as pd
import datetime
import os
import urllib
import json
import numpy as np

from . utils import date_spanish_str_to_datetime

def hg_get_data_ingresos(datapath):
    dfi = pd.read_csv(datapath, sep='|')
    P  = dfi['Paciente'].values
    sI = dfi['Ingreso'].values
    sA = dfi['Alta'].values
    H  = dfi['Hosp'].values

    dI = [date_spanish_str_to_datetime(d) for d in sI]
    dA = [date_spanish_str_to_datetime(d) for d in sA]

    return pd.DataFrame({"pacienteID":P,"ingreso":dI,"alta":dA,"hosp":H})


def hg_get_data_pacientes(datapath):
    dfi = pd.read_csv(datapath, sep='|')
    P  = dfi['Paciente'].values
    sI = dfi['Ingreso'].values
    sA = dfi['Alta'].values
    E  = dfi['Edad'].values
    S  = dfi['Sexo'].values
    C  = dfi['Code'].values

    dI = [date_spanish_str_to_datetime(d) for d in sI]
    dA = [date_spanish_str_to_datetime(d) for d in sA]

    return pd.DataFrame({"pacienteID":P,"ingreso":dI,"alta":dA,"edad":E,"sexo":S,"code":C})


def hg_get_data_age(datapath):
    dfi = pd.read_csv(datapath, sep=';')
    S  = dfi['Sexo'].values
    CA = dfi['Comunidades y provincias'].values
    E = dfi['Edad (hasta 100 y más)'].values
    T  = dfi['Total'].values


    return pd.DataFrame({"sexo":S,"ca":CA,"edad":E,"poblacion":T})
