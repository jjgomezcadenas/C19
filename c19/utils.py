import numpy as np
from   typing  import Tuple, List
from numpy import sqrt
import datetime 

NN = np.nan
from . types  import Number, Array, Str, Range

def date_spanish_str_to_datetime(d):
    if type(d) != str: # this may be a nan because patient still not released set it to infnt
        dt = datetime.datetime(int(2030), int(1), int(1))
    else:
        day, month, year = d.split("/")
        dp, mp, yp = day, month, year
        dt = datetime.datetime(int(year), int(month), int(day))
    return dt


def get_I_and_R_CAA(dca, N, I, R, norm=True):
    """Get the I and R vectors from the DF of CCAA"""
    In = I/N
    Rn = R/N
    if norm== False:
        In = I
        Rn = R
    Icaa = [In * p for p in dca['2019'].values]
    Rcaa = [Rn * p for p in dca['2019'].values]
    return Icaa, Rcaa


def uci_cases(IC, f_uci = 0.05):
    """Fraction of infected in UCI"""
    return [I * f_uci for I in IC]


def uci_beds(dca, t_beds = 4404):
    """Available UCI beds in Spain (proportional calculation)"""
    tot = dca['2019'].sum()
    beds_capita = t_beds / tot
    print(f'Población total España ={tot:.2e}')
    beds = [beds_capita * p for p in dca['2019'].values]
    print(f'Beds UCI per CA = {beds}')
    return beds
