from dataclasses import dataclass, field
import abc
import numpy as np
import pandas as pd
from scipy.linalg import norm
from scipy.special import erfc
from  . system_of_units import *

from typing      import Tuple
from typing      import Dict
from typing      import List
from typing      import TypeVar
from typing      import Optional
from enum        import Enum

Number = TypeVar('Number', None, int, float)
Str    = TypeVar('Str', None, str)
Range  = TypeVar('Range', None, Tuple[float, float])
Array  = TypeVar('Array', List, np.array)
Int    = TypeVar('Int', None, int)
Point  = TypeVar('Point', List, np.array)
Vector = TypeVar('Vector', List, np.array)

EPS = 1e-7  # default tolerance


c19_dict = {
    "Andalucia"         : {"geoId": "AN", "ca_code":"AN", "countryterritoryCode": "AND", "popData2018": 8384408},
    "Aragon"            : {"geoId": "AR", "ca_code":"AR","countryterritoryCode": "ARA", "popData2018": 1308728},
    "Asturias"          : {"geoId": "AS", "ca_code":"AS","countryterritoryCode": "AST", "popData2018": 1028244},
    "Baleares"          : {"geoId": "BA", "ca_code":"IB","countryterritoryCode": "BAL", "popData2018": 1128908},
    "Canarias"          : {"geoId": "CN", "ca_code":"CN","countryterritoryCode": "CAN", "popData2018": 2127685},
    "Cantabria"         : {"geoId": "CT", "ca_code":"CB","countryterritoryCode": "CAB", "popData2018": 580229},
    "Castilla La Mancha": {"geoId": "CM", "ca_code":"CM","countryterritoryCode": "CLM", "popData2018": 2026807},
    "Castilla y Leon"   : {"geoId": "CL", "ca_code":"CL","countryterritoryCode": "CYL", "popData2018": 2409164},
    "Cataluna"          : {"geoId": "CA", "ca_code":"CT","countryterritoryCode": "CAT", "popData2018": 7600065},
    "Ceuta"             : {"geoId": "CE", "ca_code":"CE","countryterritoryCode": "CEU", "popData2018": 85144},
    "C. Valenciana"     : {"geoId": "CV", "ca_code":"VC","countryterritoryCode": "CVA", "popData2018": 4963703},
    "Extremadura"       : {"geoId": "EX", "ca_code":"EX","countryterritoryCode": "EXT", "popData2018": 1072863},
    "Galicia"           : {"geoId": "GA", "ca_code":"GA","countryterritoryCode": "GAL", "popData2018": 2701743},
    "Madrid"            : {"geoId": "MA", "ca_code":"MD","countryterritoryCode": "MAD", "popData2018": 6578079},
    "Melilla"           : {"geoId": "ME", "ca_code":"ML","countryterritoryCode": "MEL", "popData2018": 86384},
    "Murcia"            : {"geoId": "MU", "ca_code":"MC","countryterritoryCode": "MUR", "popData2018": 1478509},
    "Navarra"           : {"geoId": "NA", "ca_code":"NC","countryterritoryCode": "NAV", "popData2018": 647554},
    "Pais Vasco"        : {"geoId": "PV", "ca_code":"PV","countryterritoryCode": "PVA", "popData2018": 2199088},
    "La Rioja"          : {"geoId": "LR", "ca_code":"RI","countryterritoryCode": "RIO", "popData2018": 315675}
}


def get_dicts(c19_dict):
    c19d = {}
    isc3d = {}
    for caaa_name, caa_dict in c19_dict.items():
        c19d[caaa_name] = caa_dict["geoId"]
        isc3d[caaa_name] = caa_dict["ca_code"]

    return c19d, isc3d

c19d, isc3d = get_dicts(c19_dict)

class Verbosity(Enum):
    mute     = 0
    warning  = 1
    info     = 2
    verbose  = 3
    vverbose = 4


class DrawLevel(Enum):
    nodraw        = 0
    geometry      = 1
    sourceRays    = 2
    refractedRays = 3
    reflectedRays = 4


def default_field(obj):
    return field(default_factory=lambda: obj)


def vprint(msg, verbosity, level=Verbosity.mute):
    if verbosity.value <= level.value and level.value >0:
        print(msg)


def vpblock(msgs, verbosity, level=Verbosity.mute):
    for msg in msgs:
        vprint(msg, verbosity, level)


def draw_level(drawLevel, level):
    if drawLevel.value >= level.value:
        return True
    else:
        return False

@dataclass
class Country:
    """Conutry data"""
    name     : str
    code     : str


@dataclass
class SIR:
    """SIR model of an epidemics"""
    N     : float  # total population
    S     : np.array  # population susceptible of infection
    I     : np.array  # population infected
    R     : np.array  # population recovered
    t     : np.array  # time axis
    R0    : float     # basic reproduction number
    beta  : float     # beta  = R0/T = R0 * gamma
    gamma : float     # gamma = 1/T


@dataclass
class SIR2(SIR):
    """SEIR model of an epidemics"""
    D     : np.array  # population in track to die
    M     : np.array  # dead
    P     : np.array  # Perception of risk
    phi   : float     # case fatality proportion CFP
    g     : float     # 1/g mean time from loss of inf to death
    lamda : float     # 1/landa mean duration of impact of death on population
    k     : float     # Parameter controlling the intensity of response


@dataclass
class SEIR(SIR):
    """SEIR model of an epidemics"""
    E     : np.array  # population exposed
    sigma : float     # sigma = 1/Ti


@dataclass
class SEIR2(SEIR):
    """SEIR extended model of an epidemics"""
    D     : np.array  # population in track to die
    M     : np.array  # dead
    P     : np.array  # Perception of risk
    phi   : float     # case fatality proportion CFP
    g     : float     # 1/g mean time from loss of inf to death
    lamda : float     # 1/landa mean duration of impact of death on population
    k     : float     # Parameter controlling the intensity of response
