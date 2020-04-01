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
