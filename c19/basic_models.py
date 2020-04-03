import numpy as np
from   typing  import Tuple, List
from numpy import sqrt

NN = np.nan
from . types  import Number, Array, Str, Range

from scipy.integrate import odeint
import scipy.integrate as spi
from scipy.interpolate import interp1d

from . types import SIR, SEIR, SEIR2

def sir_deriv(y, t, M, beta, gamma):
    """Prepares the SIR system of equations"""
    S, I, R = y
    dSdt = -beta * M(t) * S * I
    dIdt = beta * M(t) * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def set_sir_initial_conditions(N, i0=1, r0=0):
    """Set initial conditions vector where
    N  is the total population
    i0 is the initial number of infected individuals
    r0 is the initial number of recovered individuals
    """
    s0 = N - i0 - r0
    y0 = s0, i0, r0
    return y0


def compute_sir(N, Y0, R0, Gamma, t_range, ts = [(0, 400)], ms=[1.0]):
    """Full SIR run"""

    Beta      = Gamma * R0
    M = mitigation_function(t_range, ts, ms)
    ret = odeint(sir_deriv, Y0, t_range, args=(M, Beta, Gamma))
    S, I, R = ret.T

    sir = SIR(N = N, S=S, I=I,  R=R,
              beta=Beta, R0=R0, gamma=Gamma,  t= t_range)

    return sir


def seir_deriv(y, t, beta, gamma, sigma):
    """
    Prepare differential equations for SEIR

    """
    S, E, I = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    return dSdt, dEdt, dIdt


def mitigation_function(t, ts = [(0, 400)], ms=[1]):
    """Defines a mitigation function as follows:
    1) the time of the infection, t, is divided in tranches specified by ts. Thus, for example,
    ts =[(0,20), (20,60), (60,100)] would divide a time vector of 100 days in 3 tranches,
    the first one
    from day 0 to day 20 and so on.
    2) ms corresponds to the mitigation factor in each tranch. Thus, [(1, 0.5, 0.8)] in the
    example would
    mean that the first tranch no mitigation is applied (R0 * 1), and for second and third
    tranches the
    mitigation would be (R0 * 0.5) and (R0 * 0.8)
    """
    def mitigation(t, ts, ms):
        lt = len(t)
        lm = len(ms)
        x = int(lt /lm)
        c = np.ones(lt)
        for i, tsi in enumerate(ts):
            c[tsi[0]:tsi[1]] = ms[i]
        c[-1] = ms[-1]
        return c

    C = mitigation(t, ts, ms)
    M = interp1d(t, C, bounds_error=False, fill_value="extrapolate")
    return M


def seir_deriv_time(y, t, M, beta, gamma, sigma):
    """
    Prepare differential equations for SEIR
    Includes a mitigation function
    """

    S, E, I = y
    dSdt = -beta * M(t) * S * I
    dEdt = beta * M(t) * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    return dSdt, dEdt, dIdt


def compute_seir(N, Y0, R0, Gamma, Sigma, t_range, ts = [(0, 400)], ms=[1.0]):
    """Full SEIR run"""

    Beta      = Gamma * R0
    M = mitigation_function(t_range, ts, ms)
    RES       = odeint(seir_deriv_time, Y0, t_range, args=(M, Beta, Gamma, Sigma))
    S, E, I   = RES.T
    R         = 1 - S - E - I

    seir      = SEIR(N = N, S=S, I=I, E=E, R=R,
                     beta=Beta, R0=R0, gamma=Gamma, sigma = Sigma, t= t_range)
    return seir


def seir2_deriv_time(y, t, M, beta, gamma, sigma, phi, g, lamda):
    """
    Prepare differential equations for SEIR
    Includes a mitigation function
    """

    S, E, I, R, D, P = y
    dSdt = -beta * M(t) * S * I
    dEdt = beta * M(t) * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = (1 - phi) * gamma * I
    dDdt = phi * gamma * I - g * D
    dPdt = g * D  - lamda * P
    return dSdt, dEdt, dIdt, dRdt, dDdt, dPdt


def compute_seir2(N, Y0, R0, Gamma, Sigma, Phi, G, Lamda, k,
                  t_range, ts = [(0, 400)], ms=[1.0]):
    """Full SEIR run"""

    #Beta           = Gamma * R0 * (1 - P)**k
    Beta           = Gamma * R0
    M              = mitigation_function(t_range, ts, ms)
    RES            = odeint(seir2_deriv_time, Y0, t_range,
                            args=(M, Beta, Gamma, Sigma, Phi, G, Lamda))
    S, E, I, R, D, P  = RES.T
    M              = 1 - S - E - I - R - D

    seir      = SEIR2(N = N, S=S, I=I, E=E, R=R, D=D, M=M, P=P,
                     beta=Beta, R0=R0, gamma=Gamma, sigma = Sigma,
                     phi=Phi, g=G, lamda=Lamda, k=k,
                     t= t_range)

    return seir
