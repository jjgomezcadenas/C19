import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . c19stats import lognorm_pdf, hdt
from . data_functions import misc_dict
from . io import ecdc_select_country


def set_pretty_ax(ax, facecolor, xlabel, ylabel, xmin, xmax, ymax):
    ax.set_facecolor(facecolor)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,ymax)
    ax.set_xlim(xmin,xmax)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

def plot_sir(sir, T, tmin = 0, tmax = 200, absolute=True, figsize=(10,10), facecolor='LightGrey'):
    """Plot the data on three separate curves for S(t), I(t) and R(t)"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, axisbelow=True)
    ax.set_facecolor(facecolor)

    if absolute:
        S = sir.S * sir.N
        I = sir.I * sir.N
        R = sir.R * sir.N
        Y = 1.1 * sir.N
    else:
        S = sir.S
        I = sir.I
        R = sir.R
        Y = 1.1

    ax.plot(sir.t, S, 'b', alpha=0.5, lw=3, label='Susceptible')
    ax.plot(sir.t, I, 'r', alpha=0.5, lw=3, label='Infected')
    ax.plot(sir.t, R, 'g', alpha=0.5, lw=3, label='Recovered')
    xlabel = 'Time /days'
    if absolute:
        ylabel = 'Total population'
    else:
        ylabel = 'Fraction of population'

    set_pretty_ax(ax, facecolor, xlabel, ylabel, tmin, tmax, Y)
    plt.title(T)
    plt.show()


def plot_sirs(SIRs,Ts, figsize=(10,10), facecolor='LightGrey', lw=2, ylim=0.4):
    """Plot the data for various values of SIR"""

    def set_ax(ax):
        ax.set_facecolor(facecolor)
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Fraction of population')
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)


    fig = plt.figure(figsize=figsize)

    ax  = plt.subplot(2,1,1, axisbelow=True)
    for i, sir in enumerate(SIRs):
        ax.plot(sir.t, sir.I/sir.N, alpha=0.5, lw=lw, label=Ts[i])
    set_ax(ax)
    ax.set_ylim(0,ylim)

    ax  = plt.subplot(2,1,2, axisbelow=True)
    for i, sir in enumerate(SIRs):
        ax.plot(sir.t, sir.R/sir.N, alpha=0.5, lw=lw, label=Ts[i])
    set_ax(ax)
    ax.set_ylim(0,1.1)

    plt.tight_layout()
    plt.show()


def plot_ICAA(dca, uci_beds, t, IC, figsize=(10,10), facecolor='LightGrey'):
    """Plot the Is for various configs"""
    fig = plt.figure(figsize=figsize)

    for i in range(len(IC) -1) :
        ax = fig.add_subplot(6,3, i+1, axisbelow=True)
        ax.set_facecolor(facecolor)
        ax.plot(t, IC[i], alpha=0.5, lw=3, label=dca["CCAA"][i])
        ax.set_xlabel('t (días)')
        ax.set_ylabel('UCI')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.axhline(y = uci_beds[i], linewidth=2, color='r')
        ax.yaxis.major.formatter._useMathText = True
        plt.title(dca["CCAA"][i])

    plt.tight_layout()
    plt.show()


def plot_seir(sir, T, tmin = 0, tmax = 200, absolute=True, figsize=(10,10), facecolor='LightGrey'):
    """Plot the data on four separate curves for S(t), E(t), I(t) and R(t)"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, axisbelow=True)
    ax.set_facecolor(facecolor)

    if absolute:
        S = sir.S * sir.N
        E = sir.E * sir.N
        I = sir.I * sir.N
        R = sir.R * sir.N
        Y = 1.1 * sir.N
    else:
        S = sir.S
        E = sir.E
        I = sir.I
        R = sir.R
        Y = 1.1

    ax.plot(sir.t, S, 'k', alpha=0.5, lw=3, label='Susceptible')
    ax.plot(sir.t, I, 'b', alpha=0.5, lw=3, label='Infected')
    ax.plot(sir.t, E, 'r', alpha=0.5, lw=3, label='Exposed')
    ax.plot(sir.t, R, 'g', alpha=0.5, lw=3, label='Recovered')
    xlabel = 'Time /days'
    if absolute:
        ylabel = 'Total population'
    else:
        ylabel = 'Fraction of population'

    set_pretty_ax(ax, facecolor, xlabel, ylabel, tmin, tmax, Y)
    plt.title(T)
    plt.show()


def plot_seir2(sir, T, figsize=(10,10), facecolor='LightGrey'):
    """Plot the data on 6 separate curves for S(t), E(t),I(t), R(t), D(t) & P(t) """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, axisbelow=True)
    ax.set_facecolor(facecolor)
    ax.plot(sir.t, sir.S, 'k', alpha=0.5, lw=3, label='Susceptibles')
    ax.plot(sir.t, sir.E, 'b', alpha=0.5, lw=3, label='Expuestos')
    ax.plot(sir.t, sir.I, 'r', alpha=0.5, lw=3, label='Infectados')
    ax.plot(sir.t, sir.R, 'g', alpha=0.5, lw=3, label='Recuperados')
    ax.plot(sir.t, sir.D, 'y', alpha=0.5, lw=3, label='Fatales')
    ax.plot(sir.t, sir.M, 'c', alpha=0.5, lw=3, label='Muertes')
    ax.set_xlabel('Tiempo (días)')
    ax.set_ylabel('Fracción de la población')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title(T)
    plt.show()

def plot_Is(sirs, Ls, T, figsize=(10,10), ylim=0.35, facecolor='LightGrey'):
    """Plot the Is for various configs"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, axisbelow=True)
    ax.set_facecolor(facecolor)
    for i, sir in enumerate(sirs):
        l = Ls[i]
        ax.plot(sir.t, sir.I, alpha=0.5, lw=3, label=l)
    ax.set_xlabel('Tiempo (días)')
    ax.set_ylabel('Fracción de la población')
    ax.set_ylim(0,ylim)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title(T)
    plt.show()


def plot_lognorm_pdf(mu, sigmas, figsize=(8,8)):

    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)
    x=np.linspace(0,5,200)
    for sigma in sigmas:
        y = lognorm_pdf(x, mu, sigma)
        ax.plot(x, y, lw=3, alpha=0.6, label=f' sigma = {sigma}')
    plt.title(f'lognorm, mu={mu}')
    plt.legend()
    plt.show()


def plot_hospitalisation_to_dth(zmeanHDT = 13, zmedianHDT = 9.1, figsize=(8,8)):

    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)
    x=np.linspace(0,40,200)
    y = hdt(x, zmeanHDT, zmedianHDT)
    ax.plot(x, y, lw=3, alpha=0.6)
    plt.title(f' hospitalization to dth, mean={np.argmax(y)/5:.2f}')
    plt.show()


def plot_cases_and_deaths(df, country= 'Spain', figsize=(12,12), log=False, reverse=True):
    def formatter(ax):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    def cum_sum(df, data):
        if reverse:
            Y = np.flip(df[data].values)
            CY = np.cumsum(Y)
            FCY = np.flip(CY)
        else:
            FCY = np.cumsum(df[data].values)
        return FCY

    def formats(ax, xlabel, ylabel):
        formatter(ax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if log:
            plt.yscale('log')

        plt.grid()


    fig = plt.figure(figsize=figsize)
    st = fig.suptitle(country, fontsize="x-large")

    X = df['dateRep'].values
    Y = df['cases'].values
    ax      = fig.add_subplot(2, 2, 1)
    plt.plot(X, Y, 'bo')
    formats(ax,'Date','Number of cases')


    ax      = fig.add_subplot(2, 2, 2)
    plt.plot(X, cum_sum(df, 'cases'), 'bo')
    formats(ax,'Date','Cumulative number of cases')

    Y = df['deaths']
    ax      = fig.add_subplot(2, 2, 3)
    plt.plot(X, Y, 'bo')
    formats(ax,'Date','Number of deaths')

    ax      = fig.add_subplot(2, 2, 4)
    plt.plot(X, cum_sum(df, 'deaths'), 'bo')
    formats(ax,'Date','Cumulative number of deaths')

    st.set_y(0.95)
    fig.subplots_adjust(top=0.90)

    #plt.tight_layout()
    plt.show()
    print(f' Total cases   confirmed ={df.cases.sum()}')
    print(f' Total deaths  confirmed ={df.deaths.sum()}')


def plot_data_ccaa(dfca, dataType= 'cases', thr = 2, figsize=(12,12), log=False, reverse=False):
    def formatter(ax):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    def formats(ax, xlabel, ylabel):
        formatter(ax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if log:
            plt.yscale('log')

        plt.grid()


    fig = plt.figure(figsize=figsize)
    for i, key in enumerate(misc_dict.keys()):
        cdict = misc_dict[key]
        code = cdict["geoId"]
        print(i, key, code)
        df = ecdc_select_country(dfca, country=code, thr=thr)
        ax      = fig.add_subplot(5, 4, i+1)

        X = df['dateRep'].values
        Y = df[dataType].values

        plt.plot(X, Y, 'bo', label=key)
        formats(ax,'Date', dataType)

        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_data_ccaa(dfca, dataType= 'cases', thr = 2, figsize=(12,12), log=False):
    def formatter(ax):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    def formats(ax, xlabel, ylabel):
        formatter(ax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if log:
            plt.yscale('log')

        plt.grid()


    fig = plt.figure(figsize=figsize)
    for i, key in enumerate(misc_dict.keys()):
        cdict = misc_dict[key]
        code = cdict["geoId"]
        print(i, key, code)
        df = ecdc_select_country(dfca, country=code, thr=thr)
        ax      = fig.add_subplot(5, 4, i+1)

        X = df['dateRep'].values
        Y = np.cumsum(df[dataType].values)

        plt.plot(X, Y, 'bo', label=key)
        formats(ax,'Date', dataType)

        plt.legend()
    plt.tight_layout()
    plt.show()
