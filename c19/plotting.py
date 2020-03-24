import numpy as np
import matplotlib.pyplot as plt


def plot_sir(sir, T, figsize=(10,10), facecolor='LightGrey'):
    """Plot the data on three separate curves for S(t), I(t) and R(t)"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, axisbelow=True)
    ax.set_facecolor(facecolor)
    ax.plot(sir.t, sir.S/sir.N, 'b', alpha=0.5, lw=3, label='Susceptibles')
    ax.plot(sir.t, sir.I/sir.N, 'r', alpha=0.5, lw=3, label='Infectados')
    ax.plot(sir.t, sir.R/sir.N, 'g', alpha=0.5, lw=3, label='Recuperados')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Fraction of population')
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


def plot_seir(sir, T, figsize=(10,10), facecolor='LightGrey'):
    """Plot the data on three separate curves for S(t),E(t), I(t) and R(t)"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, axisbelow=True)
    ax.set_facecolor(facecolor)
    ax.plot(sir.t, sir.S, 'k', alpha=0.5, lw=3, label='Susceptibles')
    ax.plot(sir.t, sir.E, 'b', alpha=0.5, lw=3, label='Expuestos')
    ax.plot(sir.t, sir.I, 'r', alpha=0.5, lw=3, label='Infectados')
    ax.plot(sir.t, sir.R, 'g', alpha=0.5, lw=3, label='Recuperados')
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
