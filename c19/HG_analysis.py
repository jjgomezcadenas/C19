import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def recovery_time(hgi):
    tR = []
    for ti, group in hgi.groupby(['ingreso']).alta:
        #print(f'ti={ti}')
        #print(f'group index ={group.index} group values {group.values}')
        tas = group.values
        for ta in tas:
            nt = ti.to_datetime64()
            dt = ta - nt
            dtd = pd.to_datetime(dt).day
            tR.append(dtd)
    return tR


def hosp_uci(hgi):
    nH = []
    nU = []
    tD = []
    for ti, group in hgi.groupby(['ingreso']).hosp:
        tD.append(ti)
        #print(f'ti={ti}')
        #print(f'group index ={group.index} group values {group.values}')
        nh = 0
        nu = 0
        for g in group:
            if g == 'HOS':
                nh += 1
            elif g == 'UCI':
                nu += 1
        nH.append(nh)
        nU.append(nu)
    return tD, nH, nU


def exitus_group(dfp):
    df    = dfp.loc[dfp.code == "EXITUS", ('pacienteID', 'ingreso', 'alta', 'edad', 'sexo')]
    return df


def exitus_and_sex_group(dfp, sexo):
    df = exitus_group(dfp)
    dfs   = df.loc[df.sexo == sexo, ('pacienteID', 'ingreso', 'alta', 'edad')]
    return dfs


def exitus_time(dfp, sexo='Hombre'):
    if sexo == "Hombre" or sexo == "Mujer":
        dfs = exitus_and_sex_group(dfp, sexo)
    else:
        dfs = exitus_group(dfp)

    tR = []
    for ti, group in dfs.groupby(['ingreso']).alta:
        #print(f'ti={ti}')
        #print(f'group index ={group.index} group values {group.values}')
        tas = group.values
        for ta in tas:
            nt = ti.to_datetime64()
            dt = ta - nt
            dtd = pd.to_datetime(dt).day
            tR.append(dtd)
    return tR


# def exitus_age(dfp, sexo='Hombre',figsize=(14,14)):
#     df    = dfp.loc[dfp.code == "EXITUS", ('pacienteID', 'ingreso', 'alta', 'edad', 'sexo')]
#     dfs   = df.loc[df.sexo == sexo, ('pacienteID', 'ingreso', 'alta', 'edad')]
#     np = dfs.groupby(['ingreso']).edad.count()
#
#     fig = plt.figure(figsize=figsize)
#
#     X = np.index
#     Y = np.values
#     ax      = fig.add_subplot(111)
#     plt.plot(X, Y, 'bo')
#     formats(ax,'ingresos','casos',False)
#     formatter(ax)
#     plt.show()


def exitus(dfp, sexo='Hombre', groupby='edad', datestamp=True):
    if sexo == "Hombre" or sexo == "Mujer":
        dfs = exitus_and_sex_group(dfp, sexo)   s
    else:
        dfs = exitus_group(dfp)

    nps =dfs.groupby([groupby]).pacienteID.count()

    print(f' exitus total = {len(dfs)} fraction = {len(dfs)/len(dfp)}')

    if datestamp:
        return nps
    else:
        return pd.Series(data=nps.values, index= range(len(nps.index)))


def select_age_by_CA_and_sex(dfp, sexo='Hombre',CA = 'TOTAL ESPAÑA'):
    df    = dfp.loc[dfp.sexo == sexo, ('ca', 'edad', 'poblacion')]
    dfs   = df.loc[df.ca == CA, ('edad', 'poblacion')]

    AGE=[]
    P = []
    for v in dfs.values[1:]:
        #print(v)
        a = v[0].split('-')
        #print(a)
        if a[0] == '100 y más':
            ai = 100
            au = 110
        else:
            ai = int(a[0])
            au = int(a[1])
        AGE.append((ai,au))
        p = v[1].replace('.','')
        #print(p)
        P.append(float(p))

    AB = [(a[0] + a[1]) / 2 for a in AGE]
    return pd.DataFrame({"arange":AGE, "amean":AB, "poblacion":P})



def formatter(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

def formats(ax, xlabel, ylabel, logscale=False):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logscale:
        plt.yscale('log')


def plot_hgi(nps, figsize=(14,14)):

    fig = plt.figure(figsize=figsize)

    X = nps.index
    Y = nps.values
    ax      = fig.add_subplot(111)
    plt.plot(X, Y, 'bo')
    formats(ax,'ingresos','casos',False)
    formatter(ax)
    plt.show()


def plot_nh_nu(tD, nH, nU, figsize=(14,14)):

    fig = plt.figure(figsize=figsize)

    ax      = fig.add_subplot(2,1,1)
    plt.plot(tD, nH, 'bo')
    formats(ax,'ingresos','hospital',False)
    formatter(ax)

    ax      = fig.add_subplot(2,1,2)
    plt.plot(tD, nU, 'bo')
    formats(ax,'ingresos','uci',False)
    formatter(ax)

    plt.show()



def plot_XY_series(ss,xlabel='ingresos',ylabel='casos',figsize=(14,14)):
    fig = plt.figure(figsize=figsize)
    X = ss.index
    Y = ss.values
    print(type(X[0]))
    ax      = fig.add_subplot(111)
    plt.plot(X, Y, 'bo')
    formats(ax,xlabel,ylabel,False)
    formatter(ax)
    plt.show()



def bin_shift(bins):
    BINS = []
    for i in range(len(bins) -1):
        b = (bins[i] + bins[i+1]) / 2
        BINS.append(b)
    return BINS


def hist_XY_series(ss,xlabel='ingresos',ylabel='casos', bins=10, figsize=(14,14)):
    fig = plt.figure(figsize=figsize)
    bins = np.linspace(ss.index.min(), ss.index.max(), bins)
    groups = ss.groupby(pd.cut(ss.index, bins))

    vals = groups.mean().values
    ax      = fig.add_subplot(111)
    xbins = bin_shift(bins)
    print(xbins)
    print(vals)

    tot = np.sum(vals)
    plt.plot(xbins, vals/tot, 'bo')
    ax.vlines(xbins, 0, vals/tot, colors='b', lw=5, alpha=0.5)

    formats(ax,xlabel,ylabel,False)

    plt.show()


def plot_pop_age(df, figsize=(14,14)):
    fig = plt.figure(figsize=figsize)
    X = df.amean
    Y = df.poblacion
    tot = np.sum(Y)
    ax      = fig.add_subplot(111)
    plt.plot(X, Y/tot, 'bo')
    ax.vlines(X, 0, Y/tot, colors='b', lw=5, alpha=0.5)
    formats(ax,"average age","fraction of population",False)
    plt.show()
