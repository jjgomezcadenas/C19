import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


def plot_momo_oe(df, ccaa='Spain', figsize=(14,14)):
    """Plot a selected CCAA"""
    fig = plt.figure(figsize=figsize)
    st = fig.suptitle(ccaa, fontsize="x-large")

    X = df['npdate'].values

    ax = fig.add_subplot(2,2, 1, axisbelow=True)
    Y = df['defunciones_observadas'].values
    plt.plot(X, Y, 'bo')
    formats(ax,'Fechas','defs observadas',False)
    formatter(ax)

    ax = fig.add_subplot(2,2, 2, axisbelow=True)
    Y = df['defunciones_esperadas'].values
    plt.plot(X, Y, 'bo')
    formats(ax,'Fechas','defs esperadas',False)
    formatter(ax)

    ax = fig.add_subplot(2,2, 3, axisbelow=True)
    Y = df['defunciones_esperadas_q01'].values
    plt.plot(X, Y, 'bo')
    formats(ax,'Fechas','defs esperadas q01',False)
    formatter(ax)

    ax = fig.add_subplot(2, 2, 4, axisbelow=True)
    Y = df['defunciones_esperadas_q99'].values
    plt.plot(X, Y, 'bo')
    formats(ax,'Fechas','defs esperadas q99',False)
    formatter(ax)

    plt.tight_layout()
    plt.show()


def plot_momo(df, ydata='defunciones_observadas', ccaa='Spain', figsize=(14,14)):
    """Plot a selected CCAA"""
    fig = plt.figure(figsize=figsize)
    st = fig.suptitle(ccaa, fontsize="x-large")


    X = df['npdate'].values
    Y = df[ydata].values
    ax      = fig.add_subplot(111)
    plt.plot(X, Y, 'bo')
    formats(ax,'Fechas','defs',False)
    formatter(ax)
    plt.show()


def plot_momo_ccaa(dfs, xdata = 'npdate', ydata='defunciones_observadas', figsize=(14,14)):
    """Plot all CCAAS"""
    fig = plt.figure(figsize=figsize)

    for i, df in enumerate(dfs) :
        ax = fig.add_subplot(6,3, i+1, axisbelow=True)
        X = df[xdata].values
        Y = df[ydata].values

        plt.plot(X, Y, 'bo')
        formats(ax,'Fechas','defs','False')
        plt.title(ccaa_name[i])

    plt.tight_layout()
    plt.show


def plot_ccaa(dfs, xdata='date', ydata='cdead', figsize=(14,14)):
    """Plot all CCAAS"""
    fig = plt.figure(figsize=figsize)

    i=0
    for ccaa_name, df in dfs.items() :
        if ccaa_name == 'Ceuta':
            continue

        ax = fig.add_subplot(6,3, i+1, axisbelow=True)
        X = df[xdata].values
        Y = df[ydata].values

        plt.plot(X, Y, 'bo')
        formats(ax,'Fechas','defs observadas',False)
        formatter(ax)
        plt.title(ccaa_name)
        i+=1

    plt.tight_layout()
    plt.show()

def plot_ccaa_isc3_c19(dfc3, dfc19, figsize=(14,14)):
    """Plot all CCAAS"""
    fig = plt.figure(figsize=figsize)

    i=0
    for ccaa_name, df in dfc3.items() :
        #print(ccaa_name)

        if ccaa_name == 'Ceuta':
            continue

        # if ccaa_name == 'Andalucia':
        #     #print(df)
        #     df2 = dfc19[ccaa_name]
        #     #print(df2)
        #     X    = df['date'].values
        #     XC19    = df2['dateRep'].values
        #     YC3  = df['cdead'].values
        #     YC19 = df2['deaths'].values
        #     print(len(X), len(XC19), len(YC3), len(YC19))
        #     return 0


        df2 = dfc19[ccaa_name]
        ax = fig.add_subplot(6,3, i+1, axisbelow=True)
        X    = df['date'].values
        XC19    = df2['dateRep'].values
        YC3  = df['cdead'].values
        YC19 = df2['deaths'].values
        #print(len(X), len(XC19), len(YC3), len(YC19))

        plt.plot(X, YC3, 'bo')
        plt.plot(X, YC19, 'ro')
        formats(ax,'Fechas','defs observadas',False)
        formatter(ax)
        plt.title(ccaa_name)
        i+=1

    plt.tight_layout()
    plt.show()

def plot_momo_XY(tD, tS, Y, tCase='tD', yCase='Yobs', figsize=(14,14)):

    fig = plt.figure(figsize=figsize)
    st  = fig.suptitle(yCase, fontsize="x-large")
    ax  = fig.add_subplot(111)

    if tCase=='tD':
        X = tD
        formatter(ax)
    else:
        X = tS

    plt.plot(X, Y, 'bo')
    formats(ax,'Fechas','defs',logscale =False)
    #plt.tight_layout()
    plt.show()


def plot_momo_XYS(tD, tS, dY, tCase='tD', yCase='observed', figsize=(14,14)):

    fig = plt.figure(figsize=figsize)
    #st  = fig.suptitle(yCase, fontsize="x-large")
    i = 0
    for t, Y in dY.items() :
        if t == 'Ceuta':
            continue

        ax = fig.add_subplot(6,3, i+1, axisbelow=True)

        if tCase=='tD':
            X = tD
            formatter(ax)
        else:
            X = tS

        plt.plot(X, Y, 'bo')
        formats(ax,'Fechas', yCase,logscale =False)
        plt.title(t)
        i+=1

    plt.tight_layout()
    plt.show()


def plot_momo_c19_XY(tD, tS, dmomo, dc19, tCase='tD', figsize=(20,20)):

    fig = plt.figure(figsize=figsize)
    #st  = fig.suptitle(yCase, fontsize="x-large")
    i = 0
    for t, Y in dmomo.items() :
        if t == 'Ceuta' or t == 'Date':
            continue
        ax = fig.add_subplot(6,3, i+1, axisbelow=True)

        if tCase=='tD':
            X = tD
            formatter(ax)
        else:
            X = tS

        plt.plot(X, Y, 'o')
        plt.plot(X, dc19[t], 'o')
        formats(ax,'Fechas', 'deceased',logscale =False)
        plt.title(t)
        i+=1

    plt.tight_layout()
    plt.show()


def plot_comomo_XY(dY, edY, figsize=(20,20)):

    fig = plt.figure(figsize=figsize)
    X  = dY['Date']
    i = 0
    for t, Y in dY.items() :
        if t == 'Ceuta' or t == 'Date':
            continue

        ax = fig.add_subplot(6,3, i+1, axisbelow=True)
        formatter(ax)

        eY = edY[t]
        plt.errorbar(X, Y, yerr = eY,
                     ls = ':', marker = 'o', ms = 4, label = 'comomo')
        formats(ax,'Fechas', 'comomo',logscale =False)
        plt.title(t)
        i+=1

    plt.tight_layout()
    plt.show()


def plot_comomo(df, ccaa='Madrid', figsize=(14,14)):
    """Plot a selected CCAA"""
    Y  = df['values'][ccaa].values
    eY = df['errors'][ccaa].values
    X  = df['values']['Date'].values

    fig = plt.figure(figsize=figsize)

    ax      = fig.add_subplot(111)
    plt.errorbar(X, Y, yerr = eY, ls = ':', marker = 'o', ms = 4, label = 'comomo')
    formats(ax,'Fechas', 'comomo',logscale =False)
    formatter(ax)
    plt.title(ccaa)
    plt.show()
