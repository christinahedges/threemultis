import pandas as pd
from . import PACKAGEDIR
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

plt.style.use(lk.MPLSTYLE)

def get_params(name):
    df = pd.read_csv('{}/data/{}.csv'.format(PACKAGEDIR, name))
    return df

def planet_mask(time, name):
    params = get_params(name)
    mask = np.ones(len(time), bool)
    for planet, df in params.iterrows():
        p = df['Period']
        t0 = df['T0']
        d = df['Duration']
        x_fold = (time - t0 + 0.5*p) % p - 0.5*p
        mask &= (np.abs(x_fold) > d/2)
    return mask

def planet_plot(clc, name):
    params = get_params(name)
    fig, axs = plt.subplots(len(params), 1, figsize=(8, 3 * len(params)), sharex=True)
    for planet, df in params.iterrows():
        otherplanets = list(set(list(np.arange(len(params)))) - set([planet]))
        mask = np.ones(len(clc.time), bool)
        for op in otherplanets:
            p1 = params.loc[op, 'Period']
            t01 = params.loc[op, 'T0']
            d1 = params.loc[op, 'Duration']
            x_fold = (clc.time - t01 + 0.5*p1) % p1 - 0.5*p1
            mask &= (np.abs(x_fold) > d1/2)

        p = df['Period']
        t0 = df['T0']
        d = df['Duration']
        x_fold = (clc.time - t0 + 0.5*p) % p - 0.5*p
        clc[mask].fold(p, t0).errorbar(ax=axs[planet], label='Planet {} (Period: {:2.4f}d)'.format(planet + 1, p))
        if planet < len(params) - 1:
            axs[planet].set_xlabel('')
        axs[planet].set_xlim(-0.2, 0.2)
        if planet == 0:
            axs[planet].set_title(name)
    return fig
