import pandas as pd
from . import PACKAGEDIR
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import exoplanet as xo
import astropy.units as u


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

def planet_plot(clc, name, nbin=1):
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
        f = clc[mask].fold(p, t0).bin(nbin)
        f.errorbar(ax=axs[planet], label='Planet {} (Period: {:2.4f}d)'.format(planet + 1, p))

        if planet < len(params) - 1:
            axs[planet].set_xlabel('')
        axs[planet].set_xlim(-0.1, 0.1)
        if planet == 0:
            axs[planet].set_title(name)
    return fig



def plot_folded_transits(lc, trace, mask, name):
    params = get_params(name)
    lc = lc.copy()
    x, y, yerr = np.asarray(lc.time, float), np.asarray(lc.flux, float), np.asarray(lc.flux_err, float)

    yerr /= np.median(y)
    y /= np.median(y)
    y -= 1

    y *= 1e3
    yerr *= 1e3

    nplanets = trace['light_curves'].shape[-1]
    fig, axs = plt.subplots(nplanets, 1, figsize=(7, 4.5 * nplanets))

    letters = np.asarray(['b','c', 'd', 'e'])
    for idx, letter in enumerate(letters[:nplanets]):

        ax = axs[idx]
        # Get the posterior median orbital parameters
        p = np.median(trace["period"][:, idx])
        t0 = np.median(trace["t0"][:, idx])

        # Compute the median of posterior estimate of the contribution from
        # the other planet. Then we can remove this from the data to plot
        # just the planet we care about.
        otherplanets = list(set(list(np.arange(len(params)))) - set([idx]))
        other = np.zeros(mask.sum())
        for o in otherplanets:
            other += np.median(trace["light_curves"][:, :, o], axis=0)

        # Plot the folded data
        x_fold = ((x[mask] - t0 + 0.5*p) % p - 0.5*p)/p
        ax.errorbar(x_fold, y[mask] - other, yerr=yerr[mask], color="k", ls='', label="Data",
                     zorder=-1000, lw=0.2)

        # Plot the folded model
        inds = np.argsort(x_fold)
        inds = inds[np.abs(x_fold)[inds] < 0.3]
        pred = trace["light_curves"][:, inds, idx] + trace["mean"][:, None]
        pred = np.percentile(pred, [16, 50, 84], axis=0)
        ax.plot(x_fold[inds], pred[1], color="C1", label="Transit Model")
        art = ax.fill_between(x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5,
                               zorder=1000)
        art.set_edgecolor("none")

        # Annotate the plot with the planet's period
        txt = "period = {0:.4f} +/- {1:.4f} d".format(
            np.mean(trace["period"][:, idx]), np.std(trace["period"][:, idx]))
        ax.annotate(txt, (0, 0), xycoords="axes fraction",
                     xytext=(5, 5), textcoords="offset points",
                     ha="left", va="bottom", fontsize=12)

        ax.legend(fontsize=10, loc='lower right')
        ax.set_xlim(-0.5*p, 0.5*p)
        if idx == nplanets - 1:
            ax.set_xlabel("Phase")
        ax.set_ylabel("Relative Flux [ppt]")
        ax.set_title("{0} {1}".format(name, letter));
        ax.set_xlim(-0.05, 0.05)
    return fig

def latex_trace(trace, name):
    ''' Prints nice latex table of results
    '''

    nplanets = trace['light_curves'].shape[-1]

    letters = 'bcdefghejklmnop'
    results = pd.DataFrame(columns=['\\emph{{{} {}}}'.format(name, letters[idx]) for idx in range(nplanets)])
    labels = ['Period [days]', 'Transit Midpoint [JD]', 'Radius [$R_{earth}$]', 'Impact Parameter', 'Inclination [degrees]', 'Semi Major Axis [a/R$^*]$', "Equillibrium Temperature [K]"]
    keys = ['period', 't0', 'r_pl', 'b', 'incl', 'a', 'teff']
    for label, key in zip(labels, keys):
        for idx in range(nplanets):
            ans = np.percentile(trace[key][:, idx], [16, 50, 84], axis=0)
            if key == 't0':
                ans += 2454833
            if key == 'r_pl':
               ans *= (u.R_sun).to(u.R_earth)
            if key == 'incl':
                ans = (180 * u.deg - ((ans * u.radian).to(u.deg))).value

            ans -= np.asarray([0, ans[0], ans[0]])
            ac = 2 - int(np.log10(np.nanmin(np.abs(ans))))
            str = '{{{}}} $\pm _{{{{{{{}}}}}}} ^{{{{{{{}}}}}}}$'.format(*[':2.{}f'.format(ac) for i in range(3)])
            results.loc[label, results.columns[idx]] = str.format(*ans)

    star_results = pd.DataFrame(columns=['Host Star'])
    labels = ['Mass [Msol]', 'Radius [Rsol]', 'Effective Temperature [K]']
    keys = ['m_star', 'r_star', 't_star']
    for label, key in zip(labels, keys):
        ans = np.percentile(trace[key], [16, 50, 84], axis=0)

        ans -= np.asarray([0, ans[0], ans[0]])
        ac = 2 - int(np.log10(np.nanmin(np.abs(ans))))
        str = '{{{}}} $\pm _{{{{{{{}}}}}}} ^{{{{{{{}}}}}}}$'.format(*[':2.{}f'.format(ac) for i in range(3)])
        star_results.loc[label, star_results.columns[0]] = str.format(*ans)

    labels = ['Limb Darkening 1', 'Limb Darkening 2']
    for idx, label in enumerate(labels):
        ans = np.percentile(trace['u_star'][:, idx], [16, 50, 84], axis=0)

        ans -= np.asarray([0, ans[0], ans[0]])
        ac = 2 - int(np.log10(np.nanmin(np.abs(ans))))
        str = '{{{}}} $\pm _{{{{{{{}}}}}}} ^{{{{{{{}}}}}}}$'.format(*[':2.{}f'.format(ac) for i in range(3)])
        star_results.loc[label, star_results.columns[0]] = str.format(*ans)

    str = results.to_latex(escape=False) + '\n' + star_results.to_latex(escape=False)
    return str


def _plot_light_curve(map_soln, model, mask, x, y, yerr, components, gp):
    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    motion = np.dot(components, map_soln['weights']).reshape(-1)
    with model:
        stellar = xo.eval_in_model(gp.predict(x), map_soln)

    if 'light_curves' in map_soln.keys():
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    corrected = y - motion - stellar
    ax = axes[0]
    ax.plot(x, y, "k", label="Raw Data")
    ax.plot(x, motion, color="C1", label="Motion Model")
    ax.legend(fontsize=10)
    ax.set_ylabel("Relative Flux [ppt]", fontsize=8)

    ax = axes[1]
    ax.plot(x, y - motion, "k", label="Motion Corrected Data")
    ax.plot(x, stellar, color="C2", label="Stellar Variability", lw=3, alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_ylabel("Relative Flux [ppt]", fontsize=8)

    ax = axes[2]
    ax.plot(x, y - motion - stellar, "k", label="Fully Corrected Data")
    if 'light_curves' in map_soln.keys():
        for p in range(map_soln['light_curves'].shape[1]):
            ax.plot(x[mask], map_soln['light_curves'][:, p], "k", label="Planet {}".format(p), c='C{}'.format(p + 1))
        ax.set_ylim(0 + np.nanmin(map_soln['light_curves']), 0 - np.nanmin(map_soln['light_curves']))
    ax.legend(fontsize=10)
    ax.set_ylabel("Relative Flux [ppt]", fontsize=8)

    if 'light_curves' in map_soln.keys():
        ax = axes[3]
        ax.plot(x, y - motion - stellar - np.sum(map_soln["light_curves"], axis=-1), "k", label="Residuals")
        ax.set_ylim(0 + np.nanmin(map_soln['light_curves']), 0 - np.nanmin(map_soln['light_curves']))
        ax.legend(fontsize=10)
        ax.set_ylabel("Relative Flux [ppt]", fontsize=8)
    return fig
