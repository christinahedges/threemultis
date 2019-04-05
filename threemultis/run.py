from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os
import pymc3 as pm
from . import PACKAGEDIR

from astropy.io import fits
from astropy.modeling import models, fitting
import astropy.units as u

import lightkurve as lk
import corner

from . import utils
from . import fit


varnames = varnames=["r_pl", "b", "t0", "logP", "r_star", "m_star", "u_star", "mean"]


def _run(clc, name):
    params = utils.get_params(name)
    trace, mask = fit.fit_planets(clc, period_value=list(params['Period']),
                                                t0_value=list(params['T0']), depth_value=list(params['Depth']),
                                                R_star=(params.loc[0, 'R_star'], params.loc[0, 'R_star_error']),
                                                M_star=(params.loc[0, 'M_star'], params.loc[0, 'M_star_error']),
                                                T_star=(params.loc[0, 'T_star'], params.loc[0, 'T_star_error']), ndraws=4000)

    samples = pm.trace_to_dataframe(trace, varnames=varnames)

    # Convert the radius to Earth radii
    samples["r_pl__0"] = (np.array(samples["r_pl__0"]) * u.R_sun).to(u.R_earth).value
    samples["r_pl__1"] = (np.array(samples["r_pl__1"]) * u.R_sun).to(u.R_earth).value


    fig = corner.corner(samples);
    fig.savefig('figures/{}_corner.png'.format(PACKAGEDIR, name), dpi=150, bbox_inches='tight')

    fig = utils.plot_folded_transits(clc, trace, mask, name);
    fig.savefig('{}/figures/{}.png'.format(PACKAGEDIR,name), dpi=150, bbox_inches='tight')

    str = utils.latex_trace(trace, name)
    with open("{}/results/{}.dat".format(PACKAGEDIR,name), "w") as file:
        file.write(str)


def threemultis():
    # K2-198
    # ---------------------------------------------#
    print('K2-198')
    tpfs = lk.search_targetpixelfile('K2-198').download_all()

    clcs = []     # Corrected Light Curves

    for tpf in tpfs:
        tpf = tpf[10:]
        tpf = tpf[np.in1d(tpf.time, tpf.to_lightcurve(aperture_mask='all').remove_nans().time)]
        tpf = tpf[tpf.to_lightcurve().normalize().flux > 0.8]
        aper = tpf.create_threshold_mask()
        tpf.plot(aperture_mask=aper)


        mask = utils.planet_mask(tpf.time, 'K2-198')
        clc = fit.PLD(tpf, planet_mask=mask, trim=1, ndraws=1000, logrho_mu=np.log10(150),
                                  aperture=aper)
        clcs.append(clc)

    clc = clcs[0].append(clcs[1])
    clc.to_fits('{}/results/K2-198.fits'.format(PACKAGEDIR))
    clc.to_csv('{}/results/K2-198.csv'.format(PACKAGEDIR))

    _run(clc, 'K2-198')

    # K2-168
    # ---------------------------------------------#
    print('K2-168')

    tpf = lk.search_targetpixelfile('K2-168').download()
    tpf = tpf[10:]
    tpf = tpf[np.in1d(tpf.time, tpf.to_lightcurve(aperture_mask='all').remove_nans().time)]
    tpf = tpf[tpf.to_lightcurve().normalize().flux > 0.8]


    mask = utils.planet_mask(tpf.time, 'K2-168')
    aper = np.nanmedian(tpf.flux, axis=0) > 30
    # First pass, remove some very bad outliers
    bad = np.zeros(len(tpf.time), bool)
    for count in range(2):
        pld_lc = tpf[~bad].to_corrector('pld').correct(aperture_mask=aper, cadence_mask=mask[~bad])
        pld_lc = pld_lc.flatten(31, mask=~mask[~bad])
        bad |= np.in1d(tpf.time, pld_lc.time[np.abs(pld_lc.flux - 1) > 5 * np.std(pld_lc.flux - 1)])

    tpf = tpf[~bad]
    mask = mask[~bad]
    clc = fit.PLD(tpf, planet_mask=mask, trim=0, aperture=aper, logrho_mu=np.log(1))
    clc.to_fits('{}results/K2-168.fits'.format(PACKAGEDIR))
    clc.to_csv('{}results/K2-168.csv'.format(PACKAGEDIR))

    _run(clc, 'K2-168')

    # K2-43
    # ---------------------------------------------#
    print('K2-43')

    # Trim out some pixels which have a bleed column on them
    raw_tpf = lk.search_targetpixelfile('K2-43').download()
    hdu = deepcopy(raw_tpf.hdu)
    for name in hdu[1].columns.names:
        if (len(hdu[1].data[name].shape) == 3):
            hdu[1].data[name][:, :, :4] = np.nan
    fits.HDUList(hdus=list(hdu)).writeto('hack.fits', overwrite=True)
    tpf = lk.KeplerTargetPixelFile('hack.fits', quality_bitmask=raw_tpf.quality_bitmask)
    os.remove('hack.fits')

    tpf = tpf[10:]
    tpf = tpf[np.in1d(tpf.time, tpf.to_lightcurve(aperture_mask='all').remove_nans().time)]
    tpf = tpf[tpf.to_lightcurve().normalize().flux > 0.8]


    mask = utils.planet_mask(tpf.time, 'K2-43')
    aper = np.nan_to_num(np.nanpercentile(tpf.flux, 95, axis=(0))) > 50

    # First pass, remove some very bad outliers
    bad = np.zeros(len(tpf.time), bool)
    for count in range(2):
        pld_lc = tpf[~bad].to_corrector('pld').correct(aperture_mask=aper, cadence_mask=mask[~bad])
        pld_lc = pld_lc.flatten(31, mask=~mask[~bad])
        bad |= np.in1d(tpf.time, pld_lc.time[np.abs(pld_lc.flux - 1) > 5 * np.std(pld_lc.flux - 1)])

    tpf = tpf[~bad]
    mask = mask[~bad]
    clc = fit.PLD(tpf, planet_mask=mask, trim=1, aperture=aper,
                              logrho_mu=np.log(30))
    clc.to_fits('{}results/K2-43.fits'.format(PACKAGEDIR))
    clc.to_csv('{}results/K2-43.csv'.format(PACKAGEDIR))

    _run(clc, 'K2-43')
