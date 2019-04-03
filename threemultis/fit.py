import os
from contextlib import contextmanager
import warnings
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
import lightkurve as lk
import corner

from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
import astropy.units as u
import pandas as pd

log = logging.getLogger()
log.setLevel('WARNING')
plt.style.use(lk.MPLSTYLE)

@contextmanager
def silence():
    '''Suppreses all output'''
    logger = logging.getLogger()
    logger.disabled = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


def _plot_light_curve(map_soln, model, mask, x, y, yerr, components, gp):
    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    motion = np.dot(components, map_soln['weights']).reshape(-1)
    with model:
        stellar = xo.eval_in_model(gp.predict(x), map_soln)

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

    return fig


def joint_fit(tpf, period_value, t0_value, depth_value, duration_value, R_star, M_star, T_star, aperture=None, texp=0.0204335, return_quick_corrected=False, return_soln=False):
    shape = len(period_value)

    planet_mask = np.ones(len(tpf.time), bool)
    for p, t, d in zip(period_value, t0_value, duration_value):
        planet_mask &= np.abs((tpf.time - t + 0.5*p) % p - 0.5*p) > d/2


    if aperture is None:
        aperture = tpf.pipeline_mask


    time = np.asarray(tpf.time, np.float64)
    flux = np.asarray(tpf.flux, np.float64)
    flux_err = np.asarray(tpf.flux_err, np.float64)

    aper = np.asarray(aperture, bool)
    raw_flux = np.asarray(np.nansum(flux[:, aper], axis=(1)),  np.float64)
    raw_flux_err = np.asarray(np.nansum(flux_err[:, aper]**2, axis=(1))**0.5,  np.float64)

    raw_flux_err /= np.median(raw_flux)
    raw_flux /= np.median(raw_flux)
    raw_flux -= 1

    # Setting to Parts Per Thousand keeps us from hitting machine precision errors...
    raw_flux *= 1e3
    raw_flux_err *= 1e3

    # Build the first order PLD basis
#    X_pld = np.reshape(flux[:, aper], (len(flux), -1))
    saturation = (np.nanpercentile(flux, 95, axis=0) > 175000)
    X_pld = np.reshape(flux[:, aper & ~saturation], (len(tpf.flux), -1))

    extra_pld = np.zeros((len(time), np.any(saturation, axis=0).sum()))
    idx = 0
    for column in saturation.T:
        if column.any():
            extra_pld[:, idx] = np.sum(flux[:, column, :], axis=(1, 2))
            idx += 1
    X_pld = np.hstack([X_pld, extra_pld])

    # Remove NaN pixels
    X_pld = X_pld[:, ~((~np.isfinite(X_pld)).all(axis=0))]
    X_pld = X_pld / np.sum(flux[:, aper], axis=-1)[:, None]

    # Build the second order PLD basis and run PCA to reduce the number of dimensions
    X2_pld = np.reshape(X_pld[:, None, :] * X_pld[:, :, None], (len(flux), -1))
    # Remove NaN pixels
    X2_pld = X2_pld[:, ~((~np.isfinite(X2_pld)).all(axis=0))]
    U, _, _ = np.linalg.svd(X2_pld, full_matrices=False)
    X2_pld = U[:, :X_pld.shape[1]]

    ## Construct the design matrix and fit for the PLD model
    X_pld = np.concatenate((X_pld, X2_pld), axis=-1)

    def build_model(mask=None, start=None):
        ''' Build a PYMC3 model

        Parameters
        ----------
        mask : np.ndarray
            Boolean array to mask cadences. Cadences that are False will be excluded
            from the model fit
        start : dict
            MAP Solution from exoplanet

        Returns
        -------
        model : pymc3.model.Model
            A pymc3 model
        map_soln : dict
            Best fit solution
        '''

        if mask is None:
            mask = np.ones(len(time), dtype=bool)

        with pm.Model() as model:

            # Parameters for the stellar properties
            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            u_star = xo.distributions.QuadLimbDark("u_star")

            m_star = pm.Normal("m_star", mu=M_star[0], sd=M_star[1])
            r_star = pm.Normal("r_star", mu=R_star[0], sd=R_star[1])
            t_star = pm.Normal("t_star", mu=T_star[0], sd=T_star[1])

            # Prior to require physical parameters
            pm.Potential("m_star_prior", tt.switch(m_star > 0, 0, -np.inf))
            pm.Potential("r_star_prior", tt.switch(r_star > 0, 0, -np.inf))

            # Orbital parameters for the planets
            logP = pm.Normal("logP", mu=np.log(period_value), sd=1, shape=shape)
            t0 = pm.Normal("t0", mu=t0_value, sd=1, shape=shape)
            b = pm.Uniform("b", lower=0, upper=1, testval=0.5, shape=shape)
            logr = pm.Normal("logr", sd=1.0,
                        mu=0.5*np.log(np.array(depth_value))+np.log(R_star[0]), shape=shape)
            r_pl = pm.Deterministic("r_pl", tt.exp(logr))
            ror = pm.Deterministic("ror", r_pl / r_star)

            # Tracking planet parameters
            period = pm.Deterministic("period", tt.exp(logP))

            # Orbit model
            orbit = xo.orbits.KeplerianOrbit(
            r_star=r_star, m_star=m_star,
            period=period, t0=t0, b=b)

            incl = pm.Deterministic('incl', orbit.incl)
            a = pm.Deterministic('a', orbit.a)
            teff = pm.Deterministic('teff', t_star * tt.sqrt(0.5*(1/a)))

            # Compute the model light curve using starry
            light_curves = xo.StarryLightCurve(u_star).get_light_curve(
            orbit=orbit, r=r_pl, t=time[mask], texp=texp)*1e3
            light_curve = pm.math.sum(light_curves, axis=-1) + mean
            pm.Deterministic("light_curves", light_curves)


            # GP
            # --------
            logs2 = pm.Normal("logs2", mu=-2, sd=10)
            logsigma = pm.Normal("logsigma", mu=np.log(np.std(raw_flux[mask])), sd=10)
            logrho = pm.Normal("logrho", mu=np.log(30), sd=10)
            kernel = xo.gp.terms.Matern32Term(log_rho=logrho, log_sigma=logsigma)
            gp = xo.gp.GP(kernel, time[mask], tt.exp(logs2) + raw_flux_err[mask]**2)

            # Motion model
            #------------------
            A = tt.dot(X_pld[mask].T, gp.apply_inverse(X_pld[mask]))
            B = tt.dot(X_pld[mask].T, gp.apply_inverse(raw_flux[mask, None]))
            C = tt.slinalg.solve(A, B)
            motion_model = pm.Deterministic("motion_model", tt.dot(X_pld[mask], C)[:, 0])

            # Likelihood
            #------------------
            pm.Potential("obs", gp.log_likelihood(raw_flux[mask] - motion_model))

            # gp predicted flux
            gp_pred = gp.predict()
            pm.Deterministic("gp_pred", gp_pred)
            pm.Deterministic("weights", C)


            # Optimize
            #------------------
            if start is None:
                start = model.test_point

            map_soln = xo.optimize(start=start, vars=[logrho, logsigma])
            map_soln = xo.optimize(start=start, vars=[logr])
            map_soln = xo.optimize(start=map_soln, vars=[logs2])
            map_soln = xo.optimize(start=map_soln, vars=[logrho, logsigma, logs2, logr])
            map_soln = xo.optimize(start=map_soln, vars=[mean, logr])
            map_soln = xo.optimize(start=map_soln, vars=[logP, t0])
            map_soln = xo.optimize(start=map_soln, vars=[b])
            map_soln = xo.optimize(start=map_soln, vars=[u_star])
            map_soln = xo.optimize(start=map_soln, vars=[logrho, logsigma, logs2])
            map_soln = xo.optimize(start=map_soln)

            return model, map_soln, gp

    with silence():
        model0, map_soln0, gp = build_model()


    # Remove outliers
    with model0:
        motion = np.dot(X_pld, map_soln0['weights']).reshape(-1)
        stellar = xo.eval_in_model(gp.predict(time), map_soln0)
        corrected = raw_flux - motion - stellar
        mask = ~sigma_clip(corrected, sigma=5).mask
        mask = ~(convolve(mask, Box1DKernel(5), fill_value=1) != 1)
        mask |= (~planet_mask)

    with silence():
        model, map_soln, gp = build_model(start=map_soln0, mask=mask)

    lc_fig = _plot_light_curve(map_soln, model, mask, time, raw_flux, raw_flux_err, X_pld, gp)

    if return_soln:
        motion = np.dot(X_pld, map_soln['weights']).reshape(-1)
        with model:
            stellar = xo.eval_in_model(gp.predict(time), map_soln)
        return model, map_soln, motion, stellar

    if return_quick_corrected:
        raw_lc = tpf.to_lightcurve()
        clc = lk.KeplerLightCurve(time=time,
                                  flux=(raw_flux - stellar - motion) * 1e-3 + 1,
                                  flux_err=(raw_flux_err) * 1e-3,
                                  time_format=raw_lc.time_format,
                                  centroid_col=tpf.estimate_centroids()[0],
                                  centroid_row=tpf.estimate_centroids()[0], quality=raw_lc.quality, channel=raw_lc.channel,
                                  campaign=raw_lc.campaign, quarter=raw_lc.quarter, mission=raw_lc.mission, cadenceno=raw_lc.cadenceno, targetid=raw_lc.targetid,
                                  ra=raw_lc.ra, dec=raw_lc.dec, label='{} PLD Corrected'.format(raw_lc.targetid))
        return clc

    return model, map_soln, gp, X_pld, time, raw_flux, raw_flux_err, mask





def PLD(tpf, planet_mask=None, aperture=None, return_soln=False, return_quick_corrected=False, sigma=5, trim=0):
    ''' Use exoplanet, pymc3 and theano to perform PLD correction

    Parameters
    ----------
    tpf : lk.TargetPixelFile
        Target Pixel File to Correct
    planet_mask : np.ndarray
        Boolean array. Cadences where planet_mask is False will be excluded from the
        PLD correction. Use this to mask out planet transits.
    '''
    if planet_mask is None:
        planet_mask = np.ones(len(tpf.time), dtype=bool)
    if aperture is None:
        aperture = tpf.pipeline_mask

    time = np.asarray(tpf.time, np.float64)
    if trim > 0:
        flux = np.nan_to_num(np.asarray(tpf.flux[:, trim:-trim, trim:-trim], np.float64))
        flux_err = np.nan_to_num(np.asarray(tpf.flux_err[:, trim:-trim, trim:-trim], np.float64))
        aper = np.asarray(aperture, bool)[trim:-trim, trim:-trim]
    else:
        flux = np.nan_to_num(np.asarray(tpf.flux, np.float64))
        flux_err = np.nan_to_num(np.asarray(tpf.flux_err, np.float64))
        aper = np.asarray(aperture, bool)

    raw_flux = np.asarray(np.nansum(flux[:, aper], axis=(1)),  np.float64)
    raw_flux_err = np.asarray(np.nansum(flux_err[:, aper]**2, axis=(1))**0.5,  np.float64)

    raw_flux_err /= np.median(raw_flux)
    raw_flux /= np.median(raw_flux)
    raw_flux -= 1

    # Setting to Parts Per Thousand keeps us from hitting machine precision errors...
    raw_flux *= 1e3
    raw_flux_err *= 1e3

    # Build the first order PLD basis
#    X_pld = np.reshape(flux[:, aper], (len(flux), -1))
    saturation = (np.nanpercentile(flux, 100, axis=0) > 175000)
    X_pld = np.reshape(flux[:, aper & ~saturation], (len(tpf.flux), -1))

    extra_pld = np.zeros((len(time), np.any(saturation, axis=0).sum()))
    idx = 0
    for column in saturation.T:
        if column.any():
            extra_pld[:, idx] = np.sum(flux[:, column, :], axis=(1, 2))
            idx += 1
    X_pld = np.hstack([X_pld, extra_pld])

    # Remove NaN pixels
    X_pld = X_pld[:, ~((~np.isfinite(X_pld)).all(axis=0))]
    X_pld = X_pld / np.sum(flux[:, aper], axis=-1)[:, None]

    # Build the second order PLD basis and run PCA to reduce the number of dimensions
    X2_pld = np.reshape(X_pld[:, None, :] * X_pld[:, :, None], (len(flux), -1))
    # Remove NaN pixels
    X2_pld = X2_pld[:, ~((~np.isfinite(X2_pld)).all(axis=0))]
    U, _, _ = np.linalg.svd(X2_pld, full_matrices=False)
    X2_pld = U[:, :X_pld.shape[1]]

    ## Construct the design matrix and fit for the PLD model
    X_pld = np.concatenate((X_pld, X2_pld), axis=-1)


    def build_model(mask=None, start=None):
        ''' Build a PYMC3 model

        Parameters
        ----------
        mask : np.ndarray
            Boolean array to mask cadences. Cadences that are False will be excluded
            from the model fit
        start : dict
            MAP Solution from exoplanet

        Returns
        -------
        model : pymc3.model.Model
            A pymc3 model
        map_soln : dict
            Best fit solution
        '''

        if mask is None:
            mask = np.ones(len(time), dtype=bool)

        with pm.Model() as model:
            # GP
            # --------
            logs2 = pm.Normal("logs2", mu=-2, sd=10)
            logsigma = pm.Normal("logsigma", mu=np.log(np.std(raw_flux[mask])), sd=10)
            logrho = pm.Normal("logrho", mu=np.log(30), sd=10)
            kernel = xo.gp.terms.Matern32Term(log_rho=logrho, log_sigma=logsigma)
            gp = xo.gp.GP(kernel, time[mask], tt.exp(logs2) + raw_flux_err[mask]**2)

            # Motion model
            #------------------
            A = tt.dot(X_pld[mask].T, gp.apply_inverse(X_pld[mask]))
            B = tt.dot(X_pld[mask].T, gp.apply_inverse(raw_flux[mask, None]))
            C = tt.slinalg.solve(A, B)
            motion_model = pm.Deterministic("motion_model", tt.dot(X_pld[mask], C)[:, 0])

            # Likelihood
            #------------------
            pm.Potential("obs", gp.log_likelihood(raw_flux[mask] - motion_model))

            # gp predicted flux
            gp_pred = gp.predict()
            pm.Deterministic("gp_pred", gp_pred)
            pm.Deterministic("weights", C)

            # Optimize
            #------------------
            if start is None:
                start = model.test_point
            map_soln = xo.optimize(start=start, vars=[logrho, logsigma])
            map_soln = xo.optimize(start=map_soln, vars=[logs2])
            map_soln = xo.optimize(start=map_soln, vars=[logrho, logsigma, logs2])
            return model, map_soln, gp

    # First rough correction
    log.info('Optimizing roughly')
    with silence():
        model0, map_soln0, gp = build_model(mask=planet_mask)

    # Remove outliers, make sure to remove a few nearby points incase of flares.
    with model0:
        motion = np.dot(X_pld, map_soln0['weights']).reshape(-1)
        stellar = xo.eval_in_model(gp.predict(time), map_soln0)
        corrected = raw_flux - motion - stellar
        mask = ~sigma_clip(corrected, sigma=sigma).mask
        mask = ~(convolve(mask, Box1DKernel(3), fill_value=1) != 1)
        mask &= planet_mask

    # Optimize PLD
    log.info('Optimizing without outliers')
    with silence():
        model, map_soln, gp = build_model(mask, map_soln0)
    lc_fig = _plot_light_curve(map_soln, model, mask, time, raw_flux, raw_flux_err, X_pld, gp)

    if return_soln:
        motion = np.dot(X_pld, map_soln['weights']).reshape(-1)
        with model:
            stellar = xo.eval_in_model(gp.predict(time), map_soln)
        return model, map_soln, motion, stellar

    if return_quick_corrected:
        raw_lc = tpf.to_lightcurve()
        clc = lk.KeplerLightCurve(time=time,
                                  flux=(raw_flux - stellar - motion) * 1e-3 + 1,
                                  flux_err=(raw_flux_err) * 1e-3,
                                  time_format=raw_lc.time_format,
                                  centroid_col=tpf.estimate_centroids()[0],
                                  centroid_row=tpf.estimate_centroids()[0], quality=raw_lc.quality, channel=raw_lc.channel,
                                  campaign=raw_lc.campaign, quarter=raw_lc.quarter, mission=raw_lc.mission, cadenceno=raw_lc.cadenceno, targetid=raw_lc.targetid,
                                  ra=raw_lc.ra, dec=raw_lc.dec, label='{} PLD Corrected'.format(raw_lc.targetid))
        return clc

    # Burn in
    sampler = xo.PyMC3Sampler()
    with model:
        burnin = sampler.tune(tune=400, start=map_soln,
                              step_kwargs=dict(target_accept=0.9),
                              chains=4)
    # Sample
    with model:
        trace = sampler.sample(draws=1000, chains=4)

    varnames = ["logrho", "logsigma", "logs2"]
    pm.traceplot(trace, varnames=varnames);


    samples = pm.trace_to_dataframe(trace, varnames=varnames)
    corner.corner(samples);


    # Generate 50 realizations of the prediction sampling randomly from the chain
    N_pred = 50
    pred_mu = np.empty((N_pred, len(time)))
    pred_motion = np.empty((N_pred, len(time)))
    with model:
        pred = gp.predict(time)
        for i, sample in enumerate(tqdm(xo.get_samples_from_trace(trace, size=N_pred), total=N_pred)):
            pred_mu[i] = xo.eval_in_model(pred, sample)
            pred_motion[i, :] = np.dot(X_pld, sample['weights']).reshape(-1)

    star_model = np.mean(pred_mu + pred_motion, axis=0)
    star_model_err = np.std(pred_mu + pred_motion, axis=0)

    raw_lc = tpf.to_lightcurve()

    meta = {'samples':samples, 'trace':trace, 'pred_mu':pred_mu, 'pred_motion':pred_motion}

    clc = lk.KeplerLightCurve(time=time,
                              flux=(raw_flux - star_model) * 1e-3 + 1,
                              flux_err=((raw_flux_err**2 + star_model_err**2)**0.5) * 1e-3,
                              time_format=raw_lc.time_format,
                              centroid_col=tpf.estimate_centroids()[0],
                              centroid_row=tpf.estimate_centroids()[0], quality=raw_lc.quality, channel=raw_lc.channel,
                              campaign=raw_lc.campaign, quarter=raw_lc.quarter, mission=raw_lc.mission, cadenceno=raw_lc.cadenceno, targetid=raw_lc.targetid,
                              ra=raw_lc.ra, dec=raw_lc.dec, label='{} PLD Corrected'.format(raw_lc.targetid), meta=meta)
    return clc


def fit_planets(lc, period_value, t0_value, depth_value, R_star, M_star, T_star, texp=0.0204335):
    '''Fit planet parameters using exoplanet'''

    lc = lc.copy()
    shape = len(period_value)
#    if (np.asarray([len(period_value), len(t0_value), len(depth_value)]) == shape).all() == False
#        raise ValueError('All planet parameters must have the same shape!')


    x, y, yerr = np.asarray(lc.time, float), np.asarray(lc.flux, float), np.asarray(lc.flux_err, float)

    yerr /= np.median(y)
    y /= np.median(y)
    y -= 1

    y *= 1e3
    yerr *= 1e3


    def build_model(mask=None, start=None):
        if mask is None:
            mask = np.ones(len(x), dtype=bool)
        with pm.Model() as model:

            # Parameters for the stellar properties
            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            u_star = xo.distributions.QuadLimbDark("u_star")

            m_star = pm.Normal("m_star", mu=M_star[0], sd=M_star[1])
            r_star = pm.Normal("r_star", mu=R_star[0], sd=R_star[1])
            t_star = pm.Normal("t_star", mu=T_star[0], sd=T_star[1])


            # Prior to require physical parameters
            pm.Potential("m_star_prior", tt.switch(m_star > 0, 0, -np.inf))
            pm.Potential("r_star_prior", tt.switch(r_star > 0, 0, -np.inf))

            # Orbital parameters for the planets
            logP = pm.Normal("logP", mu=np.log(period_value), sd=1, shape=shape)
            t0 = pm.Normal("t0", mu=t0_value, sd=1, shape=shape)
            b = pm.Uniform("b", lower=0, upper=1, testval=0.5, shape=shape)
            logr = pm.Normal("logr", sd=1.0,
                             mu=0.5*np.log(np.array(depth_value))+np.log(R_star[0]), shape=shape)
            r_pl = pm.Deterministic("r_pl", tt.exp(logr))
            ror = pm.Deterministic("ror", r_pl / r_star)

            # Tracking planet parameters
            period = pm.Deterministic("period", tt.exp(logP))

            # Orbit model
            orbit = xo.orbits.KeplerianOrbit(
                r_star=r_star, m_star=m_star,
                period=period, t0=t0, b=b)

            incl = pm.Deterministic('incl', orbit.incl)
            a = pm.Deterministic('a', orbit.a)

            teff = pm.Deterministic('teff', t_star * tt.sqrt(0.5*(1/a)))


            # Compute the model light curve using starry
            light_curves = xo.StarryLightCurve(u_star).get_light_curve(
                orbit=orbit, r=r_pl, t=x[mask], texp=texp)*1e3
            light_curve = pm.math.sum(light_curves, axis=-1) + mean
            pm.Deterministic("light_curves", light_curves)

            pm.Normal('obs', mu=light_curve, sd=yerr[mask], observed=y[mask])

            # Optimize
            #------------------
            if start is None:
                start = model.test_point
            map_soln = xo.optimize(start=start, vars=[logr])
            map_soln = xo.optimize(start=map_soln, vars=[b])
            map_soln = xo.optimize(start=map_soln, vars=[logP, t0])
            map_soln = xo.optimize(start=map_soln, vars=[u_star])
            map_soln = xo.optimize(start=map_soln, vars=[logr])
            map_soln = xo.optimize(start=map_soln, vars=[b])
            map_soln = xo.optimize(start=map_soln, vars=[mean])
            map_soln = xo.optimize(start=map_soln)
            return model, map_soln

    with silence():
        model0, map_soln0 = build_model()

    corrected = y.copy()
    for planet in map_soln0['light_curves'].T:
        corrected -= planet
    mask = ~sigma_clip(corrected, sigma_upper=4, sigma_lower=10).mask
    mask = ~(convolve(mask, Box1DKernel(2), fill_value=1) != 1)

    with silence():
        model, map_soln = build_model(mask=mask, start=map_soln0)

    # Burn in
    sampler = xo.PyMC3Sampler()
    with model:
        burnin = sampler.tune(tune=150, start=map_soln,
                              step_kwargs=dict(target_accept=0.9),
                              chains=4)
    # Sample
    with model:
        trace = sampler.sample(draws=400, chains=4)

    return trace, mask


def plot_folded_transits(lc, trace, mask, name):
    lc = lc.copy()
    x, y, yerr = np.asarray(lc.time, float), np.asarray(lc.flux, float), np.asarray(lc.flux_err, float)

    yerr /= np.median(y)
    y /= np.median(y)
    y -= 1

    y *= 1e3
    yerr *= 1e3

    nplanets = trace['light_curves'].shape[-1]
    fig, axs = plt.subplots(1, nplanets, figsize=(7 * nplanets, 5))

    for idx, letter in enumerate("bc"):

        ax = axs[idx]
        # Get the posterior median orbital parameters
        p = np.median(trace["period"][:, idx])
        t0 = np.median(trace["t0"][:, idx])

        # Compute the median of posterior estimate of the contribution from
        # the other planet. Then we can remove this from the data to plot
        # just the planet we care about.
        other = np.median(trace["light_curves"][:, :, (idx + 1) % 2], axis=0)

        # Plot the folded data
        x_fold = (x[mask] - t0 + 0.5*p) % p - 0.5*p
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

        ax.legend(fontsize=10, loc=4)
        ax.set_xlim(-0.5*p, 0.5*p)
        ax.set_xlabel("Time from Transit Midpoint [days]")
        ax.set_ylabel("Relative Flux [ppt]")
        ax.set_title("{0} {1}".format(name, letter));
        ax.set_xlim(-0.3, 0.3)
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
