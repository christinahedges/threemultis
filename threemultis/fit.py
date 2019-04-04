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

from .utils import _plot_light_curve

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



def PLD(tpf, planet_mask=None, aperture=None, return_soln=False, return_quick_corrected=False, sigma=5, trim=0, ndraws=1000):
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
        flux = np.asarray(tpf.flux[:, trim:-trim, trim:-trim], np.float64)
        flux_err = np.asarray(tpf.flux_err[:, trim:-trim, trim:-trim], np.float64)
        aper = np.asarray(aperture, bool)[trim:-trim, trim:-trim]
    else:
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

    if (~np.isfinite(X_pld)).any():
        raise ValueError('NaNs in components.')
    if (np.any(raw_flux_err == 0)):
        raise ValueError('Zeros in raw_flux_err.')

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
            logs2 = pm.Normal("logs2", mu=np.log(np.var(raw_flux[mask])), sd=4)
#            pm.Potential("logs2_prior1", tt.switch(logs2 < -3, -np.inf, 0.0))

            logsigma = pm.Normal("logsigma", mu=np.log(np.std(raw_flux[mask])), sd=4)
            logrho = pm.Normal("logrho", mu=np.log(150), sd=4)
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
            map_soln = xo.optimize(start=start, vars=[logsigma])
            map_soln = xo.optimize(start=start, vars=[logrho, logsigma])
            map_soln = xo.optimize(start=start, vars=[logsigma])
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
        burnin = sampler.tune(tune=np.max([int(ndraws*0.3), 150]), start=map_soln,
                              step_kwargs=dict(target_accept=0.9),
                              chains=4)
    # Sample
    with model:
        trace = sampler.sample(draws=ndraws, chains=4)

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


def fit_planets(lc, period_value, t0_value, depth_value, R_star, M_star, T_star, texp=0.0204335, ndraws=1000):
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
            logP = pm.Normal("logP", mu=np.log(period_value), sd=0.01, shape=shape)
            t0 = pm.Normal("t0", mu=t0_value, sd=0.01, shape=shape)
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
        burnin = sampler.tune(tune=np.max([int(ndraws*0.3), 150]), start=map_soln,
                              step_kwargs=dict(target_accept=0.9),
                              chains=4)
    # Sample
    with model:
        trace = sampler.sample(draws=ndraws, chains=4)

    return trace, mask
