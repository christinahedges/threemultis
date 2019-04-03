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


def joint_fit(tpf, period_value, t0_value, depth_value, duration_value, R_star,
                M_star, T_star, aperture=None, texp=0.0204335,
                return_quick_corrected=False, return_soln=False, trim=0):
    shape = len(period_value)

    planet_mask = np.ones(len(tpf.time), bool)
    for p, t, d in zip(period_value, t0_value, duration_value):
        planet_mask &= np.abs((tpf.time - t + 0.5*p) % p - 0.5*p) > d/2


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
            orbit=orbit, r=r_pl, t=time[mask], texp=texp)*1e3
            light_curve = pm.math.sum(light_curves, axis=-1) + mean
            pm.Deterministic("light_curves", light_curves)


            # GP
            # --------
            logs2 = pm.Normal("logs2", mu=np.log(1e-4*np.var(raw_flux[mask])), sd=10)
            logsigma = pm.Normal("logsigma", mu=np.log(np.std(raw_flux[mask])), sd=10)
            logrho = pm.Normal("logrho", mu=np.log(150), sd=10)
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
