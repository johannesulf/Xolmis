import os
import numpy as np
from Corrfunc.theory import DDrppi, DDsmu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
S_BINS = {'DESI-2': np.logspace(-0.4, 1.6, 11)}
MU_BINS = {'DESI-2': np.linspace(-1, +1, 21)}


def wp(sample1, rp_bins, pi_max, sample2=None, randoms=None, period=None,
       do_auto=True, do_cross=False):

    if isinstance(period, float) or isinstance(period, int):
        period = np.repeat(period, 3)

    if (do_auto and do_cross) or (not do_auto and not do_cross):
        raise RuntimeError('Not implemented!')

    elif do_auto:
        r = DDrppi(1, 2, pi_max, rp_bins, sample1[:, 0], sample1[:, 1],
                   sample1[:, 2], periodic=True, boxsize=period[0],
                   xbin_refine_factor=1, ybin_refine_factor=1,
                   zbin_refine_factor=1, copy_particles=False)
        n_exp = (len(sample1) * len(sample1) / np.prod(period) * np.pi *
                 np.diff(rp_bins**2) * 2 * pi_max)

    elif do_cross:
        r = DDrppi(0, 2, pi_max, rp_bins, sample1[:, 0], sample1[:, 1],
                   sample1[:, 2], periodic=True, boxsize=period[0],
                   X2=sample2[:, 0], Y2=sample2[:, 1], Z2=sample2[:, 2],
                   xbin_refine_factor=1, ybin_refine_factor=1,
                   zbin_refine_factor=1, copy_particles=False)
        n_exp = (len(sample1) * len(sample2) / np.prod(period) * np.pi *
                 np.diff(rp_bins**2) * 2 * pi_max)

    npairs = r['npairs']
    npairs = np.array([np.sum(n) for n in np.split(npairs, len(rp_bins) - 1)])

    return (npairs / n_exp - 1) * 2 * pi_max


def s_mu_tpcf(sample1, s_bins, mu_bins, sample2=None, randoms=None,
              period=None, do_auto=True, do_cross=False):

    if isinstance(period, float) or isinstance(period, int):
        period = np.repeat(period, 3)

    if (do_auto and do_cross) or (not do_auto and not do_cross):
        raise RuntimeError('Not implemented!')

    elif do_auto:
        r = DDsmu(1, 1, s_bins, 1, len(mu_bins) - 1, sample1[:, 0],
                  sample1[:, 1], sample1[:, 2], periodic=True,
                  boxsize=period[0], xbin_refine_factor=1,
                  ybin_refine_factor=1, zbin_refine_factor=1)
        n_exp = (len(sample1) * len(sample1) / np.prod(period) * 4 *
                 np.pi / 3 * np.diff(s_bins**3) / (len(mu_bins) - 1))

    elif do_cross:
        r = DDsmu(0, 1, s_bins, 1, len(mu_bins) - 1, sample1[:, 0],
                  sample1[:, 1], sample1[:, 2], periodic=True,
                  boxsize=period[0], X2=sample2[:, 0],
                  Y2=sample2[:, 1], Z2=sample2[:, 2], xbin_refine_factor=1,
                  ybin_refine_factor=1, zbin_refine_factor=1)
        n_exp = (len(sample1) * len(sample2) / np.prod(period) * 4 *
                 np.pi / 3 * np.diff(s_bins**3) / (len(mu_bins) - 1))

    return (r['npairs'].reshape((len(s_bins) - 1, len(mu_bins) - 1)) /
            n_exp[:, np.newaxis] - 1)
