import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import w0waCDM, Planck15
from Corrfunc.theory import DDrppi, DDsmu
from halotools.sim_manager import UserSuppliedHaloCatalog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
S_BINS = {'DESI-2': np.logspace(-1.0, 1.8, 15)}
MU_BINS = {'DESI-2': np.linspace(0, 1, 21)}
COSMOLOGY_OBS = {'DESI-2': Planck15}


def wp_corrfunc(sample1, rp_bins, pi_max, sample2=None, randoms=None,
                period=None, do_auto=True, do_cross=False):

    if (do_auto and do_cross) or (not do_auto and not do_cross):
        raise RuntimeError('Not implemented!')

    elif do_auto:
        r = DDrppi(1, 2, pi_max, rp_bins, sample1[:, 0], sample1[:, 1],
                   sample1[:, 2], periodic=True, boxsize=period,
                   xbin_refine_factor=1, ybin_refine_factor=1,
                   zbin_refine_factor=1, copy_particles=False)
        n_exp = (len(sample1) * len(sample1) / np.prod(period) * np.pi *
                 np.diff(rp_bins**2) * 2 * pi_max)

    elif do_cross:
        r = DDrppi(0, 2, pi_max, rp_bins, sample1[:, 0], sample1[:, 1],
                   sample1[:, 2], periodic=True, boxsize=period,
                   X2=sample2[:, 0], Y2=sample2[:, 1], Z2=sample2[:, 2],
                   xbin_refine_factor=1, ybin_refine_factor=1,
                   zbin_refine_factor=1, copy_particles=False)
        n_exp = (len(sample1) * len(sample2) / np.prod(period) * np.pi *
                 np.diff(rp_bins**2) * 2 * pi_max)

    npairs = r['npairs']
    npairs = np.array([np.sum(n) for n in np.split(npairs, len(rp_bins) - 1)])

    return (npairs / n_exp - 1) * 2 * pi_max


def s_mu_tpcf_corrfunc(sample1, s_bins, mu_bins, sample2=None, randoms=None,
                       period=None, do_auto=True, do_cross=False):

    if (do_auto and do_cross) or (not do_auto and not do_cross):
        raise RuntimeError('Not implemented!')

    elif do_auto:
        r = DDsmu(1, 1, s_bins, 1, len(mu_bins) - 1, sample1[:, 0],
                  sample1[:, 1], sample1[:, 2], periodic=True,
                  boxsize=period, xbin_refine_factor=1,
                  ybin_refine_factor=1, zbin_refine_factor=1)
        n_exp = (len(sample1) * len(sample1) / np.prod(period) * 4 *
                 np.pi / 3 * np.diff(s_bins**3) / (len(mu_bins) - 1))

    elif do_cross:
        r = DDsmu(0, 1, s_bins, 1, len(mu_bins) - 1, sample1[:, 0],
                  sample1[:, 1], sample1[:, 2], periodic=True,
                  boxsize=period, X2=sample2[:, 0],
                  Y2=sample2[:, 1], Z2=sample2[:, 2], xbin_refine_factor=1,
                  ybin_refine_factor=1, zbin_refine_factor=1)
        n_exp = (len(sample1) * len(sample2) / np.prod(period) * 4 *
                 np.pi / 3 * np.diff(s_bins**3) / (len(mu_bins) - 1))

    return (r['npairs'].reshape((len(s_bins) - 1, len(mu_bins) - 1)) /
            n_exp[:, np.newaxis] - 1)


def simulation_directory(simulation, redshift):
    return os.path.join(BASE_DIR, 'sims', simulation, '{:.1f}'.format(
        redshift).replace('.', 'p'))


def read_simulation(simulation, redshift):
    halos = Table.read(os.path.join(simulation_directory(
        simulation, redshift), 'sim.hdf5'), path='halos')
    mdef = '{:.0f}m'.format(halos.meta['SODensityL1'])
    cosmology = w0waCDM(H0=halos.meta['H0'], Om0=halos.meta['Omega_M'],
                        Ode0=halos.meta['Omega_DE'],
                        w0=halos.meta['w0'], wa=halos.meta['wa'])

    return UserSuppliedHaloCatalog(
        redshift=redshift, Lbox=halos.meta['BoxSize'],
        particle_mass=halos.meta['ParticleMassHMsun'], simname=simulation,
        halo_x=halos['halo_x'], halo_y=halos['halo_y'], halo_z=halos['halo_z'],
        halo_vx=halos['halo_vx'], halo_vy=halos['halo_vy'],
        halo_vz=halos['halo_vz'], halo_id=np.arange(len(halos)),
        halo_pid=np.repeat(-1, len(halos)),
        halo_upid=np.repeat(-1, len(halos)),
        halo_nfw_conc=halos['halo_r{}'.format(mdef)] / halos['halo_rs'],
        halo_mvir=halos['halo_m{}'.format(mdef)],
        halo_rvir=halos['halo_r{}'.format(mdef)] * 1e-9,
        halo_hostid=np.arange(len(halos)), cosmology=cosmology,
        **{'halo_m{}'.format(mdef): halos['halo_m{}'.format(mdef)],
           'halo_r{}'.format(mdef): halos['halo_r{}'.format(mdef)]})
