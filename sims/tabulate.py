import os
import argparse
import numpy as np
from scipy.interpolate import interp1d

from astropy.cosmology import w0waCDM
from astropy import units as u
from astropy import constants

from colossus.cosmology.cosmology import setCosmology
from colossus.halo import profile_nfw

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.empirical_models import delta_vir
from halotools.empirical_models import TrivialPhaseSpace, NFWPhaseSpace
from Corrfunc.theory import DDrppi, DDsmu

from tabcorr import TabCorr


def wp_corrfunc(sample1, rp_bins, pi_max, sample2=None, randoms=None,
                period=None, do_auto=True, do_cross=True):

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
                       period=None, do_auto=True, do_cross=True):

    print(np.amin(sample1, axis=0), np.amax(sample1, axis=0))
    if do_cross:
        print(np.amin(sample2, axis=0), np.amax(sample2, axis=0))

    if (do_auto and do_cross) or (not do_auto and not do_cross):
        raise RuntimeError('Not implemented!')

    elif do_auto:
        r = DDsmu(1, 1, s_bins, 1, len(mu_bins) - 1, sample1[:, 0],
                  sample1[:, 1], sample1[:, 2], periodic=True,
                  boxsize=period[0], xbin_refine_factor=1,
                  ybin_refine_factor=1, zbin_refine_factor=1, verbose=True)
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


def read_abacus_summit_catalog(simulation, redshift):

    as_path = os.path.join('/', 'global', 'cfs', 'cdirs', 'desi',
                           'cosmosim', 'Abacus', 'AbacusSummit_')

    fields = ['id', 'x_com', 'v_com', 'N', 'SO_radius', 'vcirc_max_com']
    halocat = CompaSOHaloCatalog(os.path.join(
        as_path + simulation, 'halos', 'z{:.3f}'.format(redshift)),
        fields=fields)

    mdef = '{:.0f}m'.format(halocat.header['SODensityL1'])
    cosmology = w0waCDM(H0=halocat.header['H0'], Om0=halocat.header['Omega_M'],
                        Ode0=halocat.header['Omega_DE'],
                        w0=halocat.header['w0'], wa=halocat.header['wa'])

    halocat.halos['x_com'] += halocat.header['BoxSize'] / 2.0
    halocat.halos['SO_mass'] = (halocat.halos['N'] *
                                halocat.header['ParticleMassHMsun'])
    halocat.halos = halocat.halos[halocat.halos['N'] > 300]

    dvir = delta_vir(cosmology, redshift) * 200 / (18 * np.pi**2)
    rho_crit = (cosmology.critical_density(redshift) /
                (cosmology.H(0).value / 100)**2 / (1 + redshift)**3)
    halocat.halos['SO_radius'] = ((halocat.halos['SO_mass'] * u.M_sun / (
        4.0 / 3.0 * np.pi * rho_crit * dvir))**(1.0 / 3.0)).to(u.Mpc).value

    # Convert Vmax/Vvir to concentration. The cosmology we set here does not
    # affect the result. But colossus needs to have a cosmology chosen.
    setCosmology('planck15')

    c = np.linspace(2.163, 20, 1000)
    r = np.zeros_like(c)

    for i in range(len(c)):
        p = profile_nfw.NFWProfile(M=1E12, c=c[i], z=redshift, mdef=mdef)
        r[i] = p.Vmax()[0] / p.circularVelocity(p.RDelta(redshift, mdef))

    c = interp1d(r, c, fill_value=(c[0], c[-1]), bounds_error=False)

    vvir = np.sqrt(constants.G * halocat.halos['SO_mass'] * u.Msun / (
            halocat.halos['SO_radius'] * u.Mpc / (1 + redshift))).to(
                u.km / u.s).value
    c = c(halocat.halos['vcirc_max_com'] / vvir)

    return UserSuppliedHaloCatalog(
        redshift=redshift, Lbox=halocat.header['BoxSize'],
        particle_mass=halocat.header['ParticleMassHMsun'],
        simname=simulation,
        halo_x=halocat.halos['x_com'][:, 0],
        halo_y=halocat.halos['x_com'][:, 1],
        halo_z=halocat.halos['x_com'][:, 2],
        halo_vx=halocat.halos['v_com'][:, 0],
        halo_vy=halocat.halos['v_com'][:, 1],
        halo_vz=halocat.halos['v_com'][:, 2],
        halo_id=halocat.halos['id'],
        halo_pid=np.repeat(-1, len(halocat.halos)),
        halo_upid=np.repeat(-1, len(halocat.halos)), halo_nfw_conc=c,
        halo_mvir=halocat.halos['SO_mass'],
        halo_rvir=halocat.halos['SO_radius'] * 1e-9,
        halo_hostid=halocat.halos['id'], cosmology=cosmology,
        **{'halo_m{}'.format(mdef): halocat.halos['SO_mass'],
           'halo_r{}'.format(mdef): halocat.halos['SO_radius']}), mdef


def output_directory(simulation, redshift):
    return os.path.join(
        simulation, '{:.1f}'.format(redshift).replace('.', 'p'))


def main():

    parser = argparse.ArgumentParser(
        description='Tabulate AbacusSummit halo catalogs.')
    parser.add_argument('simulation', help='simulation')
    parser.add_argument('redshift', help='simulation redshift', type=float)

    args = parser.parse_args()

    path = output_directory(args.simulation, args.redshift)
    if not os.path.isdir(path):
        os.makedirs(path)

    halocat, mdef = read_abacus_summit_catalog(args.simulation, args.redshift)

    rp_bins = np.logspace(-1.5, 1.477, 26)
    cens_prof_model = TrivialPhaseSpace(redshift=halocat.redshift, mdef=mdef)
    sats_prof_model = NFWPhaseSpace(redshift=halocat.redshift, mdef=mdef,
                                    cosmology=halocat.cosmology)

    halotab = TabCorr.tabulate(
        halocat, wp_corrfunc, rp_bins, pi_max=40,
        cens_prof_model=cens_prof_model, sats_prof_model=sats_prof_model,
        verbose=True, num_threads=34, sats_per_prim_haloprop=1e-12,
        project_xyz=True)

    halotab.write(os.path.join(path, 'wp.hdf5'), overwrite=True)


if __name__ == "__main__":
    main()
