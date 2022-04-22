import os
import xolmis
import argparse
import numpy as np
from astropy import units as u
from astropy.cosmology import w0waCDM
from halotools.empirical_models import delta_vir
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog


def read_abacus_summit_catalog(simulation, redshift):
    as_path = os.path.join('/', 'global', 'cfs', 'cdirs', 'desi',
                           'cosmosim', 'Abacus', 'AbacusSummit_')

    fields = ['x_L2com', 'v_L2com', 'N', 'rvcirc_max_com']
    halocat = CompaSOHaloCatalog(os.path.join(
        as_path + simulation, 'halos', 'z{:.3f}'.format(redshift)),
        fields=fields)
    halocat.halos = halocat.halos[halocat.halos['N'] > 300]
    halos = halocat.halos

    mdef = '{:.0f}m'.format(halocat.header['SODensityL1'])
    halos['halo_m{}'.format(mdef)] = (
        halos['N'] * halocat.header['ParticleMassHMsun'])
    halos.remove_column('N')

    halos['x_L2com'] += halocat.header['BoxSize'] / 2.0
    halos['halo_x'] = halos['x_L2com'][:, 0]
    halos['halo_y'] = halos['x_L2com'][:, 1]
    halos['halo_z'] = halos['x_L2com'][:, 2]
    halos.remove_column('x_L2com')

    halos['halo_vx'] = halos['v_L2com'][:, 0]
    halos['halo_vy'] = halos['v_L2com'][:, 1]
    halos['halo_vz'] = halos['v_L2com'][:, 2]
    halos.remove_column('v_L2com')

    cosmology = w0waCDM(H0=halocat.header['H0'], Om0=halocat.header['Omega_M'],
                        Ode0=halocat.header['Omega_DE'],
                        w0=halocat.header['w0'], wa=halocat.header['wa'])
    dvir = delta_vir(cosmology, redshift) * 200 / (18 * np.pi**2)
    rho_crit = (cosmology.critical_density(redshift) /
                (cosmology.H(0).value / 100)**2 / (1 + redshift)**3)
    halocat.halos['halo_r{}'.format(mdef)] = ((
        halocat.halos['halo_m{}'.format(mdef)] * u.M_sun / (
            4.0 / 3.0 * np.pi * rho_crit * dvir))**(1.0 / 3.0)).to(u.Mpc).value

    halos['rvcirc_max_com'] /= 2.16258
    halos.rename_column('rvcirc_max_com', 'halo_rs')

    return halos


def main():

    parser = argparse.ArgumentParser(
        description='Convert a AbacusSummit simulation into a small file.')
    parser.add_argument('simulation', help='simulation')
    parser.add_argument('redshift', help='simulation redshift', type=float)

    args = parser.parse_args()

    path = xolmis.simulation_directory(args.simulation, args.redshift)
    if not os.path.isdir(path):
        os.makedirs(path)

    halos = read_abacus_summit_catalog(args.simulation, args.redshift)
    halos.write(os.path.join(path, 'sim.hdf5'), path='halos', overwrite=True)


if __name__ == "__main__":
    main()
