import os
import copy
import xolmis
import argparse
import numpy as np
import multiprocessing
from tabcorr import TabCorr
from halotools.empirical_models import TrivialPhaseSpace, NFWPhaseSpace
from halotools.mock_observables import s_mu_tpcf, tpcf_multipole


def tabcorr_s_mu_to_multipole(halotab_s_mu, mu_bins, order):
    halotab_mult = copy.deepcopy(halotab_s_mu)
    halotab_mult.tpcf_shape = (halotab_s_mu.tpcf_shape[0], )
    halotab_mult.tpcf_matrix = np.zeros(
        (halotab_s_mu.tpcf_shape[0], halotab_s_mu.tpcf_matrix.shape[1]))

    for i in range(halotab_s_mu.tpcf_matrix.shape[1]):
        halotab_mult.tpcf_matrix[:, i] = tpcf_multipole(
            halotab_s_mu.tpcf_matrix[:, i].reshape(
                halotab_s_mu.tpcf_shape), mu_bins, order=order)

    return halotab_mult


def main():

    parser = argparse.ArgumentParser(
        description='Tabulate AbacusSummit halo catalogs.')
    parser.add_argument('simulation', help='simulation')
    parser.add_argument('redshift', help='simulation redshift', type=float)
    parser.add_argument('config', help='which configuration to assume')

    args = parser.parse_args()

    path = xolmis.simulation_directory(args.simulation, args.redshift)

    halocat = xolmis.read_simulation(args.simulation, args.redshift)
    for key in halocat.halo_table.colnames:
        if key[:6] == 'halo_m':
            mdef = key[6:]
    cens_prof_model = TrivialPhaseSpace(redshift=halocat.redshift, mdef=mdef)
    sats_prof_model = NFWPhaseSpace(redshift=halocat.redshift, mdef=mdef,
                                    cosmology=halocat.cosmology)

    halotab_s_mu = TabCorr.tabulate(
        halocat, s_mu_tpcf, xolmis.S_BINS[args.config],
        xolmis.MU_BINS[args.config], cens_prof_model=cens_prof_model,
        sats_prof_model=sats_prof_model, verbose=True,
        num_threads=multiprocessing.cpu_count(),
        sats_per_prim_haloprop=2e-13, project_xyz=True,
        prim_haloprop_bins=100, prim_haloprop_key='halo_mvir',
        sec_haloprop_key='halo_nfw_conc', sec_haloprop_percentile_bins=0.5,
        cosmology_obs=xolmis.COSMOLOGY_OBS[args.config])

    for order in [0, 2, 4]:
        halotab_multipole = tabcorr_s_mu_to_multipole(
            halotab_s_mu, xolmis.MU_BINS[args.config], order)
        halotab_multipole.write(os.path.join(path, 'xi_{}.hdf5'.format(order)),
                                overwrite=True)


if __name__ == "__main__":
    main()
