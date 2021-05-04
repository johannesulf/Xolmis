import os
import numpy as np
import argparse
from tabcorr import TabCorr
import matplotlib.pyplot as plt
from halotools.empirical_models import PrebuiltHodModelFactory
from tabulate import output_directory


def main():

    parser = argparse.ArgumentParser(
        description='Plot the results of the tabulation.')
    parser.add_argument('simulation', help='simulation')
    parser.add_argument('redshift', help='simulation redshift', type=float)

    args = parser.parse_args()

    path = output_directory(args.simulation, args.redshift)

    halotab = TabCorr.read(os.path.join(path, 'wp.hdf5'))

    model = PrebuiltHodModelFactory(
        'zheng07', threshold=-18, redshift=args.redshift)
    rp_ave = 0.5 * (halotab.tpcf_args[0][1:] + halotab.tpcf_args[0][:-1])

    ngal, wp = halotab.predict(model)
    plt.plot(rp_ave, wp, label='total')

    ngal, wp = halotab.predict(model, separate_gal_type=True)
    for key in wp.keys():
        plt.plot(rp_ave, wp[key], label=key, ls='--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$r_p \ [h^{-1} \ \mathrm{Mpc}]$')
    plt.ylabel(r'$w_p \ [h^{-1} \ \mathrm{Mpc}]$')
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(path, 'wp_decomposition.png'), dpi=300)
    plt.close()

    print(model.param_dict)

    np.savetxt(os.path.join(path, 'wp.csv'),
               np.array([wp['centrals-centrals'], wp['centrals-satellites'],
                         wp['satellites-satellites']]).T, delimiter=',')


if __name__ == "__main__":
    main()
