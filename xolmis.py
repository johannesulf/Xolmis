import os
import numpy as np
from tabcorr import TabCorr
from astropy.table import Table
from astropy.cosmology import w0waCDM, Planck15
from Corrfunc.theory import DDrppi, DDsmu
from halotools.sim_manager import UserSuppliedHaloCatalog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
S_BINS = {'DESI-2': np.logspace(-1.0, 1.8, 15),
          'default': np.logspace(-1.0, 1.8, 15)}
MU_BINS = {'DESI-2': np.linspace(0, 1, 21)}
COSMOLOGY_OBS = {'DESI-2': Planck15}
NGAL = [5e-4, 1e-3, 2e-3]


def read_mock_observations(ngal):
    fname = '{}'.format(ngal).replace('.', 'p') + '.hdf5'
    table = Table.read(os.path.join(BASE_DIR, 'mocks', fname))
    xi = np.concatenate([table['xi0'], table['xi2'], table['xi4']])
    n_jk = table['xi0_jk'].shape[1]

    xi = np.concatenate([table['xi0'], table['xi2'], table['xi4']])
    xi_cov = np.cov(np.vstack(
        [table['xi0_jk'], table['xi2_jk'], table['xi4_jk']])) * n_jk
    return xi, xi_cov
