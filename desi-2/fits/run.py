import os
import xolmis
import numpy as np
import pyswarms as ps
from astropy.table import Table
from scipy.optimize import minimize
from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import AssembiasZheng07Cens
from halotools.empirical_models import AssembiasZheng07Sats

# %%

simulation_list = ['base_c000_ph000', 'base_c112_ph000', 'base_c113_ph000']
redshift = 0.8


class IncompleteAssembiasZheng07Cens(AssembiasZheng07Cens):

    def __init__(self, **kwargs):
        AssembiasZheng07Cens.__init__(self, **kwargs)
        self.param_dict['f_compl'] = 1.0

    def mean_occupation(self, **kwargs):
        return (AssembiasZheng07Cens.mean_occupation(self, **kwargs) *
                self.param_dict['f_compl'])


cens_occ_model = IncompleteAssembiasZheng07Cens(
    prim_haloprop_key='halo_mvir', sec_haloprop_key='halo_nfw_conc')
sats_occ_model = AssembiasZheng07Sats(
    prim_haloprop_key='halo_mvir', sec_haloprop_key='halo_nfw_conc')
model = HodModelFactory(centrals_occupation=cens_occ_model,
                        satellites_occupation=sats_occ_model,
                        redshift=redshift)
n_dim = 8


def prediction(theta):
    model.param_dict['logMmin'] = 11.0 + theta[0] * 4
    model.param_dict['sigma_logM'] = theta[1]
    model.param_dict['alpha'] = theta[2] * 2
    model.param_dict['logM0'] = 11.0 + theta[3] * 4
    model.param_dict['logM1'] = 11.0 + theta[4] * 4
    model.param_dict['f_compl'] = theta[5]
    model.param_dict['mean_occupation_centrals_assembias_param1'] = (
        theta[6] * 2 - 1)
    model.param_dict['mean_occupation_satellites_assembias_param1'] = (
        theta[7] * 2 - 1)
    n, xi = halotab['xi0'].predict(model)
    xi = np.concatenate([xi, halotab['xi2'].predict(model)[1],
                         halotab['xi4'].predict(model)[1]])
    return n, xi


def chi_squared(theta):
    if len(theta.shape) == 2:
        return [chi_squared(t) for t in theta]
    n_mod, xi_mod = prediction(theta)
    chi_sq = (n_mod - n_obs)**2 / (0.01 * n_obs)**2
    chi_sq += np.dot(np.dot((xi_mod - xi_obs), xi_pre), xi_mod - xi_obs)
    return chi_sq

# %%


path = os.path.join(xolmis.BASE_DIR, 'desi-2', 'mocks')
fname_list = os.listdir(path)

for fname in fname_list:

    if fname[-5:] != '.hdf5':
        continue

    data = Table.read(os.path.join(xolmis.BASE_DIR, 'desi-2', 'mocks', fname))
    n_jk = data['xi0_jk'].shape[1]

    xi_obs = np.concatenate([data['xi0'], data['xi2'], data['xi4']])
    xi_cov = np.cov(np.vstack(
        [data['xi0_jk'], data['xi2_jk'], data['xi4_jk']]))
    xi_cov *= n_jk
    np.savetxt(fname.split('.')[0] + '_cov.csv', xi_cov)

    # downeight small scales in the first fit while there are issues with gumbo
    xi_cov[:5, :5] *= 100
    xi_cov[14:19, 14:19] *= 100
    xi_cov[28:33, 28:33] *= 100
    xi_pre = np.linalg.inv(xi_cov)
    xi_pre *= (n_jk - len(xi_obs) - 1) / n_jk
    n_obs = float(fname.split('.')[0].replace('p', '.'))

    table = Table()
    table['simulation'] = simulation_list
    table['n'] = np.zeros(len(table))
    table['xi'] = np.zeros((len(table), len(xi_obs)))

    for i, simulation in enumerate(simulation_list):
        halotab = xolmis.read_halotab(simulation, redshift)

        print(fname, simulation)
        if i == 0:
            result = minimize(chi_squared, x0=np.ones(n_dim) * 0.4, tol=1e-15,
                              bounds=[(0.2, 0.8) for i in range(n_dim)])
            theta = result.x
            n_obs, xi_obs = prediction(theta)
            # undo the downweighting
            xi_cov[:5, :5] /= 100
            xi_cov[14:19, 14:19] /= 100
            xi_cov[28:33, 28:33] /= 100
            xi_pre = np.linalg.inv(xi_cov)
            xi_pre *= (n_jk - len(xi_obs) - 1) / n_jk
        else:
            result = minimize(chi_squared, x0=theta, tol=1e-15,
                              bounds=[(0, 1) for i in range(n_dim)])
        print(result.x, result.fun)
        table['n'][i], table['xi'][i] = prediction(result.x)
    table.write(fname, path='data', overwrite=True)
