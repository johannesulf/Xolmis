import os
import xolmis
import numpy as np
from astropy.table import Table
from tabulous import read_tabcorr
from tabulous.config import s_bins
from scipy.optimize import minimize
from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import AssembiasZheng07Cens
from halotools.empirical_models import AssembiasZheng07Sats

# %%

simulation_list = ['base_c000_ph000', 'base_c103_ph000', 'base_c109_ph000',
                   'base_c113_ph000']#, 'base_c112_ph000', 'base_c113_ph000']
redshift = 0.2
n_dim = 11

# %%

class IncompleteAssembiasZheng07Cens(AssembiasZheng07Cens):

    def __init__(self, **kwargs):
        AssembiasZheng07Cens.__init__(self, **kwargs)
        self.param_dict['f_compl'] = 1.0

    def mean_occupation(self, **kwargs):
        return (AssembiasZheng07Cens.mean_occupation(self, **kwargs) *
                self.param_dict['f_compl'])


def prediction(theta):
    model.param_dict['logMmin'] = 11.0 + theta[0] * 4
    model.param_dict['sigma_logM'] = theta[1] * 2.0
    model.param_dict['alpha'] = theta[2] * 2
    model.param_dict['logM0'] = 11.0 + theta[3] * 4
    model.param_dict['logM1'] = 11.0 + theta[4] * 4
    model.param_dict['f_compl'] = 0.01 + theta[5] * 0.99
    model.param_dict['mean_occupation_centrals_assembias_param1'] = (
        theta[6] * 2 - 1)
    model.param_dict['mean_occupation_satellites_assembias_param1'] = (
        theta[7] * 2 - 1)
    model.param_dict['alpha_s'] = 0.8 + 0.4 * theta[8]
    model.param_dict['alpha_c'] = 0.4 * theta[9]
    model.param_dict['log_eta'] = -np.log(3) + theta[10] * 2 * np.log(3)
    n, xi0 = halotab['xi0'].predict(model, same_halos=True, extrapolate=True)
    n, xi2 = halotab['xi2'].predict(model, same_halos=True, extrapolate=True)
    n, xi4 = halotab['xi4'].predict(model, same_halos=True, extrapolate=True)
    xi = np.concatenate([xi0, xi2, xi4])
    return n, xi


def chi_squared(theta):
    n_mod, xi_mod = prediction(theta)
    chi_sq = (n_mod - n_obs)**2 / (0.01 * n_obs)**2
    s = np.sqrt(s_bins['default'][1:] * s_bins['default'][:-1])
    select = np.tile(s > s_min, 3)
    d_xi = (xi_mod - xi_obs)[select]
    chi_sq += np.dot(np.dot((d_xi), xi_pre), d_xi)
    return chi_sq

# %%


path = os.path.join(xolmis.BASE_DIR, 'mocks')
fname_list = os.listdir(path)
s_min_list = [0.1, 1.0, 5.0, 10.0]

print('n_gal \t simulation \t\t s_min \t type \t chi^2')

for model_obs in ['sham', 'hod']:
    for n_obs in xolmis.NGAL:

        xi_obs, xi_cov = xolmis.read_mock_observations(n_obs)

        table = Table()
        table['simulation'] = np.repeat(simulation_list, len(s_min_list))
        table['s_min'] = np.tile(s_min_list, len(simulation_list))
        table['n'] = np.zeros(len(table))
        table['xi'] = np.zeros((len(table), len(xi_obs)))

        guess = np.ones(n_dim) * 0.5

        for i in range(len(table)):

            simulation = table['simulation'][i]
            s_min = table['s_min'][i]

            xi_obs, xi_cov = xolmis.read_mock_observations(n_obs)
            s = np.sqrt(s_bins['default'][1:] * s_bins['default'][:-1])
            select = np.tile(s > s_min, 3)
            xi_pre = np.linalg.inv(xi_cov[np.outer(select, select)].reshape(
                (np.sum(select), np.sum(select))))

            halotab = {}
            for order in [0, 2, 4]:
                halotab['xi{}'.format(order)] = read_tabcorr(
                    'AbacusSummit', int(simulation[6:9]), redshift,
                    'xi{}'.format(order))

            prim_haloprop_key = halotab['xi0'].tabcorr_list[0].attrs[
                'prim_haloprop_key']
            cens_occ_model = IncompleteAssembiasZheng07Cens(
                prim_haloprop_key=prim_haloprop_key,
                sec_haloprop_key='halo_nfw_conc')
            sats_occ_model = AssembiasZheng07Sats(
                prim_haloprop_key=prim_haloprop_key,
                sec_haloprop_key='halo_nfw_conc')
            model = HodModelFactory(centrals_occupation=cens_occ_model,
                                    satellites_occupation=sats_occ_model,
                                    redshift=redshift)
            bounds = [(0.0, 1.0) for i in range(n_dim)]

            if i == 0 and model_obs == 'hod':
                bounds = [(0.2, 0.8) for i in range(n_dim)]

            result = minimize(chi_squared, x0=guess, tol=1e-15, bounds=bounds)
            best_fit = result.x

            if i == 0:
                guess = best_fit

            if i == 0 and model_obs == 'hod':
                xi_obs = prediction(best_fit)[1]

            if model_obs == 'hod' and np.any(np.abs(result.x - 0.5) > 0.49):
                print('Warning! Fit ran into the prior!')

            print('{} \t {} \t {:.1f} \t {} \t {:.1f}'.format(
                n_obs, simulation, s_min, model_obs, result.fun))
            table['n'][i], table['xi'][i] = prediction(best_fit)

        table.write(model_obs + '_{}'.format(n_obs).replace('.', 'p') +
                    '.hdf5', path='data', overwrite=True)
