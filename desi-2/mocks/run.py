import tqdm
import xolmis
import itertools
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.cosmology import Planck15
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import tpcf_multipole
from halotools.empirical_models import enforce_periodicity_of_box
from scipy.spatial.transform import Rotation

# %%

gumbo = Table.read('gumbo_v0.0.h5')
gumbo = gumbo[gumbo['um_sm'] > 1e10]

# %%

dr_g = np.column_stack(
    (gumbo['galaxy_x'] - gumbo['unit_halo_x'],
     gumbo['galaxy_y'] - gumbo['unit_halo_y'],
     gumbo['galaxy_z'] - gumbo['unit_halo_z']))
dv_g = np.column_stack(
    (gumbo['galaxy_vx'] - gumbo['unit_halo_vx'],
     gumbo['galaxy_vy'] - gumbo['unit_halo_vy'],
     gumbo['galaxy_vz'] - gumbo['unit_halo_vz']))

m = Rotation.random(len(gumbo), random_state=0).as_matrix()

dr_r = np.einsum('...i,...ji', dr_g, m)
dv_r = np.einsum('...i,...ji', dv_g, m)

gumbo['galaxy_x'] = gumbo['galaxy_x'] - dr_g[:, 0] + dr_r[:, 0]
gumbo['galaxy_y'] = gumbo['galaxy_y'] - dr_g[:, 1] + dr_r[:, 1]
gumbo['galaxy_z'] = gumbo['galaxy_z'] - dr_g[:, 2] + dr_r[:, 2]
gumbo['galaxy_vx'] = gumbo['galaxy_vx'] - dv_g[:, 0] + dv_r[:, 0]
gumbo['galaxy_vy'] = gumbo['galaxy_vy'] - dv_g[:, 1] + dv_r[:, 1]
gumbo['galaxy_vz'] = gumbo['galaxy_vz'] - dv_g[:, 2] + dv_r[:, 2]

# %%


def ssfr_cut(log_mstar):
    return -11 + (log_mstar - 10)


plt.hist2d(np.log10(gumbo['um_sm']), gumbo['tng_lgssfr'], bins=100,
           norm=LogNorm())
cb = plt.colorbar()
plt.xlabel(r'Stellar Mass $\log M_\star [M_\odot]$')
plt.ylabel(r'sSFR $[\mathrm{yr}^{-1}]$')
plt.plot(np.array([10, 12]), ssfr_cut(np.array([10, 12])), ls='--',
         color='red')
plt.tight_layout(pad=0.3)
plt.savefig('mstar_vs_ssfr.pdf')
plt.savefig('mstar_vs_ssfr.png', dpi=300)
plt.close()

# %%

gumbo = gumbo[gumbo['tng_lgssfr'] < ssfr_cut(np.log10(gumbo['um_sm']))]

# %%

ngal_cut_list = [5e-4, 1e-3, 2e-3]

cosmology = Planck15.clone(H0=100)
period = 1000
redshift = 0.82

for ngal_cut in ngal_cut_list:

    table = Table()
    table.meta['ngal'] = ngal_cut

    ngal_cut *= 1e9
    mstar_cut = np.percentile(
        gumbo['um_sm'], 100 * (1 - (ngal_cut / len(gumbo))))
    select = gumbo['um_sm'] > mstar_cut
    pos = return_xyz_formatted_array(
        gumbo['galaxy_x'], gumbo['galaxy_y'], gumbo['galaxy_z'],
        velocity=gumbo['galaxy_vz'], velocity_distortion_dimension='z',
        redshift=redshift, cosmology=cosmology, period=period)[select]
    pos[:, 0] = enforce_periodicity_of_box(pos[:, 0], period)
    pos[:, 1] = enforce_periodicity_of_box(pos[:, 1], period)

    xi = xolmis.s_mu_tpcf_corrfunc(
        pos, xolmis.S_BINS['DESI-2'], xolmis.MU_BINS['DESI-2'],
        period=period)
    n_jk = 20
    for order in [0, 2, 4]:
        table['xi{}'.format(order)] = tpcf_multipole(
            xi, xolmis.MU_BINS['DESI-2'], order=order)
        table['xi{}_jk'.format(order)] = np.zeros((len(table), n_jk**2))
    for i_x, i_y in tqdm.tqdm(itertools.product(range(n_jk), range(n_jk)),
                              total=n_jk**2):
        select = ((pos[:, 0] < period / n_jk * i_x) |
                  (pos[:, 0] >= period / n_jk * (i_x + 1)) |
                  (pos[:, 1] < period / n_jk * i_y) |
                  (pos[:, 1] >= period / n_jk * (i_y + 1)))
        xi = xolmis.s_mu_tpcf_corrfunc(
            pos[select], xolmis.S_BINS['DESI-2'],
            xolmis.MU_BINS['DESI-2'], period=period)
        i = i_x * n_jk + i_y
        for order in [0, 2, 4]:
            table['xi{}_jk'.format(order)][:, i] = tpcf_multipole(
                xi, xolmis.MU_BINS['DESI-2'], order=order)

    table.write('{}'.format(ngal_cut / 1e9).replace('.', 'p') + '.hdf5',
                overwrite=True, path='data')
