"""
import h5py
import numpy as np
from astropy.table import Table

fstream = h5py.File('/project/projectdirs/desi/cosmosim/FirstGenMocks/Uchuu/CubicBox/BGS/z0.190/BGS_box_Uchuu.hdf5', 'r')

abs_mag = fstream['Data/abs_mag'][()]
mask = abs_mag < -19
abs_mag = abs_mag[mask]
g_r = fstream['Data/g_r'][()][mask]
pos = fstream['Data/pos'][()][mask]
vel = fstream['Data/vel'][()][mask]

table = Table()
table['M'] = abs_mag.astype(np.float32)
table['g-r'] = g_r.astype(np.float32)
table['x'] = pos[:, 0].astype(np.float32)
table['y'] = pos[:, 1].astype(np.float32)
table['z'] = pos[:, 2].astype(np.float32)
table['vx'] = vel[:, 0].astype(np.float32)
table['vy'] = vel[:, 1].astype(np.float32)
table['vz'] = vel[:, 2].astype(np.float32)
table.write('mock.hdf5', path='mock', overwrite=True)
fstream.close()
"""

import tqdm
import xolmis
import itertools
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import tpcf_multipole
from halotools.empirical_models import enforce_periodicity_of_box

mock = Table.read('mock.hdf5')

# %%

plt.hist2d(mock['M'], mock['g-r'], bins=100)
cb = plt.colorbar()
plt.xlabel(r'Absolute magnitude $M_r$')
plt.ylabel(r'Color $g-r$')
m_plot = np.linspace(np.amin(mock['M']), np.amax(mock['M']), 10)
plt.plot(m_plot, 0.21 - 0.03 * m_plot, ls='--', color='red')
plt.tight_layout(pad=0.3)
plt.savefig('mabs_vs_gr.pdf')
plt.savefig('mabs_vs_gr.png', dpi=300)
plt.close()

# %%

mock = mock[mock['g-r'] > 0.21 - 0.03 * mock['M']]

# %%

ngal_cut_list = [5e-4, 1e-3, 2e-3]

cosmology = Planck15.clone(H0=100)
period = 2000
redshift = 0.19

for ngal_cut in ngal_cut_list:

    table = Table()
    table.meta['ngal'] = ngal_cut

    ngal_cut *= period**3
    mabs_cut = np.percentile(
        mock['M'], 100 * (1 - (ngal_cut / len(mock))))
    select = mock['M'] > mabs_cut
    pos = return_xyz_formatted_array(
        mock['x'], mock['y'], mock['z'], velocity=mock['vz'],
        velocity_distortion_dimension='z', redshift=redshift,
        cosmology=cosmology, period=period)[select]
    pos[:, 0] = enforce_periodicity_of_box(pos[:, 0], period)
    pos[:, 1] = enforce_periodicity_of_box(pos[:, 1], period)

    xi = xolmis.s_mu_tpcf_corrfunc(
        pos, xolmis.S_BINS['DESI-2'], xolmis.MU_BINS['DESI-2'],
        period=period)
    n_jk = 10
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

    table.write('{}'.format(ngal_cut / period**3).replace('.', 'p') + '.hdf5',
                overwrite=True, path='data')
