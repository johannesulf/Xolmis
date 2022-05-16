import tqdm
import xolmis
import itertools
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from Corrfunc.theory import DDsmu
from astropy.cosmology import Planck15
from tabulous.config import mu_bins, s_bins
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import tpcf_multipole

mu_bins = mu_bins['default']
s_bins = s_bins['default']

mock = Table.read('uchuu_bgs.hdf5')

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
boxsize = 2000
redshift = 0.19
n_threads = 4


def xi_from_dd(dd, n, boxsize, s_bins, mu_bins):
    rr = (n**2 / boxsize**3 * 4 * np.pi / 3 * np.diff(s_bins**3) /
          (len(mu_bins) - 1))
    return (dd.reshape((len(s_bins) - 1, len(mu_bins) - 1)) /
            rr[:, np.newaxis] - 1)


for ngal in xolmis.NGAL:

    table = Table()
    table.meta['ngal'] = ngal

    ngal *= boxsize**3
    mabs_cut = np.percentile(mock['M'], 100 * (ngal / len(mock)))
    select = mock['M'] < mabs_cut

    dd_all = np.zeros((len(s_bins) - 1) * (len(mu_bins) - 1))
    for xyz in ['xyz', 'yzx', 'zxy']:

        pos = return_xyz_formatted_array(
            x=mock[xyz[0]], y=mock[xyz[1]], z=mock[xyz[2]],
            velocity=mock['v'+xyz[2]], velocity_distortion_dimension='z',
            period=boxsize, redshift=redshift, cosmology=cosmology)[select]
        pos = pos.astype(float)
        dd_all += DDsmu(1, n_threads, s_bins, 1, len(mu_bins) - 1, pos[:, 0],
                        pos[:, 1], pos[:, 2], periodic=True, boxsize=boxsize)[
                            'npairs'] / 3.0

    xi = xi_from_dd(dd_all, len(pos), boxsize, s_bins, mu_bins)
    n_jk = 10
    for order in [0, 2, 4]:
        table['xi{}'.format(order)] = tpcf_multipole(xi, mu_bins, order=order)
        table['xi{}_jk'.format(order)] = np.zeros((len(table), n_jk**3))
    for i_x, i_y, i_z in tqdm.tqdm(itertools.product(
            range(n_jk), range(n_jk), range(n_jk)), total=n_jk**3):
        select = ~((pos[:, 0] < boxsize / n_jk * i_x) |
                   (pos[:, 0] >= boxsize / n_jk * (i_x + 1)) |
                   (pos[:, 1] < boxsize / n_jk * i_y) |
                   (pos[:, 1] >= boxsize / n_jk * (i_y + 1)) |
                   (pos[:, 2] < boxsize / n_jk * i_z) |
                   (pos[:, 2] >= boxsize / n_jk * (i_z + 1)))

        dd_auto = DDsmu(1, 4, s_bins, 1, len(mu_bins) - 1, pos[:, 0][select],
                        pos[:, 1][select], pos[:, 2][select], periodic=True,
                        boxsize=boxsize)['npairs']
        dd_cross = DDsmu(0, 4, s_bins, 1, len(mu_bins) - 1,
                         pos[:, 0][~select], pos[:, 1][~select],
                         pos[:, 2][~select], X2=pos[:, 0][select],
                         Y2=pos[:, 1][select], Z2=pos[:, 2][select],
                         periodic=True, boxsize=boxsize)['npairs'] * 2

        dd_jk = dd_all - dd_auto - dd_cross
        xi = xi_from_dd(dd_jk, np.sum(~select), boxsize, s_bins, mu_bins)
        i = i_x * n_jk**2 + i_y * n_jk + i_z
        for order in [0, 2, 4]:
            table['xi{}_jk'.format(order)][:, i] = tpcf_multipole(
                xi, mu_bins, order=order)

    table.write('{}'.format(ngal / boxsize**3).replace('.', 'p') + '.hdf5',
                overwrite=True, path='data')
