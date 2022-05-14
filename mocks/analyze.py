import os
import matplotlib
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from tabulous.config import s_bins

fname_list = os.listdir()

# %%

s = np.sqrt(s_bins['default'][1:] * s_bins['default'][:-1])
n_list = []
xi0_rel_err_list = []
xi2_rel_err_list = []
xi4_rel_err_list = []

for fname in fname_list:

    try:
        n = float(fname.split('.')[0].replace('p', '.'))
    except ValueError:
        continue

    table = Table.read(fname)
    xi = np.concatenate([table['xi0'], table['xi2'], table['xi4']])
    n_jk = table['xi0_jk'].shape[1]

    xi = np.concatenate([table['xi0'], table['xi2'], table['xi4']])
    xi_cov = np.cov(np.vstack(
        [table['xi0_jk'], table['xi2_jk'], table['xi4_jk']])) * n_jk
    xi_err = np.sqrt(np.diag(xi_cov))

    for i, (color, order) in enumerate(
            zip(matplotlib.cm.plasma([0.8, 0.5, 0.2]), [0, 2, 4])):
        plotline, caps, barlinecols = plt.errorbar(
            s * (1 + order / 50), s**1.5 * np.split(xi, 3)[i],
            yerr=10 * s**1.5 * np.split(xi_err, 3)[i], color=color, fmt='o',
            ms=4)
        plt.setp(barlinecols[0], capstyle='round')
        plt.text(0.05 + order / 30, 0.05, r'$\xi_' + str(order) + '$',
                 color=color, transform=plt.gca().transAxes, ha='left',
                 va='bottom')

    plt.xlabel(r'$s \, [h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$s^{1.5} \times \xi_n \, [h^{-1.5} \, \mathrm{Mpc}^{1.5}]$')
    plt.xscale('log')
    plt.tight_layout(pad=0.3)
    plt.savefig(fname[:-5] + '_obs.pdf')
    plt.savefig(fname[:-5] + '_obs.png', dpi=300)
    plt.close()

    plt.figure(figsize=(3, 3))
    xi_cor = xi_cov / np.outer(np.sqrt(np.diag(xi_cov)),
                               np.sqrt(np.diag(xi_cov)))
    plt.imshow(xi_cor, vmin=-1, vmax=+1, cmap='PuOr', origin='lower')
    plt.xticks([7, 21, 35], [r'$\xi_0$', r'$\xi_2$', r'$\xi_4$'])
    plt.yticks([7, 21, 35], [r'$\xi_0$', r'$\xi_2$', r'$\xi_4$'])
    plt.savefig(fname[:-5] + '_cov.pdf')
    plt.savefig(fname[:-5] + '_cov.png', dpi=300)
    plt.close()

    n_list.append(n)
    xi0_rel_err_list.append(np.split(xi_err, 3)[0] / np.split(xi, 3)[0])
    xi2_rel_err_list.append(np.split(xi_err, 3)[1] / np.split(xi, 3)[0])
    xi4_rel_err_list.append(np.split(xi_err, 3)[2] / np.split(xi, 3)[0])

# %%

cmap = matplotlib.cm.plasma
vmin, vmax = np.log(np.amin(n_list)), np.log(np.amax(n_list))
vmin = vmin - (vmax - vmin) * 0.2
vmax = vmax + (vmax - vmin) * 0.2
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

for order, xi_rel_err_list in zip(
        [0, 2, 4], [xi0_rel_err_list, xi2_rel_err_list, xi4_rel_err_list]):
    for n, xi_rel_err in zip(n_list, xi_rel_err_list):
        b = int(np.floor(np.log10(n)))
        label = (r'$n_{\rm gal} =' + '{:.0f}'.format(n / 10**b) +
                 r'\times 10^{' + '{}'.format(b) + '}$')
        plt.plot(s, 100 * xi_rel_err, color=cmap(norm(np.log(n))), label=label)

    plt.xlabel(r'$s \, [h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$\sigma_{\xi_' + str(order) + r'} / \xi_0 \, [\%]$')
    plt.ylim(0.1, 10.0)
    plt.legend(loc='best', frameon=False)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout(pad=0.3)
    plt.savefig('xi{}_pre.pdf'.format(order))
    plt.close()
