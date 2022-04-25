import os
import xolmis
import numpy as np
from astropy.table import Table
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm as gauss

# %%

path = os.path.join(xolmis.BASE_DIR, 'desi-2', 'fits')
fname_list = os.listdir(path)

simulation_list = ['base_c000_ph000', 'base_c102_ph000', 'base_c108_ph000',
                   'base_c109_ph000', 'base_c112_ph000', 'base_c113_ph000']
parameter_list = [None, 'omega_m', 'w_0', 'w_0', 'sigma_8', 'sigma_8']
perturbation_list = [None, +0.02255, +0.1, -0.1, +0.807952 * 0.02,
                     -0.807952 * 0.02]
cov_list = []
n_list = []

for fname in fname_list:

    if fname[-5:] != '.hdf5':
        continue

    n_list.append(float(fname.split('.')[0].replace('p', '.')))

    table = Table.read(os.path.join(path, fname))
    xi_cov = np.genfromtxt(os.path.join(path, fname[:-5] + '_cov.csv'))

    assert np.all(table['simulation'] == np.array(simulation_list))

    s = np.sqrt(
        xolmis.S_BINS['default'][1:] * xolmis.S_BINS['default'][:-1])[5:]
    fig, axarr = plt.subplots(figsize=(7, 3.5), ncols=3, sharex=True,
                              sharey=True)
    axarr[0].set_title(r'$\Omega_{\rm m}$')
    axarr[1].set_title(r'$w_0$')
    axarr[2].set_title(r'$\sigma_8$')

    d_xi_dict = {}

    for i, (parameter, perturbation) in enumerate(zip(
            parameter_list, perturbation_list)):

        if i == 0:
            continue

        ax = axarr[['omega_m', 'w_0', 'sigma_8'].index(parameter)]
        d_xi = (table['xi'][i] - table['xi'][0]) / perturbation
        if parameter in d_xi_dict.keys():
            d_xi_dict[parameter] = (d_xi + d_xi_dict[parameter]) / 2.0
        else:
            d_xi_dict[parameter] = d_xi

        for k, (color, order) in enumerate(
                zip(matplotlib.cm.plasma([0.8, 0.5, 0.2]), [0, 2, 4])):
            ax.plot(s, np.split(d_xi / np.sqrt(np.diag(xi_cov)), 3)[k],
                    color=color, ls='--' if perturbation > 0 else '-')

    plt.xscale('log')

    yabsmax = 0

    for ax in axarr:
        ymin, ymax = ax.get_ylim()
        yabsmax = max(yabsmax, abs(ymax))
        yabsmax = max(yabsmax, abs(ymin))
        ax.axhline(0, ls=':', color='black', zorder=0)
        ax.set_xlabel(r'$s \, [h^{-1} \, \mathrm{Mpc}]$')

    axarr[0].set_ylabel(r'$\partial D / \partial \theta \, \sigma_D^{-1}$')
    plt.ylim(-yabsmax, +yabsmax)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    plt.savefig(fname[:-5] + '_derivative.pdf')
    plt.savefig(fname[:-5] + '_derivative.png', dpi=300)
    plt.close()

    d_xi = np.vstack([d_xi_dict['omega_m'], d_xi_dict['w_0'],
                      d_xi_dict['sigma_8']])
    cov_list.append(
        np.linalg.inv(np.dot(d_xi, np.dot(np.linalg.inv(xi_cov), d_xi.T))))

# %%

fig, axarr = plt.subplots(figsize=(3.5, 3.5), ncols=cov_list[0].shape[0],
                          nrows=cov_list[0].shape[0], sharex='col')
x0 = [0.31, -1.0, 0.81]

cmap = matplotlib.cm.plasma
vmin, vmax = np.log(np.amin(n_list)), np.log(np.amax(n_list))
vmin = vmin - (vmax - vmin) * 0.2
vmax = vmax + (vmax - vmin) * 0.2
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


for i in range(axarr.shape[0]):
    for j in range(axarr.shape[1]):
        if i < j:
            axarr[i, j].axis('off')
        elif i == j:
            err_max = 0
            for n, cov in zip(n_list, cov_list):
                err = np.sqrt(np.diag(cov))[i]
                err_max = max(err_max, err)
                x = np.linspace(-5 * err, +5 * err, 1000) + x0[i]
                axarr[i, j].plot(x, gauss.pdf(x, loc=x0[i], scale=err),
                                 color=cmap(norm(np.log(n))))

            axarr[i, j].set_xlim(x0[i] - 2 * err_max, x0[i] + 2 * err_max)
            axarr[i, j].set_ylim(ymin=0)
            axarr[i, j].set_yticks([])

for i in range(axarr.shape[0]):
    for j in range(axarr.shape[1]):
        if i > j:
            for n, cov in zip(n_list, cov_list):
                cov_2d = np.zeros((2, 2))
                cov_2d[0, 0] = cov[i, i]
                cov_2d[1, 1] = cov[j, j]
                cov_2d[0, 1] = cov[i, j]
                cov_2d[1, 0] = cov[j, i]
                m = np.linalg.cholesky(cov_2d)
                phi = np.linspace(0, 2 * np.pi, 1000)
                x = ((np.sin(phi) * m[0, 0] + np.cos(phi) * m[0, 1]) *
                     np.sqrt(2) + x0[i])
                y = ((np.sin(phi) * m[1, 0] + np.cos(phi) * m[1, 1]) *
                     np.sqrt(2) + x0[j])
                axarr[i, j].plot(y, x, color=cmap(norm(np.log(n))))
                ymin, ymax = axarr[i, i].get_xlim()
                axarr[i, j].set_ylim(ymin, ymax)
                if j > 0:
                    axarr[i, j].set_yticks([])

axarr[1, 0].set_ylabel(r'$w_0$')
axarr[2, 0].set_ylabel(r'$\sigma_8$')
axarr[2, 0].set_xlabel(r'$\Omega_{\rm m}$')
axarr[2, 1].set_xlabel(r'$w_0$')
axarr[2, 2].set_xlabel(r'$\sigma_8$')

for ax in axarr[:, 0]:
    for tick in ax.get_yticklabels():
        tick.set_rotation(45)

for ax in axarr[2]:
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('cosmo.pdf')
plt.savefig('cosmo.png', dpi=300)
