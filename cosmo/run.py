import os
import xolmis
import matplotlib
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from tabulous.config import s_bins
from scipy.stats import norm as gauss

# %%

s = np.sqrt(s_bins['default'][1:] * s_bins['default'][:-1])

path = os.path.join(xolmis.BASE_DIR, 'fits')
fname_list = os.listdir(path)

simulation_list = ['base_c000_ph000', 'base_c103_ph000',
                   'base_c109_ph000', 'base_c113_ph000']
parameter_list = [None, 'omega_m', 'w_0', 'sigma_8']
perturbation_list = [None, -0.02255, -0.1, -0.807952 * 0.02]

# %%


def fit_cosmology(table, ngal, s_min, axarr=None, color=None, label=None,
                  center_on_truth=True):
    table = table[table['s_min'] == s_min]
    assert np.all(table['simulation'] == np.array(simulation_list))

    # Calculate the derivates with respect to cosmoloical parameters.
    d_xi = np.zeros((3, 3 * (len(s_bins['default']) - 1)))
    for i, (parameter, perturbation) in enumerate(
            zip(parameter_list, perturbation_list)):
        if perturbation is None:
            continue
        k = ['omega_m', 'w_0', 'sigma_8'].index(parameter)
        d_xi[k] += (table['xi'][i] - table['xi'][0]) / perturbation / (
            parameter_list.count(parameter))

    # Calculat the best fits and covariances of cosmological parameters.
    s = np.sqrt(s_bins['default'][1:] * s_bins['default'][:-1])
    xi_obs, xi_cov = xolmis.read_mock_observations(ngal)
    select = np.tile(s > s_min, 3)
    xi_obs = xi_obs[select] - table['xi'][0][select]
    xi_cov = xi_cov[np.outer(select, select)].reshape(
        (np.sum(select), np.sum(select)))

    cov = np.linalg.inv(np.dot(
        d_xi[:, select], np.dot(np.linalg.inv(xi_cov), d_xi[:, select].T)))
    mu = np.dot(np.dot(np.dot(cov, d_xi[:, select]), np.linalg.inv(xi_cov)),
                xi_obs)
    mu += np.array([0.31, -1.0, 0.81])

    if center_on_truth:
        mu = np.array([0.3089, -1, 0.8159])

    # Plot the results.
    if axarr is not None:

        for i in range(axarr.shape[0]):
            err = np.sqrt(np.diag(cov))[i]
            x = np.linspace(-2, +2, 100000)
            axarr[i, i].plot(x, gauss.pdf(x, loc=mu[i], scale=err),
                             color=color, label=label)

        for i in range(axarr.shape[0]):
            for j in range(axarr.shape[1]):
                if i > j:
                    cov_2d = np.zeros((2, 2))
                    cov_2d[0, 0] = cov[i, i]
                    cov_2d[1, 1] = cov[j, j]
                    cov_2d[0, 1] = cov[i, j]
                    cov_2d[1, 0] = cov[j, i]
                    m = np.linalg.cholesky(cov_2d)
                    phi = np.linspace(0, 2 * np.pi, 1000)
                    x = ((np.sin(phi) * m[0, 0] + np.cos(phi) * m[0, 1]) *
                         np.sqrt(2) + mu[i])
                    y = ((np.sin(phi) * m[1, 0] + np.cos(phi) * m[1, 1]) *
                         np.sqrt(2) + mu[j])
                    axarr[i, j].plot(y, x, color=color)

    return d_xi, mu, cov

# %%

err = []

for ngal in xolmis.NGAL:

    table = Table.read(os.path.join(
        xolmis.BASE_DIR, 'fits',  'hod_{}'.format(ngal).replace(
            '.', 'p') + '.hdf5'))

    err.append([])
    for s_min in np.unique(table['s_min']):
        err[-1].append(np.sqrt(np.diag(fit_cosmology(table, ngal, s_min)[2])))

err = np.array(err)

# %%

s_bin = 1
for param, label in zip([0, 1, 2], [r'$\Omega_{\rm m, 0}$', r'$w_0$',
                                    r'$\sigma_8$']):
    plt.plot(xolmis.NGAL, err[:, s_bin, param] / err[0, s_bin, param],
             label=label)
plt.title(r'$s_{\rm min} =' + '{:.1f}'.format(np.unique(
    table['s_min'])[s_bin]) + r'\, \mathrm{Mpc} / h$')
plt.legend(loc='best', frameon=False)
plt.xscale('log')
plt.ylim(ymin=0)
plt.gca().set_xticks([], minor=True)
plt.xticks(xolmis.NGAL, [r'$5 \times 10^{-4}$', r'$1 \times 10^{-3}$',
                         r'$2 \times 10^{-3}$'])
plt.xlabel(r'$n_{\rm gal} \, [h^3 \, \mathrm{Mpc}^{-3}]$')
plt.ylabel(r'Uncertaintity')
plt.tight_layout(pad=0.3)
plt.savefig('cosmo_vs_ngal.pdf')
plt.savefig('cosmo_vs_ngal.png', dpi=300)
plt.close()

# %%


for model_obs in ['hod']:
    for ngal in xolmis.NGAL:

        table = Table.read(os.path.join(
            xolmis.BASE_DIR, 'fits',  model_obs + '_{}'.format(ngal).replace(
                '.', 'p') + '.hdf5'))

        fig, axarr = plt.subplots(figsize=(3.5, 3.5), ncols=3, nrows=3,
                                  sharex='col')
        truth = np.array([0.3089, -1, 0.8159])
        x_min = truth
        x_max = truth

        s_min_list = [0.1, 1.0, 4.0, 10.0]

        cmap = matplotlib.cm.plasma
        vmin, vmax = np.log(np.amin(s_min_list)), np.log(np.amax(s_min_list))
        vmin = vmin - (vmax - vmin) * 0.1
        vmax = vmax + (vmax - vmin) * 0.1
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for s_min in s_min_list:
            label = (r'$s_{\rm min} = ' + '{:g}'.format(s_min) +
                     r'\, \mathrm{Mpc}/h$')
            d_xi, mu, cov = fit_cosmology(
                table, ngal, s_min, axarr=axarr,
                color=cmap(norm(np.log(s_min))),
                center_on_truth=(model_obs == 'hod'),
                label=label)
            x_min = np.minimum(x_min, mu - 2.5 * np.sqrt(np.diag(cov)))
            x_max = np.maximum(x_max, mu + 2.5 * np.sqrt(np.diag(cov)))

        for i in range(axarr.shape[0]):
            axarr[i, i].set_xlim(x_min[i], x_max[i])
            axarr[i, i].set_ylim(ymin=0)
            if model_obs == 'sham':
                axarr[i, i].axvline(truth[i], ls='--', color='black')

        for i in range(axarr.shape[0]):
            for j in range(axarr.shape[1]):
                if i < j:
                    axarr[i, j].axis('off')
                if j > 0 or i == 0:
                    axarr[i, j].set_yticks([])
                if i > j:
                    axarr[i, j].set_ylim(x_min[i], x_max[i])
                    if model_obs == 'sham':
                        axarr[i, j].scatter(truth[j], truth[i], marker='*',
                                            color='black', zorder=99)

        axarr[1, 0].set_ylabel(r'$w_0$')
        axarr[2, 0].set_ylabel(r'$\sigma_8$')
        axarr[2, 0].set_xlabel(r'$\Omega_{\rm m}$')
        axarr[2, 1].set_xlabel(r'$w_0$')
        axarr[2, 2].set_xlabel(r'$\sigma_8$')

        handles, labels = axarr[0, 0].get_legend_handles_labels()
        axarr[0, 2].legend(handles, labels, loc='right', frameon=False)

        for ax in axarr[:, 0]:
            for tick in ax.get_yticklabels():
                tick.set_rotation(45)

        for ax in axarr[2]:
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(model_obs + '_{}'.format(ngal).replace('.', 'p') + '.pdf')
        plt.savefig(model_obs + '_{}'.format(ngal).replace('.', 'p') + '.png',
                    dpi=300)
        plt.close()


# %%

for ngal in xolmis.NGAL:

    xi_cov = xolmis.read_mock_observations(ngal)[1]

    table = Table.read(os.path.join(
        xolmis.BASE_DIR, 'fits', 'hod_{}'.format(ngal).replace('.', 'p') +
        '.hdf5'))

    fig, axarr = plt.subplots(figsize=(7, 3.5), ncols=3, sharex=True,
                              sharey=True)
    axarr[0].set_title(r'$\Omega_{\rm m}$')
    axarr[1].set_title(r'$w_0$')
    axarr[2].set_title(r'$\sigma_8$')

    d_xi = fit_cosmology(table, ngal, 0.1)[0]

    for k in range(3):
        for i, (color, order) in enumerate(
                zip(matplotlib.cm.plasma([0.8, 0.5, 0.2]), [0, 2, 4])):
            axarr[k].plot(s, np.split(d_xi[k] / np.sqrt(np.diag(xi_cov)),
                                      3)[i], color=color)

    plt.xscale('log')

    yabsmax = 0

    for ax in axarr:
        ymin, ymax = ax.get_ylim()
        yabsmax = max(yabsmax, abs(ymax))
        yabsmax = max(yabsmax, abs(ymin))
        ax.axhline(0, ls=':', color='black', zorder=0)
        ax.set_xlabel(r'$s \, [h^{-1} \, \mathrm{Mpc}]$')

    for color, order in zip(matplotlib.cm.plasma([0.8, 0.5, 0.2]), [0, 2, 4]):
        axarr[0].text(0.05 + order / 20, 0.05, r'$\xi_' + str(order) + '$',
                      color=color, transform=axarr[0].transAxes, ha='left',
                      va='bottom')
    axarr[0].set_ylabel(r'$\partial D / \partial \theta \, \sigma_D^{-1}$')
    plt.ylim(-yabsmax, +yabsmax)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    plt.savefig('{}'.format(ngal).replace('.', 'p') + '_der.pdf')
    plt.savefig('{}'.format(ngal).replace('.', 'p') + '_der.png', dpi=300)
    plt.close()
