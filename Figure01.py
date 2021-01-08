"""
Plot Figure 01
the data must be placed at a subfolder '../Data' with 2nd level subfolders
'S1', 'S2', 'S3', 'S4', 'S3_old', 'S4_old', 'S5_old'

Older recordings are from
Storm, J.-H.; Drung, D.; Burghoff, M. & Körber, R.A modular, extendible and field-tolerant multichannel vector magnetometer based on current sensor SQUIDs Superconductor Science and Technology, IOP Publishing, 2016, 29, 094001
"""

import sys
sys.path.append('./helper_scripts')

from os import path
import os

import numpy as np
import scipy
import scipy.stats
import meet
import helper_functions
from plot_settings import *

data_folder = '../Data'
results_folder = '../Results'
if not path.exists(results_folder):
    os.makedirs(results_folder)

srate = 20000
srate_old = 5000 # for the older recordings

subjects = [
        'S1',
        'S2',
        'S3',
        'S4',
        'S3_old',
        'S4_old',
        'S5_old'
        ]

reference = 'SQUID_ref.dat'

relax_spectra = []

# loop through the subject folders and calculate the spectra
for subject in subjects:
    if subject.endswith('old'):
        conversion_factor = np.r_[np.loadtxt(path.join(data_folder,
            'Storm2016_conversion_factors.txt'))*1E15, 1]
        relax_data = helper_functions.readOldMEG(
            path.join(path.join(data_folder, subject),
                'MEG_{}_relax_Storm2016.flt'.format(subject)),
            num_chans=19, factor=conversion_factor[:,np.newaxis])[
                    [0, 16, 18]]
        relax_data = relax_data[1] - relax_data[0]
        f, relax_psd = helper_functions.stitch_spectral_bootstrap(
            relax_data.reshape(1,-1), fs=srate_old, initres=0.5,
            width_factor=10, factor=2, nit=1, calc_coherence=False)
    else:
        relax_file = 'MEG_%s_relax.dat' % subject.upper()
        relax_data = helper_functions.readMEG(
            path.join(path.join(data_folder, subject), relax_file),
            s_rate=srate, num_chans=2)[0]
        f, relax_psd = helper_functions.stitch_spectral_bootstrap(
            relax_data.reshape(1,-1), fs=srate, initres=0.5,
            width_factor=10, factor=2, nit=1, calc_coherence=False)
    relax_spectra.append([f, relax_psd[0,1]])

#calculate the spectrum of the reference recording
SQUID_data = helper_functions.readMEG(
        path.join(data_folder, reference), s_rate=srate, num_chans=1)
f, SQUID_psd = helper_functions.stitch_spectral_bootstrap(
    SQUID_data.reshape(1,-1), fs=srate, initres=0.5,
    width_factor=10, factor=2, nit=1, calc_coherence=False)

cmap_inst = mpl.cm.get_cmap(cmap+'_r')
#colors = [cmap_inst(i) for i in np.linspace(0,1,5)]
colors = np.array([
        [27,158,119,255],
        [217,95,2,255],
        [230,171,2,255],
        [231,41,138,255],
        [0,0,255,255]
        ])/255

colors[-1,-1] = 1
colors = np.vstack([colors, colors[-1], colors[-1]])

# plot the spectra
fig = plt.figure(figsize=(5.51181,3))
ax = fig.add_subplot(111, fc='w')
ax.grid(ls='-', c='k', alpha=0.15, linewidth=0.5)
ax.set_yscale('log')
ax.set_xscale('log')

lines = []

for i,x in enumerate(relax_spectra):
    if i < 4:
        lines.append(ax.plot(x[0], np.sqrt(x[1]), c=colors[i], lw=2,
            zorder=4)[0])
    else:
        lines.append(ax.plot(x[0], np.sqrt(x[1]), c='k', lw=0.75,
            ls='-', alpha=1, zorder=2)[0])
sq_noise, = ax.plot(f, np.sqrt(SQUID_psd[0,1]), c=colors[-1], alpha=1.0,
        ls='-',lw=2, zorder=3)
com_noise = mpl.patches.Rectangle([100,0.1], width=10000, height=1.9,
        fc='k', alpha=0.1, transform=ax.transData, zorder=0)
ax.add_patch(com_noise)

ana_window = mpl.patches.Rectangle([450,0.1], width=300, height=1000,
        fc='c', alpha=0.75, transform=ax.transData)
# alternative color: np.r_[117,112,179]/255
ax.add_patch(ana_window)

ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda x, p: format(int(x), ',').replace(',', r'\,')))

ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda x, p: format(int(x), ',').replace(',', r'\,') if x>=1 else format(x,'g')))

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('amplitude spectral density (fT/$\mathrm{\sqrt{Hz}}$)')
ax.legend([lines[0], lines[2], sq_noise, ana_window, lines[1], lines[3],
    lines[4], com_noise],
        ['subject S1', 'subject S3', r'system noise'+'\n'+r'(empty room)',
            'hfSER band-pass\n(450-750 Hz)', 'subject S2', 'subject S4',
            'data from\nStorm et al. (2016)',
            r'dominated by noise'+ '\n' + r'(comm. systems)'],
        loc='upper right', fontsize=7, ncol=2, framealpha=1)
ax.set_xlim((0.5, 10000))
ax.set_ylim(bottom=1E-1, top=1E3)
fig.tight_layout(pad=0.3)
fig.savefig(path.join(results_folder, 'Figure01.pdf'))
fig.savefig(path.join(results_folder, 'Figure01.png'))
