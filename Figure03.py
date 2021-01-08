"""
Plot Figure 03 for a subject
the data must be placed at a subfolder '../Data' with 2nd level subfolders
'S1', 'S2', 'S3', 'S4', 'S3_old', 'S4_old', 'S5_old'

The subject to be analyzed can be given as a commandline argument to the script.
If not provided, subject S2 will be analyzed.

Older recordings are from

Storm, J.-H.; Drung, D.; Burghoff, M. & Körber, R.A modular, extendible and field-tolerant multichannel vector magnetometer based on current sensor SQUIDs Superconductor Science and Technology, IOP Publishing, 2016, 29, 094001
"""

import sys
try:
    subject = sys.argv[1]
except:
    subject = 'S2'
print(('Analyzing subject {}'.format(subject)))

sys.path.append('./helper_scripts')

import os
from os import path

import numpy as np
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import meet
import helper_functions
from tqdm import trange
from plot_settings import *

data_folder = '../Data'
results_folder = path.join('../Results', subject)
if not path.exists(results_folder):
    os.makedirs(results_folder)

if subject.endswith('_old'):
    srate = 5000
    conversion_factor = np.r_[np.loadtxt(path.join(data_folder,
        'Storm2016_conversion_factors.txt'))*1E15, 1]
    if subject == 'S3_old':
        idx = [0, 15, 18]
    elif subject == 'S4_old':
        idx = [0, 9, 18]
    elif subject == 'S5_old':
        idx = [0, 9, 18]
    stim_data = helper_functions.readOldMEG(
        path.join(path.join(data_folder, subject),
            'MEG_{}_stim_Storm2016.flt'.format(subject)),
        num_chans=19, factor=conversion_factor[:,np.newaxis])[idx]
    stim_data = np.vstack([
        stim_data[1] - stim_data[0],
        stim_data[-1]])
else:
    srate = 20000
    stim_file = 'MEG_%s_stim.dat' % subject.upper()
    stim_data = helper_functions.readMEG(
            path.join(path.join(data_folder, subject), stim_file),
            s_rate=srate, num_chans=2)

# get the stimuli, omit the first (avoid edge effects),
# and keep only the first 2500 (in Storm et al. 2016, more that 16000)
# had been collected
marker = (
        (stim_data[-1,1:]>250000) &
        (stim_data[-1,:-1]<250000)).nonzero()[0][1:2501]

# take care that we have at least 200 ms after the last marker
if (marker[-1] + 0.2*srate) >= len(stim_data):
    marker = marker[:-1]

# remove the marker channel
stim_data = stim_data[0]

# interpolate the stimulus
interpolate_win_ms = [-2, 2]
interpolate_win = np.round(np.array(interpolate_win_ms)
        / 1000. * srate).astype(int)
stim_data = meet.interpolateEEG(stim_data, marker, interpolate_win)

# apply 450Hz-750 Hz band-pass filter
stim_data_sigma = meet.iir.butterworth(stim_data,
        fs=(400,800), fp=(450,750), s_rate=srate)

# get the trials
trial_win_ms = [-60,160]
trial_win = np.round(np.array(trial_win_ms)/1000.*srate
        ).astype(int)
trial_t = (np.arange(trial_win[0], trial_win[1], 1)/
        float(srate)*1000)

trials_sigma = meet.epochEEG(stim_data_sigma, marker,
        trial_win)

########################
# plot all the results #
########################
fig = plt.figure(figsize=(7.48, 3))
gs = mpl.gridspec.GridSpec(nrows=2, ncols=2, figure=fig, height_ratios=[1,20])

all_trials_ax = fig.add_subplot(gs[1,0])
all_trials_ax.set_xlabel('time relative to stimulus (ms)')
all_trials_ax.set_ylabel('trial index')

sub_trials_ax = fig.add_subplot(gs[1,1])
sub_trials_ax.set_xlabel('time relative to stimulus (ms)')
sub_trials_ax.set_ylabel('trial index')

sub_trials_ax.tick_params(right=True, labelright=True,
        left=False, labelleft=False)
sub_trials_ax.yaxis.set_label_position('right')

colorbar_ax = fig.add_subplot(gs[0,:])

all_trials_tf = all_trials_ax.pcolormesh(
        helper_functions.extend_by_one(trial_t),
        np.arange(1, trials_sigma.shape[-1] + 2, 1),
        trials_sigma.T,
        rasterized=True, shading='flat', cmap='coolwarm',
        vmin=-20, vmax=20)

time_mask = np.all([trial_t>=10, trial_t<=35], 0)

sub_trials_tf = sub_trials_ax.pcolormesh(
        helper_functions.extend_by_one(trial_t[time_mask]),
        np.arange(850, 926, 1),
        trials_sigma[time_mask][:,850:925].T,
        rasterized=True, shading='flat', cmap='coolwarm',
        vmin=-20, vmax=20)

all_trials_ax.axvline(0, ls='-', c='k', lw=0.5)
all_trials_ax.set_xlim([-10,100])
all_trials_ax.set_ylim(bottom=1)

cbar = plt.colorbar(all_trials_tf, cax=colorbar_ax,
        label=r'amplitude (fT)', orientation='horizontal')
cbar.solids.set_edgecolor('face')
cbar.solids.set_rasterized(True)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(top=True, labeltop=True, bottom=False,
        labelbottom=False)

Rect = mpl.patches.Rectangle((10,850), width=25, height=75, fill=False,
        edgecolor='k', transform=all_trials_ax.transData, linewidth=1,
        alpha=1.0)
all_trials_ax.add_patch(Rect)

all_trials_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
sub_trials_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
all_trials_ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda x, p: format(int(x), ',').replace(',', r'\,')))
sub_trials_ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda x, p: format(int(x), ',').replace(',', r'\,')))

fig.tight_layout(pad=0.2, w_pad=2.0, h_pad=0.75)
fig.canvas.draw()

dx, dy = 0, -2/72.
offset = mpl.transforms.ScaledTranslation(dx, dy,
  fig.dpi_scale_trans)
offset2 = mpl.transforms.ScaledTranslation(0,
        all_trials_ax.get_xticklabels()[1].get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin -
        all_trials_ax.get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin,
        fig.dpi_scale_trans)
all_trials_shift_transform = mpl.transforms.blended_transform_factory(
        all_trials_ax.transData, all_trials_ax.transAxes) + offset + offset2

all_trials_ax.text(0, 0, r'\rotatebox[origin=c]{180}{\Lightning}',
        ha='right', va='top', size=16, transform=all_trials_shift_transform)

#get the coordinates of the sub_trials axis with the coordinates of
# the all_trials axes
sub_trials_ax_pos = all_trials_ax.transData.inverted().transform(
        fig.transFigure.transform(sub_trials_ax.get_position()))

all_trials_ax.plot([10,sub_trials_ax_pos[0,0]],
        [850, sub_trials_ax_pos[0,1]], 'k-', clip_on=False, lw=1.0,
        alpha=0.5)
all_trials_ax.plot([10,sub_trials_ax_pos[0,0]],
        [925, sub_trials_ax_pos[1,1]], 'k-', clip_on=False, lw=1.0,
        alpha=0.5)

all_trials_yticks = all_trials_ax.yaxis.get_major_ticks()
all_trials_yticks[0].label1.set_visible(False)

fig.savefig(path.join(results_folder, 'Figure03_{}.pdf'.format(subject)),
        dpi=300)
fig.savefig(path.join(results_folder, 'Figure03_{}.png'.format(subject)),
        dpi=300)
