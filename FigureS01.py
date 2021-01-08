"""
Plot Supplementary Figure S01 for a subject
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

# get the burst win
burst_win_ms = [15,35]
burst_win = np.round(np.array(burst_win_ms)/1000.*srate
        ).astype(int)
burst_t = (np.arange(burst_win[0], burst_win[1], 1)/
        float(srate)*1000)
burst_sigma = meet.epochEEG(stim_data_sigma, marker,
        burst_win)

# get the noise win
noise_win_ms = [-35,-15]
noise_win = np.round(np.array(noise_win_ms)/1000.*srate
        ).astype(int)
noise_t = (np.arange(noise_win[0], noise_win[1], 1)/
        float(srate)*1000)
noise_sigma = meet.epochEEG(stim_data_sigma, marker,
        noise_win)

burst_rms = np.sqrt((burst_sigma**2).mean(0))
noise_rms = np.sqrt((noise_sigma**2).mean(0))

hist_xmax = 30
hist_ymax = 0.3

hist_bins = np.linspace(0, hist_xmax, 61)
burst_50perc_score = scipy.stats.scoreatpercentile(burst_rms, 50)
noise_perc_at_burst50 = scipy.stats.percentileofscore(noise_rms,
        burst_50perc_score)
noise_50perc_score = scipy.stats.scoreatpercentile(noise_rms, 50)

noise_label = 'noise window (-35 to -15 ms)'
burst_label = 'signal window (15 to 35 ms)'

fig = plt.figure(figsize=(7,1.8))
#####################################
### plot the standard histogram ###
#####################################
hist_ax = fig.add_subplot(121)

nh = hist_ax.hist(noise_rms, bins=hist_bins, alpha=1,
        label=noise_label, cumulative=False, density=True,
        facecolor=color2, edgecolor='k', lw=0.5,
        histtype='stepfilled')
bh = hist_ax.hist(burst_rms, bins=hist_bins,
        label=burst_label, cumulative=False, density=True,
        facecolor=mpl.colors.to_rgba(color1, alpha=0.9), edgecolor='k',
        lw=0.5, histtype='stepfilled')


hist_ax.set_xlabel('rms (fT)')
hist_ax.set_ylabel('proportion')
hist_ax.set_xlim(left=0, right=hist_xmax)
hist_ax.set_ylim( (0, hist_ymax) )

#####################################
### plot the cumulative histogram ###
#####################################
cum_ax = fig.add_subplot(122, sharex = hist_ax)
cum_ax.hist(noise_rms, bins=hist_bins, alpha=1,
        label=noise_label, cumulative=True, density=True,
        facecolor=color2, edgecolor='k', lw=0.5,
        histtype='stepfilled')
cum_ax.hist(burst_rms, bins=hist_bins,
        label=burst_label, cumulative=True, density=True,
        facecolor=mpl.colors.to_rgba(color1, alpha=0.9), edgecolor='k',
        lw=0.5, histtype='stepfilled')

cum_ax.plot([0, burst_50perc_score], [0.5, 0.5], 'k-',
        lw=0.5)
cum_ax.plot(
        [burst_50perc_score, burst_50perc_score],
        [0.5, noise_perc_at_burst50/100], 'w:',
        lw=0.5)
cum_ax.plot(
        [0, burst_50perc_score],
        [noise_perc_at_burst50/100, noise_perc_at_burst50/100], 'k-',
        lw=0.5)

cum_ax.text(0.02*hist_xmax, 0.5 - 0.05, "signal 50\%",
        ha='left', va='top', fontsize=7)

if noise_perc_at_burst50 < 80:
    cum_ax.text(0.02*hist_xmax, noise_perc_at_burst50/100 + 0.05,
            "noise {:.1f}\%".format(noise_perc_at_burst50),
            ha='left', va='bottom', fontsize=7)
else:
    cum_ax.text(0.02*hist_xmax, noise_perc_at_burst50/100 - 0.05,
            "noise {:.1f}\%".format(noise_perc_at_burst50),
            ha='left', va='top', fontsize=7)

# plot the histogram of overlaps
cum_ax.set_xlabel('rms (fT)')
cum_ax.set_ylabel('proportion')
cum_ax.set_xlim(left=0, right=hist_xmax)


fig.tight_layout(rect = (0,0,1,1), pad=0.3)
fig.savefig(path.join(results_folder, 'FigureS01_{}.pdf'.format(
    subject)), format='pdf')
fig.savefig(path.join(results_folder, 'FigureS01_{}.png'.format(
    subject)), format='pdf')

leg_fig = plt.figure(figsize=(7,0.3))

leg_fig.legend(
        (nh[2][0], bh[2][0]),
        (noise_label, burst_label),
        loc='lower center',
        fontsize=7,
        ncol=2,
        columnspacing = 5)
leg_fig.savefig(path.join(results_folder, 'FigureS01_{}_legend.pdf'.format(
    subject)), format='pdf')
leg_fig.savefig(path.join(results_folder, 'FigureS01_{}_legend.png'.format(
    subject)), format='png')
