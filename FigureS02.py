"""
Plot Supplementary Figure S02 for a subject
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

# apply a 0.5 Hz high-pass filter
if subject.endswith('_old'):
    stim_data_hp = meet.stim_data = meet.iir.butterworth(
            stim_data, fs=(0.1, 2500), fp=(0.5, 2000),
            s_rate=srate)
else:
    stim_data_hp = meet.stim_data = meet.iir.butterworth(
            stim_data, fs=(0.1, 6000), fp=(0.5, 5000),
            s_rate=srate)

# apply 450Hz-750 Hz band-pass filter
stim_data_sigma = meet.stim_data = meet.iir.butterworth(stim_data,
        fs=(400,800), fp=(450,750), s_rate=srate)

# calculate Hilbert transform
stim_data_sigma_hilbert = scipy.signal.hilbert(stim_data_sigma)

# get the trials
trial_win_ms = [-60,160]
trial_win = np.round(np.array(trial_win_ms)/1000.*srate
        ).astype(int)
trial_t = (np.arange(trial_win[0], trial_win[1], 1)/
        float(srate)*1000)

# for high-pass
trials_sigma = meet.epochEEG(stim_data_sigma_hilbert, marker, trial_win)

burst_mask = np.all([trial_t>=15, trial_t<=30], 0)
noise_mask = np.all([trial_t>=55, trial_t<=70], 0)

# remove outlier trials
burst_rms = np.sqrt(np.mean(trials_sigma.real[burst_mask]**2, 0))
burst_rms_q25 = scipy.stats.scoreatpercentile(burst_rms, 25)
burst_rms_q50 = np.median(burst_rms)
burst_rms_q75 = scipy.stats.scoreatpercentile(burst_rms, 75)
burst_iqr = burst_rms_q75 - burst_rms_q25

noise_rms = np.sqrt(np.mean(trials_sigma.real[noise_mask]**2, 0))
noise_rms_q25 = scipy.stats.scoreatpercentile(noise_rms, 25)
noise_rms_q50 = np.median(noise_rms)
noise_rms_q75 = scipy.stats.scoreatpercentile(noise_rms, 75)
noise_iqr = noise_rms_q75 - noise_rms_q25

inlier_trials = np.all([
    burst_rms > (burst_rms_q50 - 1.5 * burst_iqr),
    burst_rms < (burst_rms_q50 + 1.5 * burst_iqr),
    noise_rms > (noise_rms_q50 - 1.5 * noise_iqr),
    noise_rms < (noise_rms_q50 + 1.5 * noise_iqr)],0)

trials_sigma = trials_sigma[:,inlier_trials]

plot_max=80
scatter_cmap_inst = mpl.cm.get_cmap('hsv')
gradient = np.linspace(0,1,256)
gradient = np.vstack([gradient, gradient])

burst_order = np.random.choice(
        np.size(trials_sigma[burst_mask]),
        np.size(trials_sigma[burst_mask]),
        replace=False)
noise_order = np.random.choice(
        np.size(trials_sigma[noise_mask]),
        np.size(trials_sigma[noise_mask]),
        replace=False)

period_length = 1000/600

####################
# plot the results #
####################

fig = plt.figure(figsize=(5.51181, 5.5))
gs = mpl.gridspec.GridSpec(nrows=2, ncols=1,
        figure=fig, height_ratios=(1,3))

burst_ax = fig.add_subplot(gs[0,:], frameon=False)
burst_ax.plot(trial_t, trials_sigma.real.mean(-1), 'k-')
burst_ax.plot([0,0], [0,1], 'k-', transform=burst_ax.transAxes)
burst_ax.plot([0,1], [0,0], 'k-', transform=burst_ax.transAxes)
burst_ax.text(0.01, 0.95, r'\textbf{band-pass}' + '\n' + r'(450--750 Hz)',
        ha='left', va='top', multialignment='center',
        transform=burst_ax.transAxes, fontsize=9)
burst_ax.set_xlim([0,80])
burst_ax.set_ylim([-23.5, 23.5])
burst_ax.set_xlabel('time relative to stimulus (ms)')
burst_ax.set_ylabel('ampl. (fT)')

burst_ax.axvspan(15,30, color=color1, alpha=0.4)
burst_ax.axvspan(55,70, color=color2, alpha=0.4)

gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2,2,gs[1,:],
        height_ratios=(1,0.05), hspace=0.25, wspace=0.3)
burst_polar_ax = fig.add_subplot(gs1[0,0], polar=True)
burst_polar_ax.scatter(
        np.angle(trials_sigma[burst_mask]).ravel()[burst_order],
        np.abs(trials_sigma[burst_mask]).ravel()[burst_order],
        c = scatter_cmap_inst(((
            np.ones(trials_sigma[burst_mask].shape)*
            trial_t[burst_mask][:,np.newaxis]).ravel()[burst_order])
            /(1000/600) % 1), alpha=0.4, edgecolors='none', s=10,
        rasterized=True)

burst_polar_ax.set_title(r'\textbf{ampl.} (fT) \textbf{and phase} (rad)' +
        '\n' + r'\textbf{during burst} (15 - 30 ms)',
        fontsize=12, multialignment='center', color=color1)
burst_polar_ax.set_xticks(np.linspace(0,2*np.pi,4, endpoint=False))
burst_polar_ax.set_xticklabels([r'$0$', r'$\pi / 2$', r'$\pi$',
        r'$3\pi /2$'])
plt.setp(burst_polar_ax.spines.values(), color=color1)
plt.setp(burst_polar_ax.spines.values(), linewidth=2)

noise_polar_ax = fig.add_subplot(gs1[0,1], polar=True, sharey =burst_polar_ax)
noise_polar_ax.scatter(
        np.angle(trials_sigma[noise_mask]).ravel()[noise_order],
        np.abs(trials_sigma[noise_mask]).ravel()[noise_order],
        c = scatter_cmap_inst(((
            np.ones(trials_sigma[noise_mask].shape)*
            trial_t[noise_mask][:,np.newaxis]).ravel()[noise_order])
            /(1000/600) % 1), alpha=0.4, edgecolors='none', s=10,
        rasterized=True)

noise_polar_ax.set_title(r'\textbf{ampl.} (fT) \textbf{and phase} (rad)' + '\n' + r'\textbf{during noise} (55 - 70 ms)', fontsize=12, multialignment='center',
        color=color2)
noise_polar_ax.set_xticks(np.linspace(0,2*np.pi,4, endpoint=False))
noise_polar_ax.set_xticklabels([r'$0$', r'$\pi / 2$', r'$\pi$',
        r'$3\pi /2$'])

plt.setp(noise_polar_ax.spines.values(), color=color2)
plt.setp(noise_polar_ax.spines.values(), linewidth=2)

# Create offset transform by 2 points in x and 4 in y direction
label_dx = 2/72.; label_dy = 4/72. 
label_offset = mpl.transforms.ScaledTranslation(label_dx, label_dy,
        fig.dpi_scale_trans)

burst_polar_ax.set_rlim([0,plot_max])
burst_polar_ax.set_rticks([20,40, 60])
burst_polar_ax.set_rlabel_position(0)
noise_polar_ax.set_rlabel_position(0)

[l.set_linewidth(1) for l in burst_polar_ax.yaxis.get_gridlines()]
[l.set_color('0.9') for l in burst_polar_ax.yaxis.get_gridlines()]
[l.set_zorder(10) for l in burst_polar_ax.yaxis.get_gridlines()]
[l.set_linewidth(1) for l in burst_polar_ax.xaxis.get_gridlines()]
[l.set_color('0.9') for l in burst_polar_ax.xaxis.get_gridlines()]
[l.set_zorder(10) for l in burst_polar_ax.xaxis.get_gridlines()]
[l.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.6,
    boxstyle='circle')) for l in burst_polar_ax.yaxis.get_ticklabels()]
[l.set_transform(l.get_transform() + label_offset)
        for l in burst_polar_ax.yaxis.get_ticklabels()]

[l.set_linewidth(1) for l in noise_polar_ax.yaxis.get_gridlines()]
[l.set_color('0.9') for l in noise_polar_ax.yaxis.get_gridlines()]
[l.set_zorder(10) for l in noise_polar_ax.yaxis.get_gridlines()]
[l.set_linewidth(1) for l in noise_polar_ax.xaxis.get_gridlines()]
[l.set_color('0.9') for l in noise_polar_ax.xaxis.get_gridlines()]
[l.set_zorder(10) for l in noise_polar_ax.xaxis.get_gridlines()]
[l.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.6,
    boxstyle='circle')) for l in noise_polar_ax.yaxis.get_ticklabels()]
[l.set_transform(l.get_transform() + label_offset)
        for l in noise_polar_ax.yaxis.get_ticklabels()]

cbar_ax = fig.add_subplot(gs1[1,:])
cbar_ax.imshow(scatter_cmap_inst(gradient), aspect='auto')
cbar_ax.tick_params(left=False, labelleft=False, right=False, labelright=False,
        top=False, labeltop=False, bottom=True, labelbottom=True)
cbar_ax.set_title(r'\textbf{phase} (rad) \textbf{of 600 Hz}', fontsize=12)

cbar_ax.set_xticks(np.linspace(0,256, 5))
cbar_ax.set_xticklabels([r'$0$', r'$\pi / 2$', r'$\pi$',
        r'$3\pi /2$', r'$2\pi$'])

gs.tight_layout(fig,pad=0.45, h_pad=1.5, w_pad=1.5)

fig.canvas.draw()
dx, dy = 0, -2/72.
offset = mpl.transforms.ScaledTranslation(dx, dy,
  fig.dpi_scale_trans)
offset2 = mpl.transforms.ScaledTranslation(0,
        burst_ax.get_xticklabels()[1].get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin -
        burst_ax.get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin,
        fig.dpi_scale_trans)
burst_shift_transform = mpl.transforms.blended_transform_factory(
        burst_ax.transData, burst_ax.transAxes) + offset + offset2

burst_ax.text(0, 0, r'\rotatebox[origin=c]{180}{\Lightning}',
        ha='right', va='top', size=16, transform=burst_shift_transform)

fig.savefig(os.path.join(results_folder, 'FigureS02_{}.pdf'.format(subject)))
fig.savefig(os.path.join(results_folder, 'FigureS02_{}.png'.format(subject)))

