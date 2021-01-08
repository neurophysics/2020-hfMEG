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
import autocorr_helper
from tqdm import trange, tqdm
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
trials_hp = meet.epochEEG(stim_data_hp, marker, trial_win)
trials_sigma = meet.epochEEG(stim_data_sigma_hilbert, marker, trial_win)

try:
    with np.load(path.join(results_folder, 'hilbert_acf_cacf.npz'),
            'rb') as npzfile:
        tf_pacf = npzfile['tf_pacf']
        tf_pacf_p = npzfile['tf_pacf_p']
        tf_pcacf = npzfile['tf_pcacf']
        tf_pcacf_p = npzfile['tf_pcacf_p']
        tf_pacf_time = npzfile['tf_pacf_time']
        tf_pcacf_time = npzfile['tf_pcacf_time']
        tf_pacf_time_p = npzfile['tf_pacf_time_p']
        tf_pcacf_time_p = npzfile['tf_pcacf_time_p']
except:
    tf_acf, tf_acf_boot = autocorr_helper.acf(
            np.abs(trials_sigma), nlags=51, N_bootstrap=1000)
    tf_cacf, tf_cacf_boot = autocorr_helper.cacf(
            np.angle(trials_sigma), nlags=51, N_bootstrap=1000)
    # apply smoothing along time axis
    tf_acf = autocorr_helper.smooth_acf_map(tf_acf, nlags=100,
            axis=0)
    tf_acf_boot = autocorr_helper.smooth_acf_map(tf_acf_boot,
            nlags=100, axis=1)
    tf_cacf = autocorr_helper.smooth_acf_map(tf_cacf, nlags=100,
            axis=0)
    tf_cacf_boot = autocorr_helper.smooth_acf_map(tf_cacf_boot,
            nlags=100, axis=1)
    # calculate the partial correlation coefficient
    tf_pacf = np.array([autocorr_helper.pacf(t) for t in tf_acf])
    tf_pacf_boot = np.array([
        np.array([autocorr_helper.pacf(t) for t in q])
        for q in tqdm(tf_acf_boot)])
    tf_pcacf = np.array([autocorr_helper.pacf(t) for t in tf_cacf])
    tf_pcacf_boot = np.array([
        np.array([autocorr_helper.pacf(t) for t in q])
        for q in tqdm(tf_cacf_boot)])
    # apply smoothing along lag axis
    tf_pacf = autocorr_helper.smooth_acf_map(tf_pacf,
            nlags=10, axis=-1)
    tf_pacf_boot = autocorr_helper.smooth_acf_map(tf_pacf_boot,
            nlags=10, axis=-1)
    tf_pcacf = autocorr_helper.smooth_acf_map(tf_pcacf,
            nlags=10, axis=-1)
    tf_pcacf_boot = autocorr_helper.smooth_acf_map(tf_pcacf_boot,
            nlags=10, axis=-1)
    # calculate the integrated autocorrelation time
    tf_pacf_time = 2*tf_pacf.sum(-1) - 1
    tf_pacf_time_boot = (2*tf_pacf_boot.sum(-1) - 1).T
    tf_pcacf_time = 2*tf_pcacf.sum(-1) - 1
    tf_pcacf_time_boot = (2*tf_pcacf_boot.sum(-1) - 1).T
    # calculate p values
    import stepdown_p
    tf_pacf_time_p = stepdown_p.stepdown_p(
            ((tf_pacf_time - tf_pacf_time.mean(0))/
                tf_pacf_time.std(0)),
            ((tf_pacf_time_boot - tf_pacf_time_boot.mean(0))/
            tf_pacf_time_boot.std(0)).T)
    tf_pcacf_time_p = stepdown_p.stepdown_p(
            ((tf_pcacf_time - tf_pcacf_time.mean(0))/
                tf_pcacf_time.std(0)),
            ((tf_pcacf_time_boot - tf_pcacf_time_boot.mean(0))/
            tf_pcacf_time_boot.std(0)).T)
    # at those intervals with significant partial autocorrelation,
    # check for significant lags
    tf_pacf_p = np.ones_like(tf_pacf)
    if np.any(tf_pacf_time_p < 0.05):
        tf_pacf_p[tf_pacf_time_p<0.05, 1:] = stepdown_p.stepdown_p(
                np.ravel(scipy.stats.zscore(tf_pacf, axis=0)[
                    tf_pacf_time_p<0.05, 1:]),
                np.reshape(scipy.stats.zscore(tf_pacf_boot,
                    axis=1)[:,tf_pacf_time_p<0.05, 1:],
                    [tf_pacf_boot.shape[0], -1])).reshape(
                            (tf_pacf_time_p<0.05).sum(), -1)
    tf_pcacf_p = np.ones_like(tf_pcacf)
    if np.any(tf_pcacf_time_p < 0.05):
        tf_pcacf_p[tf_pcacf_time_p<0.05, 1:] = stepdown_p.stepdown_p(
                np.ravel(scipy.stats.zscore(tf_pcacf, axis=0)[
                    tf_pcacf_time_p<0.05, 1:]),
                np.reshape(scipy.stats.zscore(tf_pcacf_boot,
                    axis=1)[:,tf_pcacf_time_p<0.05, 1:],
                    [tf_pcacf_boot.shape[0], -1])).reshape(
                            (tf_pcacf_time_p<0.05).sum(), -1)
    np.savez(path.join(results_folder, 'hilbert_acf_cacf.npz'),
        tf_pacf = tf_pacf,
        tf_pacf_p = tf_pacf_p,
        tf_pcacf = tf_pcacf,
        tf_pcacf_p = tf_pcacf_p,
        tf_pacf_time = tf_pacf_time,
        tf_pcacf_time = tf_pcacf_time,
        tf_pacf_time_p = tf_pacf_time_p,
        tf_pcacf_time_p = tf_pcacf_time_p,
            )

import matplotlib.patheffects as path_effects

# plot the results
fig = plt.figure(figsize=(3.54331, 5.7))
gs = mpl.gridspec.GridSpec(nrows=3, ncols=1, figure=fig,
        height_ratios=(1,1,1.2))

sigma_avg_ax = fig.add_subplot(gs[0,0], frame_on=False)
sigma_avg_ax.set_xlabel('time relative to stimulus (ms)')
sigma_avg_ax.set_ylabel('amplitude (fT)')
sigma_avg_ax.plot(trial_t, trials_sigma.real.mean(-1), c='k', lw=1.0)
sigma_avg_ax.plot([0,0], [0,1], 'k-', transform=sigma_avg_ax.transAxes)
sigma_avg_ax.plot([0,1], [0,0], 'k-', transform=sigma_avg_ax.transAxes)
sigma_avg_ax.text(0.05, 0.95,r'\textbf{band-pass}'+'\n'+
        r'(450--750 Hz)',
        ha='left', va='top', multialignment='center', size=9,
        transform=sigma_avg_ax.transAxes)

acf_time_ax = fig.add_subplot(gs[1,0], frame_on=False, sharex=sigma_avg_ax)
acf_time_ax.set_xlabel('time relative to stimulus (ms)')
acf_time_ax.set_ylabel('acf integral (z-score)')
acf_time_ax.plot([0,0], [0,1], 'k-', transform=acf_time_ax.transAxes)
acf_time_ax.plot([0,1], [0,0], 'k-', transform=acf_time_ax.transAxes)
acf_time_ax.plot([trial_t[0],trial_t[-1]], [0,0], 'k-',
        transform=acf_time_ax.transData, lw=0.5)

acf_time_ax.plot(trial_t,
        (tf_pacf_time - tf_pacf_time.mean())/tf_pacf_time.std(),
        c=color2, lw=1.0)
acf_time_ax.text(0.05, 0.95,r'\textbf{amplitude}'+'\n'+
        r'autocorrelation',
        ha='left', va='top', multialignment='center', size=9,
        transform=acf_time_ax.transAxes, color=color2)
if np.any(tf_pacf_time_p<0.05):
    acf_time_ax.scatter(trial_t[tf_pacf_time_p<0.05],
            np.ones(np.sum(tf_pacf_time_p<0.05), float)*4.7,
            marker='|', c=color2)

acf_time_ax.plot(trial_t,
        (tf_pcacf_time - tf_pcacf_time.mean(
            ))/tf_pcacf_time.std(),
        c=color1, lw=1.0)
acf_time_ax.text(0.95, 0.95,r'\textbf{phase}'+'\n'+
        r'autocorrelation',
        ha='right', va='top', multialignment='center', size=9,
        transform=acf_time_ax.transAxes, color=color1)
if np.any(tf_pcacf_time_p<0.05):
    acf_time_ax.scatter(trial_t[tf_pcacf_time_p<0.05],
            np.ones(np.sum(tf_pcacf_time_p<0.05), float)*5.3,
            marker='|', c=color1)

gs2 = mpl.gridspec.GridSpecFromSubplotSpec(subplot_spec = gs[2,0],
        nrows=2, ncols=1, height_ratios=[20,1], hspace=1.0)

pc_cacf_ax = fig.add_subplot(gs2[0,0], frame_on=True, sharex=sigma_avg_ax)
pc = pc_cacf_ax.pcolormesh(helper_functions.extend_by_one(trial_t), np.arange(1,52,1),
        tf_pcacf[:,1:].T, vmin=0, vmax=0.04, cmap=cmap,
        rasterized=True)
if np.any(tf_pcacf_p < 0.05):
    co = pc_cacf_ax.contour(trial_t, np.arange(1,51,1),
            tf_pcacf_p[:,1:].T, levels=[0.05], colors='w',
            linewidths=0.75)
    plt.setp(co.collections, path_effects = (
                [path_effects.Stroke(linewidth=2., foreground='black'),
                       path_effects.Normal()]))

pc_cacf_ax.axvline(0, ls='-', c='k', lw=0.5)
pc_cacf_ax.set_xlabel('time relative to stimulus (ms)')
pc_cacf_ax.set_ylabel('lag (trials)')
pc_cacf_ax.set_title(r'smoothed partial acf of phases')

cb_ax = fig.add_subplot(gs2[1,0])
plt.colorbar(pc, cax=cb_ax,
        label='partial circular correlation coefficient',
        orientation='horizontal',
        ticks=[0, 0.01, 0.02, 0.03, 0.04])


fig.align_ylabels()
sigma_avg_ax.set_xlim([-10,60])
sigma_avg_ax.set_ylim([-24,24])
acf_time_ax.set_ylim([-2, 5.7])

verts =[(0,0), (0,0.3), (0.1,0.3), (0.1, 0.5), (0.1,1),
        (0.7, 1), (0.7, 0.7), (1,0.7), (1, 0.5), (0.9, 0.5), (0.9, 0.3),
        (0.8, 0.3), (0.8, 0.1), (0.3, 0.1), (0.3,0), (0, 0)]
verts = np.array(verts)*2 - 1
p_marker = pc_cacf_ax.scatter([None], [None], c='none', edgecolors='w',
        marker=verts, s=72)

cacf_l = pc_cacf_ax.legend([p_marker],
        [
            r'$\mathrm{p<0.05}$'
            ],
    loc='upper right', fontsize=7, frameon=True, framealpha=0.4,
    facecolor='k', edgecolor='none')
[t.set_color('white') for t in cacf_l.get_texts()]

gs.tight_layout(fig, pad=0.2, h_pad=0.75)

fig.canvas.draw()

dx, dy = 0, -2/72.
offset = mpl.transforms.ScaledTranslation(dx, dy,
  fig.dpi_scale_trans)
offset2 = mpl.transforms.ScaledTranslation(0,
        pc_cacf_ax.get_xticklabels()[1].get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin -
        pc_cacf_ax.get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin,
        fig.dpi_scale_trans)
sigma_shift_transform = sigma_avg_ax.transData + offset
acf_shift_transform = acf_time_ax.transData + offset
cacf_shift_transform = mpl.transforms.blended_transform_factory(
        pc_cacf_ax.transData, pc_cacf_ax.transAxes) + offset + offset2

sigma_avg_ax.text(0,  0,
        r'\rotatebox[origin=c]{180}{\Lightning}', ha='right', va='top',
    size=16, transform=sigma_shift_transform)
acf_time_ax.text(0,  0,
        r'\rotatebox[origin=c]{180}{\Lightning}', ha='right', va='top',
    size=16, transform=acf_shift_transform)

pc_cacf_ax.text(0,  0,
    r'\rotatebox[origin=c]{180}{\Lightning}', ha='right', va='top',
    size=16, transform=cacf_shift_transform)

fig.savefig(path.join(results_folder, 'FigureS04_{}.pdf'.format(subject)))
fig.savefig(path.join(results_folder, 'FigureS04_{}.png'.format(subject)))
