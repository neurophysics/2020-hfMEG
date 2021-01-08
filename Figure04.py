"""
Plot Figure 04 for a subject
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

# remove the marker channel
stim_data = stim_data[0]

# interpolate the stimulus
interpolate_win_ms = [-2, 2]
interpolate_win = np.round(np.array(interpolate_win_ms)
        / 1000. * srate).astype(int)
stim_data = meet.interpolateEEG(stim_data, marker, interpolate_win)

# wideband filter
stim_data_hp = meet.stim_data = meet.iir.butterworth(stim_data,
        fs=(20, 250), fp=(30, 200), s_rate=srate)
# apply 450Hz-750 Hz band-pass filter
stim_data_sigma = meet.stim_data = meet.iir.butterworth(stim_data,
        fs=(400,800), fp=(450,750), s_rate=srate)

# get the trials
trial_win_ms = [-60,160]
trial_win = np.round(np.array(trial_win_ms)/1000.*srate
        ).astype(int)
trial_t = (np.arange(trial_win[0], trial_win[1], 1)/
        float(srate)*1000)

marker = marker[
        np.all([
    (marker+trial_win[0])>=0,
    (marker+trial_win[1])<stim_data_sigma.shape[-1]
    ], 0)
        ]

trials_hp = meet.epochEEG(stim_data_hp, marker,
        trial_win)
#remove the mean in the time window -30 to 0
trials_hp -= trials_hp[(trial_t>=-30) & (trial_t<0)].mean(0)

trials_sigma = meet.epochEEG(stim_data_sigma, marker,
        trial_win)

# get the temporal windows for the burst and periburst intervals
peristim_win1_ms = [5, 15]
peristim_win2_ms = [30, 40]
stim_win_ms = [15, 30]

peristim_win1 = np.round(np.array(peristim_win1_ms)/1000.*srate
        ).astype(int)
peristim_win2 = np.round(np.array(peristim_win2_ms)/1000.*srate
        ).astype(int)
stim_win = np.round(np.array(stim_win_ms)/1000.*srate
        ).astype(int)

# get the rms and outlier threshold
peristim1_sigma = meet.epochEEG(stim_data_sigma, marker,
        peristim_win1)
peristim2_sigma = meet.epochEEG(stim_data_sigma, marker,
        peristim_win2)
stim_sigma = meet.epochEEG(stim_data_sigma, marker,
        stim_win)
peristim_rms = 0.5*(np.sqrt(np.mean(peristim1_sigma**2, 0)) + 
        np.sqrt(np.mean(peristim2_sigma**2, 0)))
stim_rms = np.sqrt(np.mean(stim_sigma**2, 0))

# get the outlier thresholds
peristim_rms_q1 = scipy.stats.scoreatpercentile(peristim_rms, 25)
peristim_rms_q3 = scipy.stats.scoreatpercentile(peristim_rms, 75)
peristim_rms_iqr = peristim_rms_q3 - peristim_rms_q1
stim_rms_q1 = scipy.stats.scoreatpercentile(stim_rms, 25)
stim_rms_q3 = scipy.stats.scoreatpercentile(stim_rms, 75)
stim_rms_iqr = stim_rms_q3 - stim_rms_q1

#calculate correlation of stim sigma to the average
stim_sigma_avg = stim_sigma.mean(-1)
stim_sigma_avg_corr = np.mean(
        (stim_sigma - stim_sigma.mean(0))/stim_sigma.std(0)*
        ((stim_sigma_avg - stim_sigma_avg.mean(0))/stim_sigma_avg.std(0))[
                :,np.newaxis],
        0)

inlier_mask = (
        (peristim_rms>(peristim_rms_q1 - 1.5*peristim_rms_iqr)) &
        (peristim_rms<(peristim_rms_q3 + 1.5*peristim_rms_iqr)) &
        (stim_rms>(stim_rms_q1 - 1.5*stim_rms_iqr)) &
        (stim_rms<(stim_rms_q3 + 1.5*stim_rms_iqr)) &
        (stim_sigma_avg_corr > 0.0)
        )

stim_order = np.argsort(stim_rms[inlier_mask])

st_percentile_steps = np.array([0.05, 0.95]) # for single-trial plotting
avg_percentile_steps = np.array([0.05,0.2,0.4,0.6,0.8, 0.95]) # for avg plotting

st_plot_trial_idx = (inlier_mask.sum()*st_percentile_steps).astype(int)
avg_plot_trial_idx = (inlier_mask.sum()*avg_percentile_steps).astype(int)
plot_trial_num = 50

# calculate, whether the average between the 5th and the 95th percentile 
# are different
sigma_test_win_ms = [15,30]
hp_test_win_ms = [15,30]
N_bootstrap = 50000

l_trials =  np.arange(trials_sigma.shape[-1])[inlier_mask][stim_order
        [st_plot_trial_idx[0]:st_plot_trial_idx[0]+plot_trial_num]]
h_trials =  np.arange(trials_sigma.shape[-1])[inlier_mask][stim_order
        [st_plot_trial_idx[1]:st_plot_trial_idx[1]+plot_trial_num]]
c_trials = np.r_[l_trials, h_trials]

test_trials_sigma = trials_sigma[
        (trial_t>=sigma_test_win_ms[0]) &
        (trial_t<sigma_test_win_ms[1])]
test_trials_hp = trials_hp[
        (trial_t>=hp_test_win_ms[0]) &
        (trial_t<hp_test_win_ms[1])]

def compound_welch(x1, x2):
    s = np.sqrt(x1.var(-1)/x1.shape[-1] + x2.var(-1)/x2.shape[-1])
    t = (x1.mean(-1) - x2.mean(-1))/s
    return np.sum(np.abs(t))

sigma_test_t = compound_welch(
        test_trials_sigma[:,l_trials],
        test_trials_sigma[:,h_trials])
hp_test_t = compound_welch(
        test_trials_hp[:,l_trials],
        test_trials_hp[:,h_trials])

sigma_test_t_boot = []
hp_test_t_boot = []

for _ in trange(N_bootstrap):
    c_trials_boot = c_trials[np.random.rand(2*plot_trial_num).argsort()]
    l_trials_boot = c_trials_boot[:plot_trial_num]
    h_trials_boot = c_trials_boot[-plot_trial_num:]
    sigma_test_t_boot.append(compound_welch(
            test_trials_sigma[:,l_trials_boot],
            test_trials_sigma[:,h_trials_boot]))
    hp_test_t_boot.append(compound_welch(
            test_trials_hp[:,l_trials_boot],
            test_trials_hp[:,h_trials_boot]))

sigma_test_t_boot = np.asarray(sigma_test_t_boot)
hp_test_t_boot = np.asarray(hp_test_t_boot)

sigma_test_p = ((sigma_test_t_boot>=sigma_test_t).sum() + 1.)/(
        N_bootstrap + 1.)
hp_test_p = ((hp_test_t_boot>=hp_test_t).sum() + 1.)/(
        N_bootstrap + 1.)

# plot the results
fig = plt.figure(figsize=(3.54331, 6.5))
gs = mpl.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1,1.1])

st_gs = mpl.gridspec.GridSpecFromSubplotSpec(subplot_spec=gs[0], nrows=3,
        ncols=1, height_ratios=[0.13,0.4,0.4])

q = (0.4*1)/(0.93*1.1)

avg_gs = mpl.gridspec.GridSpecFromSubplotSpec(subplot_spec=gs[1], nrows=3,
        ncols=1, height_ratios=[1-2*q, q, q])

st_title_ax = fig.add_subplot(st_gs[0], frame_on=False)
st_title_ax.tick_params(**blind_ax)

hf_st_ax1 = fig.add_subplot(st_gs[1])
hf_st_ax2 = fig.add_subplot(st_gs[2], sharex=hf_st_ax1, sharey=hf_st_ax1)
lf_st_ax1 = plt.twinx(hf_st_ax1)
lf_st_ax2 = plt.twinx(hf_st_ax2)
lf_st_ax1.get_shared_y_axes().join(lf_st_ax1, lf_st_ax2)

hf_st_ax1.plot(trial_t,  trials_sigma[:,inlier_mask][:,stim_order[
                st_plot_trial_idx[0]:st_plot_trial_idx[0]+plot_trial_num]],
            alpha=0.1, c=color1, lw=1)
hf_st_ax2.plot(trial_t,  trials_sigma[:,inlier_mask][:,stim_order[
                st_plot_trial_idx[1]:st_plot_trial_idx[1]+plot_trial_num]],
            alpha=0.1, c=color1, lw=1)
lf_st_ax1.plot(trial_t,  trials_hp[:,inlier_mask][:,stim_order[
                st_plot_trial_idx[0]:st_plot_trial_idx[0]+plot_trial_num]],
            alpha=0.5, c=color2, lw=1)
lf_st_ax2.plot(trial_t,  trials_hp[:,inlier_mask][:,stim_order[
                st_plot_trial_idx[1]:st_plot_trial_idx[1]+plot_trial_num]],
            alpha=0.5, c=color2, lw=1)

hf_st_ax1.set_zorder(lf_st_ax1.get_zorder()+1)
hf_st_ax1.patch.set_visible(False)
hf_st_ax2.set_zorder(lf_st_ax2.get_zorder()+1)
hf_st_ax2.patch.set_visible(False)

hf_st_ax1.spines['left'].set_edgecolor(color1)
hf_st_ax2.spines['left'].set_edgecolor(color1)
hf_st_ax1.spines['right'].set_edgecolor(color2)
hf_st_ax2.spines['right'].set_edgecolor(color2)

[t.set_color(color1) for t in hf_st_ax1.yaxis.get_ticklines()]
[t.set_color(color1) for t in hf_st_ax2.yaxis.get_ticklines()]
[t.set_color(color1) for t in hf_st_ax1.yaxis.get_ticklabels()]
[t.set_color(color1) for t in hf_st_ax2.yaxis.get_ticklabels()]
[t.set_color(color2) for t in lf_st_ax1.yaxis.get_ticklines()]
[t.set_color(color2) for t in lf_st_ax2.yaxis.get_ticklines()]
[t.set_color(color2) for t in lf_st_ax1.yaxis.get_ticklabels()]
[t.set_color(color2) for t in lf_st_ax2.yaxis.get_ticklabels()]

#hf_st_ax1.set_xlabel('time relative to stimulus (ms)')
hf_st_ax1.tick_params(labelbottom=False)
hf_st_ax2.set_xlabel('time relative to stimulus (ms)')
hf_st_ax1.set_ylabel('ampl. (fT)', color=color1)
lf_st_ax1.set_ylabel('ampl. (fT)', color=color2)
hf_st_ax2.set_ylabel('ampl. (fT)', color=color1)
lf_st_ax2.set_ylabel('ampl. (fT)', color=color2)

hf_st_ax1.text(0.05, 0.95, r'\textbf{5th} percentile', ha='left', va='top',
        size=7, transform=hf_st_ax1.transAxes)
hf_st_ax2.text(0.05, 0.95, r'\textbf{95th} percentile', ha='left', va='top',
        size=7, transform=hf_st_ax2.transAxes)

#create some fake lines for the legend
st_burst_line, = hf_st_ax1.plot(np.NaN, np.NaN, alpha=1.0, c=color1, lw=1)
st_N20_line, = hf_st_ax1.plot(np.NaN, np.NaN, alpha=1.0, c=color2, lw=1)

lg_st = fig.legend(
        (st_burst_line, st_N20_line),
        (r'\textbf{450-750 Hz}', r'\textbf{30-200 Hz}'),
        loc='upper center', fontsize=7,
        title= r'\textbf{single trials} (\textrm{$N=50$}),'+'\n'+
        r'selected according to rms of hfSERs',
        frameon=False, ncol=2)
lg_st.get_title().set_fontsize(10)
lg_st.get_title().set_ma('center')

avg_title_ax = fig.add_subplot(avg_gs[0], frame_on=False)
avg_title_ax.tick_params(**blind_ax)

hf_avg_ax = fig.add_subplot(avg_gs[1], sharex=hf_st_ax1)
lf_avg_ax = fig.add_subplot(avg_gs[2], sharex=hf_st_ax1)

cmap_inst = mpl.cm.get_cmap(cmap+'_r')
burst_lines = []

for i, idx in enumerate(avg_plot_trial_idx[::-1]):
    burst_lines.extend(
            hf_avg_ax.plot(trial_t,
                np.mean(
                    trials_sigma[:,inlier_mask][
                        :,stim_order[idx:idx+50]],
                    axis=-1),
                c=cmap_inst(avg_percentile_steps[i]), lw=1)
            )
    lf_avg_ax.plot(trial_t,
            np.mean(
                trials_hp[:,inlier_mask][:,stim_order[idx:idx+50]],
                axis=-1),
            c=cmap_inst(avg_percentile_steps[i]), lw=1)
hf_st_ax1.axhline(0, c='k', lw=0.5)
hf_st_ax2.axhline(0, c='k', lw=0.5)
hf_avg_ax.axhline(0, c='k', lw=0.5)
lf_avg_ax.axhline(0, c='k', lw=0.5)
hf_avg_ax.tick_params(labelbottom=False)
hf_avg_ax.set_ylabel('ampl. (fT)')
lf_avg_ax.set_xlabel('time relative to stimulus (ms)')
lf_avg_ax.set_ylabel('ampl. (fT)')


hf_avg_ax.text(0.05, 0.95, r'\textbf{450-750 Hz}', ha='left', va='top',
        size=7, transform=hf_avg_ax.transAxes)
lf_avg_ax.text(0.05, 0.95, r'\textbf{30-200 Hz}', ha='left', va='top',
        size=7, transform=lf_avg_ax.transAxes)

hf_st_ax1.set_xlim([-10,60])

lg_avg = fig.legend(
        burst_lines[::-1],
        [(r'\textbf{%dth} percentile' % (100*p)) for p in avg_percentile_steps],
        loc='upper center', fontsize=7,
        title= r'\textbf{subaverages} (\textrm{$N=50$}),'+'\n' +
        r'selected according to rms of hfSERs',
        frameon=False, ncol=3,
        columnspacing=1,
        bbox_to_anchor=(0.5,0.51))
lg_avg.get_title().set_fontsize(10)
lg_avg.get_title().set_ma('center')

#symmetrize all the axes
[helper_functions.symmetrize_y(ax)
        for ax in [
            hf_st_ax1,
            hf_st_ax2,
            lf_st_ax1,
            lf_st_ax2,
            hf_avg_ax,
            lf_avg_ax]]

hf_st_ax1.set_ylim([-60,60])
lf_st_ax1.set_ylim([-850,850])

hf_avg_ax.set_ylim([-45,45])
lf_avg_ax.set_ylim([-500,500])

hf_avg_ax.axvspan(
        sigma_test_win_ms[0], sigma_test_win_ms[1], color='k', alpha=0.1)
hf_avg_ax.text(0.98,0.95,
        r'\textbf{5th $\neq$ 95th}:''\n'+r'p={:.3f}'.format(sigma_test_p),
        transform=hf_avg_ax.transAxes, ha='right', va='top', size=7,
        multialignment='center',
        bbox=dict(alpha=0.1))
lf_avg_ax.axvspan(
        hp_test_win_ms[0], hp_test_win_ms[1], color='k', alpha=0.1)
lf_avg_ax.text(0.98,0.95,
        r'\textbf{5th $\neq$ 95th}:''\n'+r'p={:.3f}'.format(hp_test_p),
        transform=lf_avg_ax.transAxes, ha='right', va='top', size=7,
        multialignment='center',
        bbox=dict(alpha=0.1))

gs.tight_layout(fig, pad=0.2, h_pad=0.)
fig.align_ylabels()

fig.canvas.draw()
hf_st_ax1.text(0, 0.1*hf_st_ax1.get_ylim()[0],
        r'\rotatebox[origin=c]{180}{\Lightning}', ha='right', va='top',
        size=20, transform=hf_st_ax1.transData)
hf_st_ax2.text(0, 0.1*hf_st_ax2.get_ylim()[0],
        r'\rotatebox[origin=c]{180}{\Lightning}', ha='right', va='top',
        size=20, transform=hf_st_ax2.transData)
hf_avg_ax.text(0, 0.1*hf_avg_ax.get_ylim()[0],
        r'\rotatebox[origin=c]{180}{\Lightning}', ha='right', va='top',
        size=20, transform=hf_avg_ax.transData)
lf_avg_ax.text(0, 0.1*lf_avg_ax.get_ylim()[0],
        r'\rotatebox[origin=c]{180}{\Lightning}', ha='right', va='top',
        size=20, transform=lf_avg_ax.transData)

fig.savefig(os.path.join(results_folder, 'Figure04_{}.pdf'.format(subject)))
fig.savefig(os.path.join(results_folder, 'Figure04_{}.png'.format(subject)))
