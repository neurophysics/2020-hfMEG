"""
Plot Figure 02 for a subject
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
    stim_data_hp = meet.iir.butterworth(
            stim_data, fs=(0.1, 6000), fp=(0.5, 5000),
            s_rate=srate)

# apply 450Hz-750 Hz band-pass filter
stim_data_sigma = meet.iir.butterworth(stim_data, fs=(400, 800), fp=(450, 750),
        s_rate=srate)

# get the trials
if subject.endswith('_old'):
    trial_win_ms = [-10, 60]
else:
    trial_win_ms = [-60, 160]

trial_win = np.round(np.array(trial_win_ms)/1000.*srate
        ).astype(int)
trial_t = (np.arange(trial_win[0], trial_win[1], 1)/
        float(srate)*1000)

# for high-pass
trials_hp = meet.epochEEG(stim_data_hp, marker, trial_win)
# for sigma
trials_sigma = meet.epochEEG(stim_data_sigma, marker, trial_win)

# create a custom sampling scheme for the S transform
def custom_sampling_meg(N):
    if subject.endswith('_old'):
        S_frange = [5, 2500]
        S_fnum = 25
    else:
        S_frange = [5, 5000]
        S_fnum = 30
    S_Nperperiod = 4
    wanted_freqs = np.exp(np.linspace(np.log(S_frange[0]),
        np.log(S_frange[1]), S_fnum))
    fftfreqs = np.fft.fftfreq(N, d=1./srate)
    # find the nearest frequency indices
    y = np.unique([np.argmin((w - fftfreqs)**2)
        for w in wanted_freqs])
    x = ((S_Nperperiod*fftfreqs[y]*N/float(srate))//2).astype(int)
    return x,y

#calculate the S-transforms
coords, tf = meet.tf.gft(trials_hp, axis=0, sampling=custom_sampling_meg)

# get reference trials to normalize the s transform
if subject.endswith('_old'):
    ref_win_ms = [-80, -10]
else:
    ref_win_ms = [-230, -10]
ref_win = np.round(np.array(ref_win_ms)/1000. * srate).astype(int)
ref_t = (np.arange(ref_win[0], ref_win[1], 1) / float(srate)*1000)

# get trials for high-pass reference window
ref_hp = meet.epochEEG(stim_data_hp, marker, ref_win)

# calculate the S-transforms of a pre-stimulus reference
ref_coords, ref_tf = meet.tf.gft(ref_hp, axis=0, sampling=custom_sampling_meg)

# normalize the S transform of the average
avg_norm = 20*np.log10(helper_functions.normalize_Stransform(
    coords,
    np.abs(np.mean(tf, axis=0)),
    np.abs(np.mean(ref_tf, axis=0))))
# interpolate onto a regular grid
avg_norm_interp = meet.tf.interpolate_gft(
        coords, avg_norm,
        IM_shape=(len(trial_t)//2, len(trial_t)),
        data_len=len(trial_t),
        kindf='linear', kindt='linear')[-1]

# normalize the mean of the S transforms
ampavg_norm = 20*np.log10(helper_functions.normalize_Stransform(
    coords,
    np.mean(np.abs(tf), axis=0),
    np.mean(np.abs(ref_tf), axis=0)))
# interpolate the S transforms onto a regular grid
ampavg_norm_interp = meet.tf.interpolate_gft(
        coords, ampavg_norm,
        IM_shape=(len(trial_t)//2, len(trial_t)),
        data_len=len(trial_t),
        kindf='linear',
        kindt='linear')[-1]

# normalize the variance of the S transforms
var_norm = 20*np.log10(helper_functions.normalize_Stransform(
    coords,
    np.var(np.abs(tf), axis=0),
    np.var(np.abs(ref_tf), axis=0)
    ))
# interpolate the S transforms onto a regular grid
var_norm_interp = meet.tf.interpolate_gft(
        coords, var_norm,
        IM_shape=(len(trial_t)//2, len(trial_t)),
        data_len=len(trial_t),
        kindf='linear',
        kindt='linear')[-1]

# get the frequency arrays
f = np.linspace(0, srate/2., len(trial_t)//2)

tf_f, tf_t = coords

test_mask = np.all([
    tf_t > tf_t.min() + 0.1*tf_t.ptp(),
    tf_t < tf_t.max() - 0.1*tf_t.ptp(),
    np.any([
        trial_t[tf_t.astype(int)] > 5,
        trial_t[tf_t.astype(int)] < -5,
        ], 0)
    ], 0)

############### Run the statistical test between pre-post stim #############
############################################################################
def jackknife_variance(x):
    Nid, Ntr = x.shape
    means = (np.sum(x, 1)[:,np.newaxis] - x)/(Ntr - 1)
    return (np.sum(x**2, 1)[:,np.newaxis] - x**2)/(Ntr-1) - means**2

def jackknife_avg(x):
    Nid, Ntr = x.shape
    means = np.abs((np.sum(x, 1)[:,np.newaxis] - x)/(Ntr - 1))
    return means

def avg_test_stat(coords, sig, ref):
    # Careful, we need to give trial axis as 2nd axis
    assert sig.shape == ref.shape
    N = sig.shape[1]
    sig_jackknife_avg = jackknife_avg(sig)
    ref_jackknife_avg = jackknife_avg(ref)
    avg_diff = helper_functions.difference_Stransform(
        coords, sig_jackknife_avg, ref_jackknife_avg).T
    stat = avg_diff.mean(0)/np.sqrt(avg_diff.var(0) * (N-1))
    return stat

def var_test_stat(coords, sig, ref):
    # Careful, we need to give trial axis as 2nd axis
    assert sig.shape == ref.shape
    N = sig.shape[1]
    sig_jackknife_var = jackknife_variance(sig)
    ref_jackknife_var = jackknife_variance(ref)
    var_diff = helper_functions.difference_Stransform(
        coords, sig_jackknife_var, ref_jackknife_var).T
    stat = var_diff.mean(0)/np.sqrt(var_diff.var(0) * (N-1))
    return stat

avg_stat = avg_test_stat(
        coords[:,test_mask],
        tf.T[test_mask],
        ref_tf.T[test_mask])

ampavg_stat = avg_test_stat(
        coords[:,test_mask],
        np.abs(tf).T[test_mask],
        np.abs(ref_tf).T[test_mask])

var_stat = var_test_stat(
        coords[:,test_mask],
        np.abs(tf).T[test_mask],
        np.abs(ref_tf).T[test_mask])

all_tf = np.vstack([
    tf[:,test_mask],
    ref_tf[:,test_mask],
    ])
all_var_tf = np.vstack([
    np.abs(tf[:,test_mask]) - np.abs(tf[:,test_mask]).mean(0),
    np.abs(ref_tf[:,test_mask]) - np.abs(ref_tf[:,test_mask]).mean(0)
    ])

avg_boot_stat  = []
ampavg_boot_stat  = []
var_boot_stat  = []

for _ in trange(1000):
    idx1 = np.random.choice(all_var_tf.shape[0], size=tf.shape[0],
            replace=True)
    idx2 = np.random.choice(all_var_tf.shape[0], size=ref_tf.shape[0],
            replace=True)
    avg_boot_stat.append(
            avg_test_stat(
                coords[:,test_mask],
                all_tf[idx1].T,
                all_tf[idx2].T))
    ampavg_boot_stat.append(
            avg_test_stat(
                coords[:,test_mask],
                np.abs(all_tf[idx1]).T,
                np.abs(all_tf[idx2].T)))
    var_boot_stat.append(
            var_test_stat(
                coords[:,test_mask],
                all_var_tf[idx1].T,
                all_var_tf[idx2].T))
avg_boot_stat = np.array(avg_boot_stat)
ampavg_boot_stat = np.array(ampavg_boot_stat)
var_boot_stat = np.array(var_boot_stat)

avg_p = helper_functions.stepdown_p(avg_stat, avg_boot_stat)
ampavg_p = helper_functions.stepdown_p(ampavg_stat, ampavg_boot_stat)
var_p = helper_functions.stepdown_p(var_stat, var_boot_stat)

# interpolate the p value onto a regular grid
avg_p_interp = meet.tf.interpolate_gft(
        coords[:, test_mask], avg_p,
        IM_shape=(len(trial_t)//2, len(trial_t)),
        data_len=len(trial_t),
        kindf='nearest',
        kindt='nearest')[-1]
ampavg_p_interp = meet.tf.interpolate_gft(
        coords[:, test_mask], ampavg_p,
        IM_shape=(len(trial_t)//2, len(trial_t)),
        data_len=len(trial_t),
        kindf='nearest',
        kindt='nearest')[-1]
var_p_interp = meet.tf.interpolate_gft(
        coords[:, test_mask], var_p,
        IM_shape=(len(trial_t)//2, len(trial_t)),
        data_len=len(trial_t),
        kindf='nearest',
        kindt='nearest')[-1]

########################
# plot all the results #
########################
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure(figsize=(7.48031, 4.5))
gs = mpl.gridspec.GridSpec(nrows=2, ncols=2, figure=fig,
        width_ratios=[1,1/0.92])

hp_avg_ax = fig.add_subplot(gs[0,0], frame_on=False)
hp_avg_ax.tick_params(bottom=False, labelbottom=False)
hp_avg_ax.set_ylabel('amplitude (fT)')
hp_avg_ax.plot([0,0], [0,1], 'k-', transform=hp_avg_ax.transAxes)
hp_avg_ax.text(0.05, 0.95,r'\textbf{wideband}'+'\n'+r'(0.5--5\,000 Hz)',
        ha='left', va='top', multialignment='center', size=9,
        transform=hp_avg_ax.transAxes)

sigma_avg_ax = fig.add_subplot(gs[1,0], sharex=hp_avg_ax, frame_on=False)
sigma_avg_ax.set_xlabel('time relative to stimulus (ms)')
sigma_avg_ax.set_ylabel('amplitude (fT)')
sigma_avg_ax.plot([0,0], [0,1], 'k-', transform=sigma_avg_ax.transAxes)
sigma_avg_ax.plot([0,1], [0,0], 'k-', transform=sigma_avg_ax.transAxes)
sigma_avg_ax.text(0.05, 0.95,r'\textbf{passband}'+'\n'+
        r'(450--750 Hz)',
        ha='left', va='top', multialignment='center', size=9,
        transform=sigma_avg_ax.transAxes)

hp_avg_ax.plot(trial_t, trials_hp.mean(-1), c='k', lw=1.0)
sigma_avg_ax.plot(trial_t, trials_sigma.mean(-1), c='k', lw=1.0)

hp_avg_ax.text(0.95,0.95, s=r'$\mathrm{{N={:,d} }}$'.format(
    trials_hp.shape[-1]).replace(',', r'\,'),
    ha='right', va='top', transform=hp_avg_ax.transAxes,
    size=9, color='k')

hp_avg_ax.set_xlim([-10,60])
hp_avg_ax.set_ylim([-340,565])
sigma_avg_ax.set_ylim([-23.5,23.5])

gs1 = mpl.gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1,
        subplot_spec=gs[:,1], hspace=0.4)

avg_tf_ax = fig.add_subplot(gs1[0])
avg_tf_ax.set_ylabel('frequency (Hz)')
avg_tf_ax.set_title(r'average \textbf{phase-locked} MEG response')

ampavg_tf_ax = fig.add_subplot(gs1[1], sharex=avg_tf_ax, sharey=avg_tf_ax)
ampavg_tf_ax.set_ylabel('frequency (Hz)')
ampavg_tf_ax.set_title(r'average \textbf{phase-insensitive} MEG response')

var_tf_ax = fig.add_subplot(gs1[2], sharex=avg_tf_ax, sharey=avg_tf_ax)
var_tf_ax.set_xlabel('time relative to stimulus (ms)')
var_tf_ax.set_ylabel('frequency (Hz)')
var_tf_ax.set_title(r'\textbf{amplitude-variance} of MEG response')

avg_tf = avg_tf_ax.pcolormesh(trial_t, f, avg_norm_interp,
        rasterized=True, cmap=cmap, vmin=0, vmax=40, shading='nearest')
if np.any(avg_p_interp < 0.05):
    avg_tf_ax.contour(trial_t, f, avg_p_interp, levels=[0.05],
            colors='w', alpha=1.0)
avg_tf_ax_divider = make_axes_locatable(avg_tf_ax)
avg_tf_cbax = avg_tf_ax_divider.append_axes("right", size="4%", pad="4%")
avg_tf_cb = plt.colorbar(avg_tf, cax=avg_tf_cbax, label='SNNR (dB)')
avg_tf_cb.solids.set_edgecolor('face')
avg_tf_cb.solids.set_rasterized(True)

ampavg_tf = ampavg_tf_ax.pcolormesh(trial_t, f, ampavg_norm_interp,
        rasterized=True, cmap=cmap, vmin=0, vmax=10, shading='nearest')
if np.any(ampavg_p_interp < 0.05):
    ampavg_tf_ax.contour(trial_t, f, ampavg_p_interp, levels=[0.05],
            colors='w', alpha=1.0)
ampavg_tf_ax_divider = make_axes_locatable(ampavg_tf_ax)
ampavg_tf_cbax = ampavg_tf_ax_divider.append_axes("right", size="4%", pad="4%")
ampavg_tf_cb = plt.colorbar(ampavg_tf, cax=ampavg_tf_cbax, label='SNNR (dB)')
ampavg_tf_cb.solids.set_edgecolor('face')
ampavg_tf_cb.solids.set_rasterized(True)

var_tf = var_tf_ax.pcolormesh(trial_t, f, var_norm_interp,
        rasterized=True, cmap=cmap, vmin=0, vmax=10, shading='nearest')
if np.any(var_p_interp < 0.05):
    var_tf_ax.contour(trial_t, f, var_p_interp, levels=[0.05],
            colors='w', alpha=1.0)
var_tf_ax_divider = make_axes_locatable(var_tf_ax)
var_tf_cbax = var_tf_ax_divider.append_axes("right", size="4%", pad="4%")
var_tf_cb = plt.colorbar(var_tf, cax=var_tf_cbax, label='SNNR (dB)')
var_tf_cb.solids.set_edgecolor('face')
var_tf_cb.solids.set_rasterized(True)

avg_tf_ax.set_xlim([-10,60])
avg_tf_ax.set_yscale('log')
avg_tf_ax.set_ylim([50,5000])

# make some dummy lines for a legend
N_line, = ampavg_tf_ax.plot([None, None], [None, None], visible=False)
verts =[(0,0), (0,0.3), (0.1,0.3), (0.1, 0.5), (0.1,1),
        (0.7, 1), (0.7, 0.7), (1,0.7), (1, 0.5), (0.9, 0.5), (0.9, 0.3),
        (0.8, 0.3), (0.8, 0.1), (0.3, 0.1), (0.3,0), (0, 0)]
verts = np.array(verts)*2 - 1
p_marker = ampavg_tf_ax.scatter([None], [None], c='none', edgecolors='w',
        marker=verts, s=72)

avg_tf_l = avg_tf_ax.legend([N_line, p_marker],
        [
            r'$\mathrm{{N={:,d} }}$'.format(
                trials_hp.shape[-1]).replace(',', r'\,'),
            r'$\mathrm{p<0.05}$'
            ],
    loc='upper right', fontsize=7, frameon=True, framealpha=0.4,
    facecolor='k', edgecolor='none')
[t.set_color('white') for t in avg_tf_l.get_texts()]

ampavg_tf_l = ampavg_tf_ax.legend([N_line, p_marker],
        [
            r'$\mathrm{{N={:,d} }}$'.format(
                trials_hp.shape[-1]).replace(',', r'\,'),
            r'$\mathrm{p<0.05}$'
            ],
    loc='upper right', fontsize=7, frameon=True, framealpha=0.4,
    facecolor='k', edgecolor='none')
[t.set_color('white') for t in ampavg_tf_l.get_texts()]

var_tf_l = var_tf_ax.legend([N_line, p_marker],
        [
            r'$\mathrm{{N={:,d} }}$'.format(
                trials_hp.shape[-1]).replace(',', r'\,'),
            r'$\mathrm{p<0.05}$'
            ],
    loc='upper right', fontsize=7, frameon=True, framealpha=0.4,
    facecolor='k', edgecolor='none')
[t.set_color('white') for t in var_tf_l.get_texts()]

avg_tf_ax.axvline(0, ls='-', c='k', lw=0.5)
ampavg_tf_ax.axvline(0, ls='-', c='k', lw=0.5)
var_tf_ax.axvline(0, ls='-', c='k', lw=0.5)
avg_tf_ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda x, p: format(int(x), ',').replace(',', r'\,')))
avg_tf_ax.set_yticks([50,100,500,1000, 5000])

gs.tight_layout(fig, pad=0.45, h_pad=1.5, w_pad=1.5)
fig.align_ylabels([hp_avg_ax,sigma_avg_ax])
fig.align_ylabels([avg_tf_ax,ampavg_tf_ax])
fig.canvas.draw()

dx, dy = 0, -2/72.
offset = mpl.transforms.ScaledTranslation(dx, dy,
  fig.dpi_scale_trans)
hp_avg_shift_transform = mpl.transforms.blended_transform_factory(
        hp_avg_ax.transData, hp_avg_ax.transData) + offset
sigma_avg_shift_transform = mpl.transforms.blended_transform_factory(
        sigma_avg_ax.transData, sigma_avg_ax.transData) + offset

hp_avg_ax.text(0,  trials_hp.mean(-1)[trial_t==0][0],
    r'\rotatebox[origin=c]{180}{\Lightning}', ha='right', va='top',
    size=16, transform=hp_avg_shift_transform)
sigma_avg_ax.text(0,  trials_sigma.mean(-1)[trial_t==0][0],
    r'\rotatebox[origin=c]{180}{\Lightning}', ha='right', va='top',
    size=16, transform=sigma_avg_shift_transform)

# draw lines from the highest peaks to the lf component
avg = trials_sigma.mean(-1)
idx = np.arange(len(avg))
# detect peaks in the range 12-28 ms
peak_idx = idx[1:-1][np.all([
    avg[1:-1] > avg[:-2],
    avg[1:-1]>avg[2:],
    trial_t[1:-1] > 12,
    trial_t[1:-1] < 28,
    ], 0)]
maxpeak_idx = peak_idx[np.argsort(avg[peak_idx])][-6:]
maxpeak_t = trial_t[maxpeak_idx]

[sigma_avg_ax.plot([t,t],
    [trials_sigma.mean(-1)[i],
        sigma_avg_ax.transData.inverted().transform(
            hp_avg_ax.transData.transform((t, trials_hp.mean(-1)[i]))
            )[1]], 'k--', alpha=0.2,
        clip_on=False) for t,i in zip(maxpeak_t, maxpeak_idx)]

offset = mpl.transforms.ScaledTranslation(dx, dy,
  fig.dpi_scale_trans)
offset2 = mpl.transforms.ScaledTranslation(0,
        avg_tf_ax.get_xticklabels()[1].get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin -
        avg_tf_ax.get_window_extent(
            ).transformed(fig.dpi_scale_trans.inverted()).ymin,
        fig.dpi_scale_trans)
avg_tf_shift_transform = mpl.transforms.blended_transform_factory(
        avg_tf_ax.transData, avg_tf_ax.transAxes) + offset + offset2
ampavg_tf_shift_transform = mpl.transforms.blended_transform_factory(
        ampavg_tf_ax.transData, ampavg_tf_ax.transAxes) + offset + offset2
var_tf_shift_transform = mpl.transforms.blended_transform_factory(
        var_tf_ax.transData, var_tf_ax.transAxes) + offset + offset2

var_tf_ax.text(0, 0, r'\rotatebox[origin=c]{180}{\Lightning}',
        ha='right', va='top', size=16, transform=var_tf_shift_transform)

#plot a dividing line in the center
divider_line_transform = mpl.transforms.blended_transform_factory(
        hp_avg_ax.transAxes, fig.transFigure)
hp_avg_ax.plot((1.04,1.04), (0.025,0.975), 'k-', lw=1, transform=divider_line_transform,
        clip_on=False, alpha=0.5)

hp_avg_ax.text(-0.1,1.0, r'\textbf{A}', fontsize=14,
        transform=hp_avg_ax.transAxes, ha='right', va='bottom')
avg_tf_ax.text(-0.1,1.0, r'\textbf{B}', fontsize=14,
        transform=sigma_avg_ax.transAxes, ha='right', va='bottom')
ampavg_tf_ax.text(-0.14,1.0, r'\textbf{C}', fontsize=14,
        transform=avg_tf_ax.transAxes, ha='right', va='bottom')
ampavg_tf_ax.text(-0.14,1.0, r'\textbf{D}', fontsize=14,
        transform=ampavg_tf_ax.transAxes, ha='right', va='bottom')
var_tf_ax.text(-0.14,1.0, r'\textbf{E}', fontsize=14,
        transform=var_tf_ax.transAxes, ha='right', va='bottom')

fig.savefig(path.join(results_folder,
    'Figure02_{}.pdf'.format(subject)))
fig.savefig(path.join(results_folder,
    'Figure02_{}.png'.format(subject)))
