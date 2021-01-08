import numpy as np
import scipy
import scipy.signal
import scipy.stats
import meet
import tqdm

def extend_by_one(x):
    diff = x[1]-x[0]
    if not np.allclose(np.diff(x), diff):
        raise ValueError('Values must be uniformly spaced')
    return np.r_[x, x[-1] + diff]

def bootstrap_SNR(x, marker, win, bootstrap_idx, N_rep=1000):
    trials = meet.epochEEG(x, marker, win)
    signal_rms = np.sqrt(scipy.stats.trim_mean(trials**2,
        proportiontocut=0.2, axis=None))
    SNNR = np.empty(N_rep, float)
    ###########################
    # bootstrap the noise rms #
    ##########################
    for i in range(N_rep):
        noise_marker = np.random.choice(bootstrap_idx, size=len(marker),
                replace=True)
        noise_trials = meet.epochEEG(x, noise_marker, win)
        noise_rms = np.sqrt(scipy.stats.trim_mean(noise_trials**2,
            proportiontocut=0.2, axis=None))
        SNNR[i] = signal_rms/noise_rms
    return SNNR

def moving_rms(x, marker, trial_win, rms_win):
    moving_rms = np.array([
        np.sqrt(scipy.stats.trim_mean(
            meet.epochEEG(x, marker + tau -
                rms_win[0] - (rms_win[1] - rms_win[0])//2, rms_win)**2,
            proportiontocut=0.2,
            axis=None))
        for tau in range(trial_win[0], trial_win[1], 1)])
    return moving_rms

def get_temp_filter(x, marker, win, idx, N_train=10000):
    trials = meet.epochEEG(x, marker, win)
    noise = meet.epochEEG(x, np.random.choice(idx, replace=True,
                size=N_train), win)
    temp_filter, temp_eigvals = meet.spatfilt.CSP(
            trials,
            noise,
            center=False)
    return temp_filter, temp_eigvals

def getBootstrapMarkerIdx(x, marker, win, clip_points=0):
    # find array of possible markers
    idx =np.arange(clip_points, len(x) - clip_points)
    idx = idx[np.all([(idx + win[0])>0, (idx + win[1]) < len(x)], axis=0)]
    win_range = win[1] - win[0]
    inwin = np.ravel([np.arange(m - win_range, m + win_range, 1)
            for m in marker])
    temp = np.setdiff1d(idx, inwin)
    return temp

def hilbert(x, axis):
    x = x.swapaxes(0, axis)
    N = x.shape[0]
    N_padded = scipy.fftpack.next_fast_len(N)
    temp = np.zeros([N_padded]+list(x.shape)[1:], x.dtype)
    temp[:N] = x
    return scipy.signal.hilbert(temp, axis=0)[:N].swapaxes(0, axis)

def normalize_Stransform(coords, sig, ref):
    result = np.zeros_like(sig)
    f, t = coords
    for f_now in np.unique(f):
        idx = np.isclose(f, f_now)
        result[idx] = sig[idx]/np.mean(ref[idx], 0)
    return result

def difference_Stransform(coords, sig, ref):
    result = np.zeros_like(sig)
    f, t = coords
    for f_now in np.unique(f):
        idx = np.isclose(f, f_now)
        result[idx] = sig[idx] - np.mean(ref[idx], 0)
    return result

def get_S_meanstd(coords, S):
    f, t = coords
    mean = np.empty(S.shape[-1], float)
    std = np.empty(S.shape[-1], float)
    for f_now in np.unique(f):
        idx = np.isclose(f, f_now)
        mean[idx] = np.abs(S[...,idx]).mean()
        std[idx] = np.abs(S[...,idx]).std()
    return mean, std

def symmetrize_y(ax):
    ylim = ax.get_ylim()
    new_ylim = np.abs(ylim).max()
    return ax.set_ylim([-new_ylim, new_ylim])

def readMEG(fname, s_rate, num_chans, dtype='>f4', factor=133*1000):
    '''Read the data of newer 1 channel files
    '''
    data = np.fromfile(fname, dtype=np.dtype(dtype)).reshape(
            s_rate, num_chans, -1, order='F').swapaxes(0,1).reshape(
                    num_chans, -1, order='F')
    return data*factor

def readOldMEG(fname, num_chans, dtype='<f4', factor=1):
    '''Read the data of older 18 channels files
    '''
    data = np.fromfile(fname, dtype=np.dtype(dtype)).reshape(
            num_chans, -1, order='F')
    return data*factor

def stitch_spectral_bootstrap(data, fs, initres, width_factor, factor=2,
        nit=1000, ci=95, trim=0.2, **kwargs):
    """
    Stitches the output several subspectra of bootstrap_spectral together
    for the arguments look at bootstrap_spectral

    Extra arguments:
    initres - the initial frequency resolution in Hz
    width_factor - the number of steps for each resolution
                   (i.e., the width in Hz is res x width)
    factor - the factor by which the frequency resolution is increased
             for each subspectrum

    Example:
    --------
    initres = 1
    width_factor = 5
    factor = 2

    0 - 5 Hz: resolution 1 Hz
    5 - 15 Hz: resolution 2 Hz
    15 - 35 Hz, resolution 4 Hz
    35 - 75 Hz, resolution 8 Hz
    75 - 155 Hz, resolution 16 Hz
    etc.
    """
    res = [initres]
    fwin = [(0, (width_factor - 1)*initres)]
    while (fwin[-1][1] + res[0]) < fs/2.:
        res += [res[-1]*factor]
        fwin += [(fwin[-1][1] + res[0],
            fwin[-1][1] + res[-1]*width_factor)]
    # calculate the fourier nperseg from the resolution
    nperseg = np.ceil(fs/np.array(res)).astype(int)
    result = list(zip(*[bootstrap_spectral(data, fs, N, win_now, nit=nit, ci=ci,
        trim=trim, **kwargs) for N, win_now in zip(nperseg, fwin)]))
    return [np.concatenate(r, axis=-1) for r in result]

def bootstrap_spectral(data, fs, nperseg, fwin, nit=1000, ci=95,
        trim=0.2, calc_coherence=True):
    """
    Calculate the the psd, coherece and imaginary coherence
    of the data and the bootstrap confidence interval of each
    of these measures

    If trim > 0, a trimmed mean is used for better robustness of the estimates

    Welch's average periodogram is used for the calculation internally

    Input:
    data - numpy array of channels x samples
    fs - sampling frequency
    nperseg - numper of samples for each FFT calculation
              if n output frequencies are requested, n//2+1 input points
              are necessary
    fwin - tuple - the frequency window to calculate,
                   fwin[0] <= f <= fwin[1]
    nit - number of iterations for the bootstrap, the more, the bette will
          be the bootstrap estimate
    ci - float 0 <= ci <= 100: the range of the confidence interval
    trim - float 0 <= trim < 1, the proportion of the data to trim at each
           tail of the distribution (0 amounts to standard mean)

    Output:
    f - numpy array with the frequencies
    psd - shape (channels x 3 x len(f)), i.e. psd[0,0] is lower CI of the
          psd of channel 0, psd[0,1] is the 50% percentile, psd[0,2] is the
          upper percentile
    icoh - imaginary part of coherence
    phs - the phase spectrum
    
    The coherence/imaginary part of coherence is calculated for 
    the lower triangular part of the channels x channels matrix
    
    coh[0] belongs to (channels[1], channels[0])
    coh[1] belongs to (channels[2], channels[0])
    coh[2] belongs to (channels[2], channels[1])
    coh[3] belongs to (channels[3], channels[0])
    .
    .
    .
    """
    print(r"Spectral estimates from {:.1f} to {:.1f} Hz".format(*fwin))
    nchans = data.shape[0]
    # get the indices for the confidence intervals
    ci_idx = np.array([
        int((0.5 - ci/200.)*(nit-1)), # lower CI
        (nit-1)//2, # mean
        int(np.ceil((0.5 + ci/200.)*(nit-1))) # upper CI
        ])
    # get the frequencies
    f = np.fft.rfftfreq(nperseg, d=1./fs)
    f_keep = np.all([
        f >= fwin[0],
        f <= fwin[1]],
        axis = 0)
    print('Number of Fourier coefficients: %d' % f_keep.sum())
    f = f[f_keep]
    psd_segs = scipy.signal.spectral._spectral_helper(data, data, axis=-1,
            nperseg = nperseg, fs=fs, mode='psd',
            scaling='density')[2][:,f_keep,:]
    # get the indices with replacement of the array for the bootstrap
    bootstrap_indices = np.random.random_integers(
            low = 0, high = psd_segs.shape[-1] - 1,
            size = (nit, psd_segs.shape[-1]))
    # perform the bootstrap for the psd
    psd_bootstrap = np.array(
            [scipy.stats.trim_mean(psd_segs[...,idx], trim, axis=-1)
            for idx in bootstrap_indices])
    if calc_coherence:
        # perform the bootstrap for coh and icoh
        coh = []
        icoh = []
        phs = []
        for i in range(nchans):
            for j in range(i):
                print('Channel %d vs. %d.' % (i + 1, j + 1))
                csd_segs = scipy.signal.spectral._spectral_helper(
                data[i], data[j], axis=-1, nperseg = nperseg, fs=fs,
                mode='psd', scaling='density')[2][f_keep]
                # perform the bootstrap
                csd_bootstrap = np.array([
                        (scipy.stats.trim_mean(
                            np.real(csd_segs[...,idx]), trim, axis=-1) + 
                        1j*scipy.stats.trim_mean(
                            np.imag(csd_segs[...,idx]), trim, axis=-1))
                        for idx in bootstrap_indices])
                # get the phase spectrum confidence intervals
                phs.append(np.sort(np.angle(csd_bootstrap,
                    deg=True), axis=0)[ci_idx])
                # normalize the csd bootstrap with the product of the psds
                # for the coherence estimates
                csd_bootstrap /= np.sqrt(psd_bootstrap[:,i]*psd_bootstrap[:,j])
                # get the confidence interval for coherence and icoh
                coh.append(np.sort(np.abs(csd_bootstrap), axis=0)[ci_idx])
                icoh.append(np.sort(np.imag(csd_bootstrap), axis=0)[ci_idx])
    # get the CI of the psd
    psd = np.swapaxes(np.sort(psd_bootstrap, axis=0)[ci_idx], 0, 1)
    if calc_coherence:
        return f, psd, np.array(coh), np.array(icoh), np.array(phs)
    else:
        return f, psd

def stitch_spectral_bootstrap2(data1, data2, fs, initres, width_factor, factor=2,
        nit=1000, ci=95, trim=0.2):
    """
    Stitches the output several subspectra of bootstrap_spectral together
    for the arguments look at bootstrap_spectral

    Extra arguments:
    initres - the initial frequency resolution in Hz
    width_factor - the number of steps for each resolution
                   (i.e., the width in Hz is res x width)
    factor - the factor by which the frequency resolution is increased
             for each subspectrum

    Example:
    --------
    initres = 1
    width_factor = 5
    factor = 2

    0 - 5 Hz: resolution 1 Hz
    5 - 15 Hz: resolution 2 Hz
    15 - 35 Hz, resolution 4 Hz
    35 - 75 Hz, resolution 8 Hz
    75 - 155 Hz, resolution 16 Hz
    etc.
    """
    res = [initres]
    fwin = [(0, (width_factor - 1)*initres)]
    while (fwin[-1][1] + res[0]) < fs/2.:
        res += [res[-1]*factor]
        fwin += [(fwin[-1][1] + res[0],
            fwin[-1][1] + res[-1]*width_factor)]
    # calculate the fourier nperseg from the resolution
    nperseg = np.ceil(fs/np.array(res)).astype(int)
    result = list(zip(*[bootstrap_spectral2(data1, data2, fs, N, win_now,
        nit=nit, ci=ci, trim=trim) for N, win_now in zip(nperseg, fwin)]))
    return [np.concatenate(r, axis=-1) for r in result]

def bootstrap_spectral2(data1, data2, fs, nperseg, fwin, nit=1000, ci=95,
        trim=0.2, calc_coherence=True):
    """
    Calculate the the psd, coherece and imaginary coherence
    of the data and the bootstrap confidence interval of each
    of these measures

    If trim > 0, a trimmed mean is used for better robustness of the estimates

    Welch's average periodogram is used for the calculation internally

    Input:
    data - numpy array of channels x samples
    fs - sampling frequency
    nperseg - numper of samples for each FFT calculation
              if n output frequencies are requested, n//2+1 input points
              are necessary
    fwin - tuple - the frequency window to calculate,
                   fwin[0] <= f <= fwin[1]
    nit - number of iterations for the bootstrap, the more, the bette will
          be the bootstrap estimate
    ci - float 0 <= ci <= 100: the range of the confidence interval
    trim - float 0 <= trim < 1, the proportion of the data to trim at each
           tail of the distribution (0 amounts to standard mean)

    Output:
    f - numpy array with the frequencies
    psd - shape (channels x 3 x len(f)), i.e. psd[0,0] is lower CI of the
          psd of channel 0, psd[0,1] is the 50% percentile, psd[0,2] is the
          upper percentile
    icoh - imaginary part of coherence
    phs - the phase spectrum
    
    The coherence/imaginary part of coherence is calculated for 
    the lower triangular part of the channels x channels matrix
    
    coh[0] belongs to (channels[1], channels[0])
    coh[1] belongs to (channels[2], channels[0])
    coh[2] belongs to (channels[2], channels[1])
    coh[3] belongs to (channels[3], channels[0])
    .
    .
    .
    """
    print("Spectral estimates from %.1f to %.1f Hz" % (fwin[0], fwin[1]))
    nchans1 = data1.shape[0]
    nchans2 = data2.shape[0]
    # get the indices for the confidence intervals
    ci_idx = np.array([
        int((0.5 - ci/200.)*(nit-1)), # lower CI
        (nit-1)//2, # mean
        int(np.ceil((0.5 + ci/200.)*(nit-1))) # upper CI
        ])
    # get the frequencies
    f = np.fft.rfftfreq(nperseg, d=1./fs)
    f_keep = np.all([
        f >= fwin[0],
        f <= fwin[1]],
        axis = 0)
    print('Number of Fourier coefficients: %d' % f_keep.sum())
    f = f[f_keep]
    psd_segs1 = scipy.signal.spectral._spectral_helper(data1, data1, axis=-1,
            nperseg = nperseg, fs=fs, mode='psd',
            scaling='density')[2][:,f_keep,:]
    psd_segs2 = scipy.signal.spectral._spectral_helper(data2, data2,
            axis=-1, nperseg = nperseg, fs=fs, mode='psd',
            scaling='density')[2][:,f_keep,:]
    # get the indices with replacement of the array for the bootstrap
    bootstrap_indices = np.random.random_integers(
            low = 0, high = psd_segs1.shape[-1] - 1,
            size = (nit, psd_segs1.shape[-1]))
    # perform the bootstrap for the psd
    psd_bootstrap1 = np.array(
            [scipy.stats.trim_mean(psd_segs1[...,idx], trim, axis=-1)
            for idx in bootstrap_indices])
    psd_bootstrap2 = np.array(
            [scipy.stats.trim_mean(psd_segs2[...,idx], trim, axis=-1)
            for idx in bootstrap_indices])
    if calc_coherence:
        # perform the bootstrap for coh and icoh
        coh = []
        icoh = []
        phs = []
        for i in range(nchans1):
            for j in range(nchans2):
                print('Channel %d vs. %d.' % (i + 1, j + 1))
                csd_segs = scipy.signal.spectral._spectral_helper(
                data1[i], data2[j], axis=-1, nperseg = nperseg, fs=fs,
                mode='psd', scaling='density')[2][f_keep]
                # perform the bootstrap
                csd_bootstrap = np.array([
                        (scipy.stats.trim_mean(
                            np.real(csd_segs[...,idx]), trim, axis=-1) + 
                        1j*scipy.stats.trim_mean(
                            np.imag(csd_segs[...,idx]), trim, axis=-1))
                        for idx in bootstrap_indices])
                # get the phase spectrum confidence intervals
                phs.append(np.sort(np.angle(csd_bootstrap,
                    deg=True), axis=0)[ci_idx])
                # normalize the csd bootstrap with the product of the psds
                # for the coherence estimates
                csd_bootstrap /= (
                        np.sqrt(psd_bootstrap1[:,i]*psd_bootstrap2[:,j]))
                # get the confidence interval for coherence and icoh
                coh.append(np.sort(np.abs(csd_bootstrap), axis=0)[ci_idx])
                icoh.append(np.sort(np.imag(csd_bootstrap), axis=0)[ci_idx])
    # get the CI of the psd
    psd1 = np.swapaxes(np.sort(psd_bootstrap1, axis=0)[ci_idx], 0, 1)
    psd2 = np.swapaxes(np.sort(psd_bootstrap2, axis=0)[ci_idx], 0, 1)
    if calc_coherence:
        return f, psd1, psd2, np.array(coh), np.array(icoh), np.array(phs)
    else:
        return f, psd1, psd2

def stepdown_p(stat, stat_boot):
    """
    Calculate FWER corrected p values in multiple comparison problems.

    This implements the method from:

    Romano, J.P., and Wolf, M. (2016). Efficient computation of adjusted
    p-values for resampling-based stepdown multiple testing.
    Stat. Probab. Lett. 113, 38–40.

    Input:
        stat - array shape N - "true" values of the test statistic
        stat_boot - array shape M x N - the values of the test statistic
            obtained from M bootstraps (or permutations, or ...)
    Returns:
        p_values
    """
    M, N = stat_boot.shape
    if not N == len(stat):
        raise ValueError('length of stat must match number of variables'
                ' in stat_boot')
    # order the test hypotheses with decreasing significance 
    order = np.argsort(stat)[::-1]
    stat = stat[order]
    stat_boot = stat_boot[:,order]
    # initialize results array
    p = [(np.sum(np.max(stat_boot[:,i:], 1) >= stat[i]) + 1)/float(M + 1)
            for i in tqdm.trange(N)]
    # enforce monotonicity
    p = np.maximum.accumulate(p)
    # revert the original order of the hypothesis
    return p[np.argsort(order)]
