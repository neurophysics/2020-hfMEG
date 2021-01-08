import numpy as np
import scipy.linalg
import scipy.ndimage
from scipy import fftpack
from tqdm import trange

"""
This module implements functions to calculate the autocorrelation function (acf)
and partial autocorrelation function (pacf) of a 1d time series.

It has been tested to be significantly faster than the implementations in stattools
"""

def autocorr_time(acf, c=5):
    taus = 2*np.cumsum(acf, axis=-1) - 1
    m = (np.ones_like(acf)*np.arange(acf.shape[-1])) >= (c*taus)
    win = np.where(np.any(m, axis=-1), np.argmax(m, axis=-1),
            acf.shape[-1]-1)
    return (taus.reshape(-1, taus.shape[-1])[
        list(range(np.prod(taus.shape[:-1]))), np.ravel(win)]).reshape(win.shape)

def _lagdot(x, nlags):
    if x.ndim == 1:
        return np.r_[1, [np.dot(x[:-l],x[l:])/(len(x) - l)
            for l in range(1,nlags+1)]] 

def acf2(x, axis, nlags):
    fftlen = fftpack.next_fast_len(x.shape[axis]*2 - 1)
    S = fftpack.fft(x, n=fftlen, axis=axis)
    corr = fftpack.ifft(S*np.conj(S), n=fftlen, axis=axis)
    return np.take(corr.real/x.shape[axis], np.arange(nlags),
            axis=axis)

def acf(x, nlags=50, N_bootstrap=0):
    """
    Calculate the acf of x
    the unbiased version (with n-l in the demonitor, where n is the number of
    samples and l is the current lag) is calculated.

    Input:
    x (1d iterable): time series
    nlags (int): number of lags for which the acf is to becalculated

    Output:
    acf (ndarray of length nlags + 1): the acf function, starting at lag 0
    """
    x = x.argsort(-1).argsort(-1)
    x = (x - x.mean(-1)[...,np.newaxis])/x.std(-1)[...,np.newaxis]
    x_acf = acf2(x, axis=-1, nlags=nlags)
    if N_bootstrap:
        x_acf_boot = np.array([
            acf2(x[...,np.argsort(np.random.randn(x.shape[-1]))],
                axis=-1, nlags=nlags)
            for _ in trange(N_bootstrap)])
        return x_acf, x_acf_boot
    else:
        return x_acf

def cacf(x, nlags=50, N_bootstrap=0):
    """
    Calculate the circular acf of x
    the unbiased version (with n-l in the demonitor, where n is the number of
    samples and l is the current lag) is calculated.

    Input:
    x (1d iterable): time series of circular (phase) data
    nlags (int): number of lags for which the acf is to becalculated

    Output:
    acf (ndarray of length nlags + 1): the acf function, starting at lag 0
    """
    # the only difference between the cacf and the acf is proper standardization
    mean = np.angle(np.mean(np.exp(1j*x), axis=-1))
    std = np.sqrt(np.mean(np.sin(x - mean[...,np.newaxis])**2, axis=-1))
    x = np.sin(x - mean[...,np.newaxis])/std[...,np.newaxis]
    x_acf = acf2(x, axis=-1, nlags=nlags)
    if N_bootstrap:
        x_acf_boot = np.array([
            acf2(x[...,np.argsort(np.random.randn(x.shape[-1]))],
                axis=-1, nlags=nlags)
            for _ in trange(N_bootstrap)])
        return x_acf, x_acf_boot
    else:
        return x_acf

def smooth_acf_map(acf_map, nlags, axis=0):
    """
    Smooth an autocorrelation map across one axis

    If smoothed across the last axis (lags), the autocorrelation at lag
    0 is not included in the smoothing

    smoothing is performed as moving average across nlags
    """
    # make the axis positive
    axis = list(range(acf_map.ndim))[axis]
    if axis == (acf_map.ndim -1):
        return np.concatenate([
            np.ones(
                np.r_[acf_map.shape[:axis], 1,
                    acf_map.shape[axis + 1:]].astype(int),
                float),
            scipy.ndimage.convolve1d(
                np.take(acf_map, list(range(1, acf_map.shape[axis],1)), axis),
                    nlags*[1./nlags],
                mode='mirror', axis=axis)],
            axis=axis)
    else:
        return scipy.ndimage.convolve1d(acf_map, nlags*[1./nlags],
                mode='mirror', axis=axis)

def pacf(x_acf):
    """
    Calculate the pacf from the acf using the levinson-durban recursion

    As input, the acf is needed
    """
    x, pacf = scipy.linalg._solve_toeplitz.levinson(
            np.r_[x_acf[-2:0:-1], x_acf[:-1]], x_acf[1:])
    return pacf
    #return np.r_[1, [scipy.linalg.solve_toeplitz(x_acf[:l], x_acf[1:l+1])[-1]
    #    for l in xrange(1,  len(x_acf), 1)]]

def _lagdiff(x, lag):
    return np.abs(x[:-lag] - x[lag:]).mean()

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def phase_diff(x, lag=1, N_bootstrap=0):
    # divide by modulus
    x = x/np.abs(x)
    # get the average phase diff
    avg_diff = np.abs(x - x[:,np.newaxis]).sum()/float(len(x)*(len(x) - 1))
    # get the difference between lagged trials
    lag_diff = _lagdiff(x, lag) 
    # if requested, perform a bootstrap
    if N_bootstrap:
        lag_diff_boot = np.array([
            _lagdiff(x[np.argsort(np.random.randn(len(x)))],
                lag)
            for _ in range(N_bootstrap)])
        return lag_diff/avg_diff, lag_diff_boot/avg_diff
    else:
        return lag_diff/avg_diff

def mean_std_phasediff(x, lag=1):
    x = x/np.abs(x)
    # get the average phase diff
    x_diff = upper_tri_masking(np.abs(x - x[:,np.newaxis]))
    avg_diff = x_diff.mean()
    std_diff = np.sqrt(np.mean((x_diff - avg_diff)**2))
    return avg_diff, std_diff

if __name__ == '__main__':
    def gen_AR1(p, N=1000):
        x = np.random.randn(N)
        for i in range(1,N):
            x[i] += p*x[i - 1]
        return x
    import statsmodels.tsa.stattools as tsa
    # generate some testing data
    x = gen_AR1(0.9, 2000)
    x_acf = acf(x, lags=50)
    x_pacf = pacf(x_acf)
    assert np.allclose(x_acf, tsa.acf(x, unbiased=True, nlags=50))
    print('Calculation of autocorrelation correct')
    assert np.allclose(x_pacf, tsa.pacf(x, nlags=50, method='ld'))
    print('Calculation of partial autocorrelation correct')


