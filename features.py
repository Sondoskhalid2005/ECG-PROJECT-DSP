import numpy as np
from scipy.fftpack import dct

# -------------------------
# 1) Morphological Features
# -------------------------
def morphological_features(beat, fs=360):
    beat = np.array(beat)

    peak_idx = np.argmax(beat)
    peak_amp = beat[peak_idx]

    rel_peak_time = peak_idx / fs

    # width at half maximum
    half = 0.5 * peak_amp
    left = np.where(beat[:peak_idx] <= half)[0]
    left_idx = left[-1] if len(left) > 0 else 0
    right = np.where(beat[peak_idx:] <= half)[0]
    right_idx = peak_idx + (right[0] if len(right) > 0 else len(beat)-1)
    width_samples = right_idx - left_idx
    width_sec = width_samples / fs

    area = np.trapz(np.abs(beat))
    energy = np.sum(beat**2)

    before_slope = beat[peak_idx] - beat[peak_idx-1] if peak_idx > 0 else 0
    after_slope = beat[peak_idx+1] - beat[peak_idx] if peak_idx < len(beat)-1 else 0

    return np.array([peak_amp, rel_peak_time, width_sec, area, energy, before_slope, after_slope])


# -------------------------
# 2) DCT Features
# -------------------------
def dct_features(beat, n_coeff=20):
    coeffs = dct(beat, norm='ortho')[:n_coeff]
    return coeffs


# -------------------------
# 3) Autocorrelation Features
# -------------------------
def autocorr_features(beat, n_lags=50):
    ac_full = np.correlate(beat, beat, mode='full')
    ac = ac_full[len(ac_full)//2 : len(ac_full)//2 + n_lags]
    if ac[0] != 0:
        ac = ac / ac[0]
    return ac


# -------------------------
# Combine all features
# -------------------------
def build_feature_vector(beat, fs=360, n_dct=20, n_lags=50):
    beat = np.array(beat)
    morph = morphological_features(beat, fs)
    dct_f = dct_features(beat, n_coeff=n_dct)
    ac_f = autocorr_features(beat, n_lags=n_lags)
    return np.hstack([morph, dct_f, ac_f])


def build_feature_matrix(signals):
    return np.vstack([build_feature_vector(sig) for sig in signals])
