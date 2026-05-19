#!/usr/bin/env python3
"""
EEG WebSocket Server for Brain Concentration Visualization
Supports:
  --source sim     : Random simulation (default)
  --source deap    : DEAP dataset .dat file playback
  --source mental  : Kaggle EEG Mental State CSV playback
  --source emotiv  : Emotiv EPOC X via Cortex API (real connection)

Usage:
  pip install websockets numpy scipy pandas
  python eeg_server.py --source sim
  python eeg_server.py --source deap --file data/s01.dat --trial 0 --speed 2.0
  python eeg_server.py --source mental --file data/mental-state.csv --speed 1.0
  python eeg_server.py --source emotiv

DEAP Dataset: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
Kaggle Mental State: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state
"""

import asyncio
import json
import pickle
import argparse
import random
import time
import os
from typing import Optional

try:
    from eeg_model import make_detector
    MODEL_OK = True
except ImportError:
    MODEL_OK = False

try:
    from calibration import Calibrator
    CAL_OK = True
except ImportError:
    CAL_OK = False

try:
    from experiment import ExperimentRunner, PROTOCOLS
    EXP_OK = True
except ImportError:
    EXP_OK = False

try:
    import numpy as np
    from scipy.signal import welch, butter, filtfilt, iirnotch
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False
    print("[WARN] numpy/scipy not found. Install with: pip install numpy scipy")

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False
    print("[WARN] pandas not found. Install with: pip install pandas")

try:
    import websockets
except ImportError:
    print("[ERROR] websockets not found. Install with: pip install websockets")
    exit(1)


# ── DEAP Channel Mapping ──────────────────────────────────────────────────────
DEAP_CH = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1',
           'P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4',
           'F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']

EMOTIV_TO_DEAP = {
    'AF3':1,'AF4':17,'F7':3,'F3':2,'F4':19,'F8':20,
    'FC5':4,'FC6':21,'T7':7,'T8':25,'P7':11,'P8':29,
    'O1':13,'O2':31
}
EMOTIV_CHS = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2']
DEAP_IDX   = [EMOTIV_TO_DEAP[ch] for ch in EMOTIV_CHS]

FS = 128

# ── Muse S Athena channel config ──────────────────────────────────────────────
# EEG channels: TP9, AF7, AF8, TP10  |  Reference: FP1, FP2
MUSE_CHS    = ['TP9', 'AF7', 'FP1', 'FP2', 'AF8', 'TP10']
MUSE_EEG    = ['TP9', 'AF7', 'AF8', 'TP10']   # active EEG only
MUSE_EEG_IDX = [0, 1, 4, 5]                   # indices in MUSE_CHS
FS_MUSE     = 256

# Spatial weights per electrode for synthesizing 14ch from band powers
# (frontal electrodes more sensitive to beta/concentration)
SPATIAL_W = [1.2,1.2, 1.1,1.1,1.1,1.1, 1.0,1.0, 0.8,0.8, 0.7,0.7, 0.6,0.6]

# Frontal electrode indices (AF3=0, AF4=1, F7=2, F3=3, F4=4, F8=5)
# used for eye-blink detection
FRONTAL_IDX = [0, 1, 2, 3, 4, 5]

# ── Global State ──────────────────────────────────────────────────────────────
STATE = {
    'channels': [50.0] * 14,
    'bands': {'delta': 30.0, 'theta': 40.0, 'alpha': 50.0,
              'beta': 50.0, 'gamma': 30.0, 'concentration': 50.0,
              'engagement_index': 0.5, 'faa': 0.0},
    'settings': {'thLow': 30, 'thHigh': 70, 'dt': 3, 'slope': 1.5},
    'source': 'sim',
    'baseline': None,   # set after calibration: {'engagement_index': float}
}

def update_state(channels, bands):
    STATE['channels'] = [round(v, 2) for v in channels]
    STATE['bands'] = bands

# ── Signal Processing ─────────────────────────────────────────────────────────

# Pre-build filters once (avoids recomputing per frame)
def _make_filters(fs):
    # Bandpass 1-45 Hz (removes DC drift + high-freq noise)
    b_bp, a_bp = butter(4, [1.0/(fs/2), 45.0/(fs/2)], btype='band')
    # Notch 50 Hz (EU power line) — change to 60.0 for US/JP
    b_n50, a_n50 = iirnotch(50.0, Q=30, fs=fs)
    # Notch 60 Hz (US power line)
    b_n60, a_n60 = iirnotch(60.0, Q=30, fs=fs)
    return (b_bp, a_bp), (b_n50, a_n50), (b_n60, a_n60)

_FILTERS_CACHE = {}  # keyed by fs so Muse(256Hz) and Emotiv(128Hz) use correct filters

def get_filters(fs=None):
    if fs is None:
        fs = FS
    if fs not in _FILTERS_CACHE:
        _FILTERS_CACHE[fs] = _make_filters(fs)
    return _FILTERS_CACHE[fs]

def preprocess(eeg_14ch, fs=None, notch_hz=50, blink_thresh_uv=80.0, frontal_idx=None):
    """
    Signal processing pipeline applied to raw EEG before band-power extraction.

    Steps:
      1. Bandpass filter  1-45 Hz  (removes DC + EMG noise)
      2. Notch filter     50/60 Hz (removes power-line interference)
      3. Eye-blink rejection       (interpolate blink segments on frontal ch)

    Args:
      eeg_14ch      : np.ndarray (14, n_samples) raw EEG in μV
      notch_hz      : 50 (EU) or 60 (US) power-line frequency
      blink_thresh_uv: amplitude threshold for blink detection (μV)

    Returns:
      clean         : np.ndarray (14, n_samples) cleaned EEG
      artifacts     : dict with blink count and rejected sample ratio
    """
    _fs = fs if fs is not None else FS
    (b_bp, a_bp), (b_n50, a_n50), (b_n60, a_n60) = get_filters(_fs)
    n_samples = eeg_14ch.shape[1]
    clean = np.empty_like(eeg_14ch, dtype=float)

    # Need at least 3× filter order samples for filtfilt
    min_len = 13  # 4th-order butter → padlen = 12
    if n_samples < min_len:
        return eeg_14ch.astype(float), {'blinks': 0, 'rejected_ratio': 0.0}

    for i, ch in enumerate(eeg_14ch):
        sig = ch.astype(float)
        # 1. Bandpass
        sig = filtfilt(b_bp, a_bp, sig)
        # 2. Notch
        if notch_hz == 60:
            sig = filtfilt(b_n60, a_n60, sig)
        else:
            sig = filtfilt(b_n50, a_n50, sig)
        clean[i] = sig

    # 3. Eye-blink rejection on frontal channels
    blink_win  = max(1, int(_fs * 0.2))  # 200 ms window at actual sample rate
    blink_count   = 0
    rejected_samp = 0
    n_ch = clean.shape[0]
    _frontal = [fi for fi in (frontal_idx if frontal_idx is not None else FRONTAL_IDX) if fi < n_ch]

    for fi in _frontal:
        sig = clean[fi]
        t = 0
        while t <= n_samples - blink_win:
            segment = sig[t:t + blink_win]
            if np.max(np.abs(segment)) > blink_thresh_uv:
                # Mark blink window — interpolate linearly across it
                start = max(0, t - 2)
                end   = min(n_samples - 1, t + blink_win + 2)
                # Use edge values for linear interpolation
                v0 = sig[start]
                v1 = sig[end]
                interp = np.linspace(v0, v1, end - start)
                clean[fi, start:end] = interp
                blink_count  += 1
                rejected_samp += blink_win
                t += blink_win  # skip past this blink
            else:
                t += 1

    rejected_ratio = round(rejected_samp / (n_samples * max(1, len(_frontal)) + 1e-9), 3)
    return clean, {'blinks': blink_count, 'rejected_ratio': rejected_ratio}

def bandpower(signal, fs, fmin, fmax):
    nperseg = min(len(signal), fs * 2)
    if nperseg < 4:
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    result = float(np.mean(psd[mask])) if mask.any() else 0.0
    return result if np.isfinite(result) else 0.0

def compute_frame(eeg_14ch, fs=FS, apply_preprocess=True, notch_hz=50, baseline=None):
    """
    baseline: dict with keys 'alpha', 'theta', 'beta' (from calibration relax phase)
              If provided, concentration is baseline-relative.
    """
    artifacts = {'blinks': 0, 'rejected_ratio': 0.0}
    if apply_preprocess and NUMPY_OK:
        eeg_14ch, artifacts = preprocess(eeg_14ch, fs=fs, notch_hz=notch_hz)

    ratios, ch_bands = [], []
    for ch in eeg_14ch:
        b_delta = bandpower(ch, fs, 1,  4)
        b_theta = bandpower(ch, fs, 4,  8)
        b_alpha = bandpower(ch, fs, 8,  12)
        b_beta  = bandpower(ch, fs, 12, 30)
        b_gamma = bandpower(ch, fs, 30, 45)
        # Engagement Index: β/(α+θ) — more discriminating than β/α alone
        ratios.append(b_beta / (b_alpha + b_theta + 1e-9))
        ch_bands.append((b_delta, b_theta, b_alpha, b_beta, b_gamma))

    mn, mx = min(ratios), max(ratios)
    norm = [(r-mn)/(mx-mn)*100 for r in ratios] if mx > mn else [50.0]*14

    avg = lambda i: float(np.mean([ch_bands[c][i] for c in range(14)]))
    g_alpha = avg(2)
    g_theta = avg(1)
    g_beta  = avg(3)

    # Engagement Index (global)
    ei = g_beta / (g_alpha + g_theta + 1e-9)

    # Baseline-relative concentration
    if baseline:
        b_ei = baseline.get('engagement_index', 0.5)
        # Scale so that baseline EI → 30%, focused EI → 70%+
        conc = float(np.clip((ei / (b_ei + 1e-9) - 0.5) * 60 + 50, 0, 100))
    else:
        conc = float(np.clip((ei - 0.3) * 60 + 50, 0, 100))

    # Frontal Alpha Asymmetry: (F4_alpha - F3_alpha) / (F4_alpha + F3_alpha)
    # EMOTIV_CHS: AF3=0,AF4=1,F7=2,F3=3,F4=4,F8=5 ...
    f3_alpha  = bandpower(eeg_14ch[3], fs, 8, 12)  # F3
    f4_alpha  = bandpower(eeg_14ch[4], fs, 8, 12)  # F4
    faa_denom = f3_alpha + f4_alpha + 1e-9
    faa       = round((f4_alpha - f3_alpha) / faa_denom, 3)  # −1..+1, +ve = approach/focus

    # EMG proxy: high-frequency power (>40Hz) — flag if dominant
    emg_l = bandpower(eeg_14ch[0], fs, 40, 45)   # AF3
    emg_r = bandpower(eeg_14ch[1], fs, 40, 45)   # AF4
    emg_ratio = (emg_l + emg_r) / (g_beta + 1e-9)
    emg_warn  = bool(emg_ratio > 2.0)  # beta-range dominated by EMG-like noise

    # Signal quality proxy per channel (0=disconnected, 1=excellent)
    # Based on coefficient of variation: flat line → 0, noisy → low, good signal → high
    sig_quality = []
    for ch in eeg_14ch:
        std_v = float(np.std(ch))
        mean_abs = float(np.mean(np.abs(ch))) + 1e-9
        cv = std_v / mean_abs
        if cv < 0.05:           q = 0.05   # flat / disconnected
        elif cv > 10.0:         q = 0.15   # excessive noise / clipping
        else:                   q = float(np.clip(0.15 + cv * 0.17, 0.15, 1.0))
        sig_quality.append(round(q, 3))

    return {
        'type': 'eeg',
        'channels': [round(v, 2) for v in norm],
        'contact_quality': sig_quality,
        'artifacts': {**artifacts, 'emg_warn': emg_warn},
        'bands': {
            'delta': round(avg(0)*1e6, 2),
            'theta': round(g_theta*1e6, 2),
            'alpha': round(g_alpha*1e6, 2),
            'beta':  round(g_beta*1e6,  2),
            'gamma': round(avg(4)*1e6, 2),
            'concentration': round(conc, 2),
            'engagement_index': round(ei, 4),
            'faa': faa,
        }
    }

# ── Mental State bands → 14ch synthesis ──────────────────────────────────────
def bands_to_14ch(delta, theta, alpha, beta, gamma, noise=6.0):
    """
    Synthesize 14 electrode values from scalar band powers.
    Beta/Alpha ratio drives the base concentration, spatial weights
    distribute it across electrodes (frontal > occipital).
    """
    ratio = beta / (alpha + 1e-9)
    base  = float(np.clip((ratio - 0.3) * 40 + 50, 5, 95))
    vals  = []
    for w in SPATIAL_W:
        v = base * w + (random.random() - 0.5) * noise
        vals.append(float(np.clip(v, 0, 100)))
    # re-normalize to keep mean = base
    mn, mx = min(vals), max(vals)
    if mx > mn:
        vals = [(v-mn)/(mx-mn)*90+5 for v in vals]
    return vals

# ── DEAP Source ───────────────────────────────────────────────────────────────
async def _tick_experiment(ws, exp_holder, bands):
    """Advance experiment timer; send transition/done immediately, progress every 1s."""
    runner = exp_holder[0]
    if runner is None or runner.done:
        return
    status = runner.tick(bands)
    state  = status.get('state')

    async def _send(payload):
        try:
            await ws.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            pass

    if status.get('transition') or state == 'done':
        await _send({'type': 'experiment', **status})
        if state == 'done':
            path = runner.save()
            runner.export_csv(path.replace('.json', '.csv'))
            await _send({'type': 'experiment', 'state': 'saved', 'path': path,
                         'summary': runner._summary()})
            print(f"[EXP] Experiment done. Saved → {path}")
    else:
        remaining = status.get('remaining', 0)
        last_rem  = getattr(runner, '_last_sent_remaining', None)
        if last_rem is None or abs(last_rem - remaining) >= 1.0:
            runner._last_sent_remaining = remaining
            await _send({'type': 'experiment', **status})

async def _push_cal(ws, cal_holder, bands):
    """Feed bands to calibrator; send progress every ~1s, transition/done immediately."""
    cal = cal_holder[0]
    if cal is None or cal.done:
        return
    status = cal.push(bands)
    state  = status.get('state')

    async def _send(payload):
        try:
            await ws.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            pass

    if status.get('transition') or state == 'done':
        await _send({'type': 'calibration', **status})
        if state == 'done':
            STATE['settings'].update(cal.to_settings())
            r = cal._result or {}
            rb = r.get('relax', {}).get('bands', {})
            if rb:
                ei_base = rb.get('beta', 1) / (rb.get('alpha', 1) + rb.get('theta', 1) + 1e-9)
                STATE['baseline'] = {'engagement_index': round(ei_base, 4)}
                print(f"[CAL] Baseline EI set: {STATE['baseline']['engagement_index']:.4f}")
            await _send({'type': 'settings', 'settings': STATE['settings']})
            print(f"[CAL] {cal.summary()}")
    else:
        remaining = status.get('remaining', 0)
        last_rem  = getattr(cal, '_last_sent_remaining', None)
        if last_rem is None or abs(last_rem - remaining) >= 1.0:
            cal._last_sent_remaining = remaining
            await _send({'type': 'calibration', **status})

def _add_vpattern(frame, detector, eeg_seg=None):
    """Attach V-pattern detection result to a frame dict in-place."""
    if detector is None:
        return
    conc = frame.get('bands', {}).get('concentration', 50.0)
    if hasattr(detector, '_buf'):
        # Rule-based scalar detector (VPatternRuleBased)
        result = detector.push(conc)
    elif eeg_seg is not None and hasattr(detector, '_ring'):
        # ML detector with EEG segment available
        result = detector.push(eeg_seg, conc)
    else:
        # ML detector called without EEG segment (stream_sim/stream_mental) — return inactive
        result = {'v_prob': 0.0, 'v_active': False, 'rule_based': False}
    frame['vpattern'] = result

async def stream_deap(ws, dat_file, trial=0, speed=1.0, notch_hz=50, no_preprocess=False,
                      detector=None, cal_holder=None, exp_holder=None):
    print(f"[DEAP] Loading {dat_file}, trial={trial}, speed={speed}x")
    with open(dat_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    n_trials = data['data'].shape[0]
    if trial >= n_trials:
        raise ValueError(f"Trial {trial} out of range — file has {n_trials} trials (0–{n_trials-1})")
    eeg_all = data['data'][trial, :32, :]
    labels  = data['labels'][trial]
    emotiv  = eeg_all[DEAP_IDX, :]

    n    = emotiv.shape[1]
    win  = FS
    step = FS // 4
    interval = (step / FS) / speed

    print(f"[DEAP] Labels: valence={labels[0]:.1f} arousal={labels[1]:.1f}")
    await ws.send(json.dumps({
        'type': 'meta', 'trial': trial,
        'labels': {'valence': float(labels[0]), 'arousal': float(labels[1]),
                   'dominance': float(labels[2]), 'liking': float(labels[3])},
        'duration': n / FS
    }))

    for start in range(0, n - win, step):
        seg   = emotiv[:, start:start+win]
        frame = compute_frame(seg, notch_hz=notch_hz, baseline=STATE['baseline'],
                              apply_preprocess=not no_preprocess)
        frame['progress']  = round(start/n, 3)
        frame['timestamp'] = start/FS
        _add_vpattern(frame, detector, eeg_seg=seg)
        update_state(frame['channels'], frame['bands'])
        if cal_holder:
            await _push_cal(ws, cal_holder, frame['bands'])
        if exp_holder:
            await _tick_experiment(ws, exp_holder, frame['bands'])
        try:
            await ws.send(json.dumps(frame))
            await asyncio.sleep(interval)
        except websockets.exceptions.ConnectionClosed:
            break

    await ws.send(json.dumps({'type': 'done', 'trial': trial}))
    print(f"[DEAP] Trial {trial} finished")

# ── Mental State CSV Source ───────────────────────────────────────────────────
async def stream_mental(ws, csv_file, speed=1.0, detector=None, cal_holder=None, exp_holder=None):
    """
    Kaggle EEG Brainwave Dataset - Mental State
    Supports two column formats:
      A) Simple band power : delta, theta, lowAlpha/highAlpha, lowBeta/highBeta, label
      B) Frequency bins    : freq_XXX_N (Hz×10 encoded), Label
    """
    if not PANDAS_OK:
        await ws.send(json.dumps({'type':'error','message':'pandas required: pip install pandas'}))
        return

    print(f"[MENTAL] Loading {csv_file}, speed={speed}x")
    df = pd.read_csv(csv_file)
    print(f"[MENTAL] Rows: {len(df)}, Columns: {len(df.columns)}")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_label = find_col(['label'])
    col_delta = find_col(['delta'])
    col_theta = find_col(['theta'])
    col_alpha = find_col(['highalpha', 'alpha'])
    col_beta  = find_col(['highbeta',  'beta'])
    col_gamma = find_col(['highgamma', 'gamma'])

    # ── Format B: freq_XXX_N frequency-bin columns ───────────────────────────
    freq_cols = [c for c in df.columns if c.startswith('freq_')]
    if freq_cols and not all([col_delta, col_theta, col_alpha, col_beta]):
        print(f"[MENTAL] Detected freq-bin format ({len(freq_cols)} bins)")

        def band_mean(fmin, fmax):
            """Average power across frequency bins in [fmin, fmax) Hz."""
            cols = []
            for c in freq_cols:
                parts = c.split('_')
                if len(parts) >= 2:
                    try:
                        hz = int(parts[1]) / 10.0
                        if fmin <= hz < fmax:
                            cols.append(c)
                    except ValueError:
                        pass
            if cols:
                return df[cols].mean(axis=1)
            return pd.Series([0.0] * len(df))

        df['_delta'] = band_mean(1,  4)
        df['_theta'] = band_mean(4,  8)
        df['_alpha'] = band_mean(8,  12)
        df['_beta']  = band_mean(12, 30)
        df['_gamma'] = band_mean(30, 45)
        col_delta, col_theta = '_delta', '_theta'
        col_alpha, col_beta, col_gamma = '_alpha', '_beta', '_gamma'
        print(f"[MENTAL] Band cols: delta={len([c for c in freq_cols if int(c.split('_')[1])/10<4])} "
              f"theta=... alpha=... beta=... gamma=...")

    if not all([col_delta, col_theta, col_alpha, col_beta]):
        msg = (f'Cannot find band columns.\n'
               f'Expected: delta/theta/alpha/beta  OR  freq_XXX_N format\n'
               f'First 5 cols: {list(df.columns[:5])}')
        await ws.send(json.dumps({'type':'error', 'message': msg}))
        print(f"[MENTAL] ERROR: {msg}")
        return

    # ── Label encoding ────────────────────────────────────────────────────────
    label_names = {0:'FOCUSED', 1:'RELAXED', 2:'NEUTRAL'}
    if col_label and df[col_label].dtype == object:
        uniq = df[col_label].str.strip().str.upper().unique()
        print(f"[MENTAL] Labels: {uniq}")
        lmap = {}
        for u in uniq:
            if 'FOCUS' in u or 'CONCENTRAT' in u: lmap[u] = 0
            elif 'RELAX' in u:                     lmap[u] = 1
            else:                                  lmap[u] = 2
        df['_label_int'] = df[col_label].str.strip().str.upper().map(lmap).fillna(2)
    elif col_label:
        df['_label_int'] = pd.to_numeric(df[col_label], errors='coerce').fillna(2)
    else:
        df['_label_int'] = 2

    n = len(df)

    # Per-band normalisation: scale each band so its median ≈ 50
    # This keeps all bands in a comparable 0-200 range regardless of 1/f slope
    def _band_scale(col):
        med = float(df[col].abs().quantile(0.5))
        return (50.0 / med) if med > 1e-9 else 1.0

    sc_d = _band_scale(col_delta)
    sc_t = _band_scale(col_theta)
    sc_a = _band_scale(col_alpha)
    sc_b = _band_scale(col_beta)
    sc_g = _band_scale(col_gamma) if col_gamma else sc_a
    print(f"[MENTAL] Per-band scales: δ={sc_d:.1f} θ={sc_t:.1f} α={sc_a:.1f} β={sc_b:.1f} γ={sc_g:.1f}")

    await ws.send(json.dumps({
        'type': 'meta',
        'source': 'mental_state',
        'rows': n,
        'columns': list(df.columns),
        'duration': n * 0.5
    }))

    interval = 0.5 / speed  # ~2 rows/s at real speed

    for i, row in df.iterrows():
        delta = float(row[col_delta]) * sc_d
        theta = float(row[col_theta]) * sc_t
        alpha = float(row[col_alpha]) * sc_a
        beta  = float(row[col_beta])  * sc_b
        gamma = (float(row[col_gamma]) * sc_g) if col_gamma else alpha * 0.5
        lbl   = int(row['_label_int'])

        ratio = beta / (alpha + 1e-9)
        conc  = float(np.clip((ratio - 0.3) * 40 + 50, 0, 100))

        channels = bands_to_14ch(delta, theta, alpha, beta, gamma)

        bands_out = {
            'delta': round(delta, 2),
            'theta': round(theta, 2),
            'alpha': round(alpha, 2),
            'beta':  round(beta,  2),
            'gamma': round(gamma, 2),
            'concentration': round(conc, 2)
        }
        frame = {
            'type': 'eeg',
            'channels': [round(v, 2) for v in channels],
            'bands': bands_out,
            'progress': round(i / n, 3),
            'timestamp': round(i * 0.5, 2),
            'label': label_names.get(lbl, 'UNKNOWN')
        }
        update_state(frame['channels'], frame['bands'])
        _add_vpattern(frame, detector)
        if cal_holder:
            await _push_cal(ws, cal_holder, frame['bands'])
        if exp_holder:
            await _tick_experiment(ws, exp_holder, frame['bands'])

        try:
            await ws.send(json.dumps(frame))
            await asyncio.sleep(interval)
        except websockets.exceptions.ConnectionClosed:
            break

    await ws.send(json.dumps({'type': 'done', 'source': 'mental_state'}))
    print("[MENTAL] Finished playback")

# ── Simulation Source ─────────────────────────────────────────────────────────
async def stream_sim(ws, detector=None, cal_holder=None, exp_holder=None):
    print("[SIM] Starting simulation stream")
    vals = [50.0] * 14
    t    = 0.0
    while True:
        t += 0.1
        for i in range(14):
            vals[i] += (random.random() - 0.5) * 6
            vals[i]  = max(5.0, min(95.0, vals[i]))
        avg   = sum(vals) / 14
        focus = avg / 100
        b = {
            'delta': round(30 + random.gauss(0, 5),  2),
            'theta': round(40 + random.gauss(0, 7),  2),
            'alpha': round(60 - focus*40 + random.gauss(0, 5), 2),
            'beta':  round(30 + focus*50 + random.gauss(0, 5), 2),
            'gamma': round(20 + focus*30 + random.gauss(0, 4), 2),
            'concentration': round(avg, 2)
        }
        update_state(vals, b)
        frame = {'type': 'eeg', 'channels': STATE['channels'],
                 'bands': b, 'progress': -1, 'timestamp': round(t, 2)}
        _add_vpattern(frame, detector)
        if cal_holder:
            await _push_cal(ws, cal_holder, b)
        if exp_holder:
            await _tick_experiment(ws, exp_holder, b)
        try:
            await ws.send(json.dumps(frame))
            await asyncio.sleep(0.1)
        except websockets.exceptions.ConnectionClosed:
            break

# ── Emotiv EPOC X — Cortex API ────────────────────────────────────────────────
async def stream_emotiv(ws, detector=None, cal_holder=None, exp_holder=None):
    """
    Connects to Emotiv Cortex API at wss://localhost:6868.
    Requires: Emotiv App running + pip install websockets

    Cortex flow:
      1. getCortexInfo  → get version
      2. requestAccess  → user grants in Emotiv App
      3. authorize      → get auth token
      4. createSession  → get session id
      5. subscribe eeg  → receive raw EEG frames
    """
    CORTEX_URL = 'wss://localhost:6868'
    # Credentials from env vars or command-line args; fall back to placeholder
    CLIENT_ID  = STATE.get('emotiv_client_id')  or os.environ.get('EMOTIV_CLIENT_ID',  'YOUR_CLIENT_ID')
    CLIENT_SEC = STATE.get('emotiv_client_sec') or os.environ.get('EMOTIV_CLIENT_SEC', 'YOUR_CLIENT_SECRET')

    if CLIENT_ID == 'YOUR_CLIENT_ID':
        await ws.send(json.dumps({'type':'error',
            'message':'Emotiv credentials not set. Run with --emotiv-id YOUR_ID --emotiv-secret YOUR_SECRET  '
                      'or set env EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SEC. '
                      'Register at emotiv.com/developer → falling back to simulation.'}))
        await stream_sim(ws, detector=detector, cal_holder=cal_holder, exp_holder=exp_holder)
        return

    print(f"[EMOTIV] Connecting to Cortex API: {CORTEX_URL}")
    await ws.send(json.dumps({'type':'status',
        'message':'Connecting to Emotiv Cortex API...'}))

    try:
        import ssl
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode    = ssl.CERT_NONE  # Cortex uses self-signed cert

        async with websockets.connect(CORTEX_URL, ssl=ssl_ctx) as cortex:
            print("[EMOTIV] Connected to Cortex")

            # 1. requestAccess — user must approve in Emotiv App if first time
            await ws.send(json.dumps({'type':'status', 'step':'requestAccess',
                'message':'Step 1/4: Requesting Cortex access… (approve in EmotivApp if prompted)'}))
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'requestAccess','id':1,
                'params':{'clientId': CLIENT_ID,'clientSecret': CLIENT_SEC}
            }))
            resp = json.loads(await cortex.recv())
            granted = resp.get('result',{}).get('accessGranted', False)
            print(f"[EMOTIV] requestAccess: accessGranted={granted}")
            if not granted:
                await ws.send(json.dumps({'type':'status',
                    'message':'Waiting for EmotivApp approval — open EmotivApp and grant access, then retry.'}))
                # Non-fatal: authorize may still succeed if previously granted

            # 2. authorize
            await ws.send(json.dumps({'type':'status', 'step':'authorize',
                'message':'Step 2/4: Authorizing…'}))
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'authorize','id':2,
                'params':{'clientId': CLIENT_ID,'clientSecret': CLIENT_SEC,
                          'debit': 1}
            }))
            resp  = json.loads(await cortex.recv())
            token = resp.get('result',{}).get('cortexToken','')
            if token:
                print(f"[EMOTIV] Token obtained: {token[:20]}…")
            else:
                err_msg = resp.get('error',{}).get('message','unknown')
                print(f"[EMOTIV] Auth failed: {err_msg}")
                raise RuntimeError(f"Cortex authorization failed ({err_msg}). "
                                   "Check Client ID/Secret and ensure EmotivApp is running.")

            # 3. queryHeadsets → pick first
            await ws.send(json.dumps({'type':'status', 'step':'queryHeadsets',
                'message':'Step 3/4: Searching for EPOC X headset…'}))
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'queryHeadsets','id':3,'params':{}
            }))
            resp     = json.loads(await cortex.recv())
            headsets = resp.get('result', [])
            if not headsets:
                raise RuntimeError(
                    'No Emotiv headset detected. Power on your EPOC X, '
                    'pair via Bluetooth, then retry.')
            headset_id = headsets[0]['id']
            battery    = headsets[0].get('settings',{}).get('batteryPercent', None)
            batt_str   = f' · Battery {battery}%' if battery is not None else ''
            print(f"[EMOTIV] Headset: {headset_id}{batt_str}")
            await ws.send(json.dumps({'type':'status',
                'message':f'Headset found: {headset_id}{batt_str}'}))
            if battery is not None:
                await ws.send(json.dumps({'type':'battery', 'pct': battery}))

            # 4. createSession
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'createSession','id':4,
                'params':{'cortexToken': token,'headset': headset_id,
                          'status':'active'}
            }))
            resp       = json.loads(await cortex.recv())
            session_id = resp.get('result',{}).get('id','')
            if not session_id:
                err_msg = resp.get('error',{}).get('message','unknown')
                raise RuntimeError(f'createSession failed: {err_msg}')
            print(f"[EMOTIV] Session: {session_id}")

            # 5. subscribe EEG
            await ws.send(json.dumps({'type':'status', 'step':'subscribe',
                'message':'Step 4/4: Subscribing to EEG stream…'}))
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'subscribe','id':5,
                'params':{'cortexToken': token,'session': session_id,
                          'streams':['eeg']}
            }))
            resp = json.loads(await cortex.recv())
            success_list = resp.get('result',{}).get('success', [])
            cols = success_list[0].get('cols',[]) if success_list else []
            if not cols:
                raise RuntimeError('EEG subscription failed — no column list returned from Cortex')
            print(f"[EMOTIV] EEG columns: {cols}")

            await ws.send(json.dumps({'type':'status', 'step':'streaming',
                'message':f'Emotiv {headset_id} connected — streaming EEG…'}))

            # Map Cortex column order → EMOTIV_CHS order (by name, Cortex column list
            # typically starts with COUNTER/INTERPOLATED then channels in hardware order)
            ch_order = []
            missing  = []
            for name in EMOTIV_CHS:
                idx = next((i for i, c in enumerate(cols)
                            if c.strip().upper() == name.upper()), None)
                ch_order.append(idx)
                if idx is None:
                    missing.append(name)
            if missing:
                print(f"[EMOTIV] Warning: channels not found in Cortex cols: {missing}")
                await ws.send(json.dumps({'type':'status',
                    'message':f'Warning: missing Cortex channels {missing}. Check headset firmware.'}))

            # Sliding-window buffer: emit every EMIT_EVERY samples, compute over
            # up to COMPUTE_WIN samples for better frequency resolution (≥ 0.5s)
            buf          = {i: [] for i in range(14)}
            EMIT_EVERY   = 32    # emit frame every ~0.25 s (32 samples @ 128 Hz)
            COMPUTE_WIN  = 256   # use up to 2 s for bandpower when available
            MIN_WIN      = 64    # minimum 0.5 s before first frame

            async for raw in cortex:
                pkt = json.loads(raw)
                if 'eeg' not in pkt:
                    continue
                row = pkt['eeg']
                for ci, col_i in enumerate(ch_order):
                    # Always append so all 14 channels stay length-synchronized
                    val = float(row[col_i]) if (col_i is not None and col_i < len(row)) else 0.0
                    buf[ci].append(val)

                # Safety cap: discard oldest samples if buffer grows too large
                if len(buf[0]) > COMPUTE_WIN * 2:
                    for i in range(14):
                        buf[i] = buf[i][-COMPUTE_WIN:]

                if len(buf[0]) >= EMIT_EVERY and len(buf[0]) >= MIN_WIN:
                    use_n = min(len(buf[0]), COMPUTE_WIN)
                    # Use last use_n samples (overlapping window for smooth update)
                    seg = np.array([buf[i][-use_n:] for i in range(14)])
                    frame = compute_frame(seg, fs=128, baseline=STATE['baseline'])
                    update_state(frame['channels'], frame['bands'])
                    _add_vpattern(frame, detector, eeg_seg=seg)
                    if cal_holder:
                        await _push_cal(ws, cal_holder, frame['bands'])
                    if exp_holder:
                        await _tick_experiment(ws, exp_holder, frame['bands'])
                    try:
                        await ws.send(json.dumps(frame))
                    except websockets.exceptions.ConnectionClosed:
                        return
                    # Advance buffer by EMIT_EVERY (sliding window)
                    for i in range(14):
                        buf[i] = buf[i][EMIT_EVERY:]

    except OSError as e:
        msg = ('Emotiv App (Cortex) not running at wss://localhost:6868. '
               'Start EmotivApp first, then retry.')
        print(f"[EMOTIV] {msg} ({e})")
        await ws.send(json.dumps({'type':'error', 'message': msg, 'fallback':'sim'}))
        await stream_sim(ws, detector=detector, cal_holder=cal_holder, exp_holder=exp_holder)
    except RuntimeError as e:
        msg = str(e)
        print(f"[EMOTIV] Runtime error: {msg}")
        await ws.send(json.dumps({'type':'error', 'message': msg, 'fallback':'sim'}))
        await stream_sim(ws, detector=detector, cal_holder=cal_holder, exp_holder=exp_holder)
    except Exception as e:
        msg = f'Emotiv connection error: {e}'
        print(f"[EMOTIV] {msg}")
        await ws.send(json.dumps({'type':'error', 'message': msg, 'fallback':'sim'}))
        await stream_sim(ws, detector=detector, cal_holder=cal_holder, exp_holder=exp_holder)

# ── TGAM / NeuroSky — ThinkGear Serial Protocol ───────────────────────────────
async def stream_tgam(ws, serial_port=None, detector=None, cal_holder=None, exp_holder=None):
    """
    Reads TGAM (ThinkGear ASIC Module) via serial port.
    Compatible: MindWave Mobile (BT serial), MindWave USB, raw TGAM UART.

    Requirements: pip install pyserial

    Setup:
      Windows : --serial-port COM3
      Linux   : --serial-port /dev/ttyUSB0
      Mac     : --serial-port /dev/tty.MindWave-DevB

    ThinkGear Communication Protocol (TGCP):
      Sync:     0xAA 0xAA
      Length:   1 byte  (payload size)
      Payload:  N bytes (code-value pairs)
      Checksum: 1 byte  = ~sum(payload) & 0xFF

    Payload codes:
      0x02  Poor Signal    (0=good, 200=no contact)
      0x04  Attention      (eSense 0-100)
      0x05  Meditation     (eSense 0-100)
      0x16  Blink Strength (0-255)
      0x80  Raw EEG        (2 bytes big-endian int16, 512Hz)
      0x83  Band Powers    (24 bytes: 8×3-byte uint — delta,theta,
                            lowAlpha,highAlpha,lowBeta,highBeta,lowGamma,midGamma)
    """
    try:
        import serial
    except ImportError:
        await ws.send(json.dumps({'type': 'error',
            'message': 'pyserial not installed. Run: pip install pyserial'}))
        await stream_sim(ws, detector=detector, cal_holder=cal_holder,
                         exp_holder=exp_holder)
        return

    port = serial_port or '/dev/ttyUSB0'
    print(f"[TGAM] Opening serial port: {port}")
    await ws.send(json.dumps({'type': 'status',
        'message': f'Connecting to TGAM on {port}...'}))

    try:
        ser = serial.Serial(port, baudrate=57600, timeout=1)
        await ws.send(json.dumps({'type': 'status',
            'message': f'TGAM connected on {port}. Streaming...', 'source': 'tgam'}))
        print(f"[TGAM] Connected on {port}")

        attention = 50.0
        meditation = 50.0
        poor_signal = 200
        bands_raw = [0] * 8   # delta,theta,lowA,highA,lowB,highB,lowG,midG
        raw_buf = []

        def parse_payload(payload):
            nonlocal attention, meditation, poor_signal, bands_raw
            i = 0
            while i < len(payload):
                code = payload[i]; i += 1
                if code == 0x02:                     # poor signal
                    poor_signal = payload[i]; i += 1
                elif code == 0x04:                   # attention
                    attention = float(payload[i]); i += 1
                elif code == 0x05:                   # meditation
                    meditation = float(payload[i]); i += 1
                elif code == 0x16:                   # blink
                    i += 1
                elif code == 0x80:                   # raw EEG (2 bytes)
                    if i + 1 < len(payload):
                        raw_val = int.from_bytes(payload[i:i+2], 'big', signed=True)
                        raw_buf.append(raw_val)
                        if len(raw_buf) > 512:
                            raw_buf.pop(0)
                    i += 2
                elif code == 0x83:                   # band powers (24 bytes)
                    for b in range(8):
                        bands_raw[b] = int.from_bytes(payload[i:i+3], 'big')
                        i += 3
                else:
                    break

        async def send_loop():
            while True:
                # Normalise band powers to 0-100 range
                bp_max = max(bands_raw) or 1
                d,th,la,ha,lb,hb,lg,mg = [min(100, v/bp_max*100) for v in bands_raw]
                alpha = (la + ha) / 2
                beta  = (lb + hb) / 2
                gamma = (lg + mg) / 2

                # Concentration from attention eSense (direct from chip)
                conc = attention

                bands_out = {
                    'delta': round(d, 1), 'theta': round(th, 1),
                    'alpha': round(alpha, 1), 'beta': round(beta, 1),
                    'gamma': round(gamma, 1),
                    'concentration': round(conc, 1),
                    'engagement_index': round(beta / (alpha + th + 1e-9), 3),
                    'faa': 0.0,   # single channel — FAA not available
                    'attention': round(attention, 1),
                    'meditation': round(meditation, 1),
                }
                artifacts = {'emg_warn': poor_signal > 50, 'poor_signal': poor_signal}

                vdet = detector
                v_result = vdet.push(conc) if vdet else {'v_prob': 0.0, 'v_active': False}

                if v_result.get('v_active'):
                    print(f"[TGAM] V-pattern! prob={v_result['v_prob']:.2f} conc={conc:.1f}")

                payload_ws = {
                    'type': 'eeg',
                    'channels': [round(conc, 2)],
                    'bands': bands_out,
                    'artifacts': artifacts,
                    'vpattern': v_result,
                    'source': 'tgam'
                }
                await ws.send(json.dumps(payload_ws))
                await asyncio.sleep(0.25)   # 4fps UI update

        send_task = asyncio.create_task(send_loop())

        loop = asyncio.get_event_loop()
        try:
            while True:
                # Read serial in executor to avoid blocking
                data = await loop.run_in_executor(None, lambda: ser.read(256))
                i = 0
                while i < len(data) - 3:
                    if data[i] == 0xAA and data[i+1] == 0xAA:
                        plen = data[i+2]
                        if i + 3 + plen + 1 > len(data):
                            break
                        payload = data[i+3: i+3+plen]
                        chk = data[i+3+plen]
                        if (~sum(payload) & 0xFF) == chk:
                            parse_payload(list(payload))
                        i += 3 + plen + 1
                    else:
                        i += 1
        finally:
            send_task.cancel()
            ser.close()

    except Exception as e:
        print(f"[TGAM] Error: {e}")
        await ws.send(json.dumps({'type': 'error',
            'message': f'TGAM error: {e}. Falling back to simulation.'}))
        await stream_sim(ws, detector=detector, cal_holder=cal_holder,
                         exp_holder=exp_holder)

# ── Muse S Athena — muselsl / pylsl ───────────────────────────────────────────
async def stream_muse(ws, detector=None, cal_holder=None, exp_holder=None):
    """
    Streams live EEG from Muse S Athena via Lab Streaming Layer (LSL).

    Requirements:
      pip install muselsl pylsl
      1. Pair Muse via Bluetooth
      2. Run in a separate terminal: muselsl stream
      3. Then start the server with --source muse

    Channels sent: TP9, AF7, FP1(ref), FP2(ref), AF8, TP10  (6 total)
    Active EEG:    TP9, AF7, AF8, TP10  (indices 0,1,4,5)
    Sampling rate: 256 Hz
    """
    try:
        from pylsl import StreamInlet, resolve_byprop
    except ImportError:
        await ws.send(json.dumps({'type':'error',
            'message':'pylsl not installed. Run: pip install muselsl pylsl'}))
        await stream_sim(ws, detector=detector, cal_holder=cal_holder,
                         exp_holder=exp_holder)
        return

    print("[MUSE] Searching for Muse LSL stream...")
    await ws.send(json.dumps({'type':'status',
        'message':'Searching for Muse S LSL stream... (run: muselsl stream)'}))

    try:
        streams = resolve_byprop('type', 'EEG', timeout=10)
        if not streams:
            raise RuntimeError("No Muse LSL stream found. Run: muselsl stream")
        inlet = StreamInlet(streams[0])
        info  = inlet.info()
        print(f"[MUSE] Connected: {info.name()} @ {info.nominal_srate()}Hz")
        await ws.send(json.dumps({'type':'status',
            'message':f'Muse S connected: {info.name()} @ {info.nominal_srate()}Hz',
            'source': 'muse'}))

        # Buffer for 256-sample window (1s)
        WIN = 256
        buf = []

        while True:
            sample, _ = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue
            buf.append(sample[:6])   # TP9, AF7, FP1, FP2, AF8, TP10
            if len(buf) < WIN:
                continue

            eeg_arr = np.array(buf[-WIN:]).T   # (6, 256)
            eeg_eeg = eeg_arr[MUSE_EEG_IDX]    # (4, 256) active EEG only

            # Preprocess at Muse sample rate (256 Hz); frontal_idx=[1,2] = AF7,AF8 in 4-ch array
            clean, artifacts = preprocess(eeg_eeg, fs=FS_MUSE, notch_hz=50, frontal_idx=[1, 2])

            # compute_frame expects 14ch; use inline 4ch band computation for Muse
            nch4 = clean.shape[0]
            _bp = lambda pos, lo, hi: bandpower(clean[pos], FS_MUSE, lo, hi)
            _g_delta = float(np.mean([_bp(i,1,4)  for i in range(nch4)]))
            _g_theta = float(np.mean([_bp(i,4,8)  for i in range(nch4)]))
            _g_alpha = float(np.mean([_bp(i,8,12) for i in range(nch4)]))
            _g_beta  = float(np.mean([_bp(i,12,30)for i in range(nch4)]))
            _g_gamma = float(np.mean([_bp(i,30,45)for i in range(nch4)]))
            _ei  = _g_beta / max(_g_alpha + _g_theta, 1e-9)
            _tot = max(_g_delta+_g_theta+_g_alpha+_g_beta+_g_gamma, 1e-9)
            _conc = float(np.clip(_ei * 20 + 40, 0, 100))
            # FAA: ln(AF8_alpha) - ln(AF7_alpha)
            # MUSE_EEG order: [TP9, AF7, AF8, TP10] → clean[1]=AF7, clean[2]=AF8
            _faa_l = bandpower(clean[1], FS_MUSE, 8, 12)  # AF7 (left frontal)
            _faa_r = bandpower(clean[2], FS_MUSE, 8, 12)  # AF8 (right frontal)
            _faa = float(np.clip((np.log(_faa_r+1e-9)-np.log(_faa_l+1e-9))/np.log(100), -0.5, 0.5))
            bands_out = {
                'delta':  round(_g_delta/_tot*100),
                'theta':  round(_g_theta/_tot*100),
                'alpha':  round(_g_alpha/_tot*100),
                'beta':   round(_g_beta /_tot*100),
                'gamma':  round(_g_gamma/_tot*100),
                'concentration':    round(_conc, 2),
                'engagement_index': round(_ei,   4),
                'faa':              round(_faa,  4),
                'meditation': None,
            }

            # Build 6-channel vals (ref channels get avg value)
            avg_val = float(np.mean([bands_out['concentration']] * 4))
            channels = [0.0] * 6
            for pos, idx in enumerate(MUSE_EEG_IDX):
                ch_conc = float(np.clip(
                    (bandpower(clean[pos], FS_MUSE, 12, 30) /
                     (bandpower(clean[pos], FS_MUSE, 8, 12) + 1e-9) - 0.3) * 40 + 50, 0, 100))
                channels[idx] = ch_conc
            # ref channels mirror avg
            channels[2] = channels[3] = avg_val

            vdet = detector
            v_result = vdet.push(bands_out['concentration']) if vdet else {'v_prob': 0.0, 'v_active': False}

            payload = {
                'type': 'eeg',
                'channels': [round(v, 2) for v in channels],
                'bands': bands_out,
                'artifacts': artifacts,
                'vpattern': v_result,
                'source': 'muse'
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(0.25)   # ~4fps update

    except Exception as e:
        print(f"[MUSE] Error: {e}")
        await ws.send(json.dumps({'type':'error',
            'message': f'Muse error: {e}. Falling back to simulation.'}))
        await stream_sim(ws, detector=detector, cal_holder=cal_holder,
                         exp_holder=exp_holder)

# ── Settings + Calibration + Experiment receiver ──────────────────────────────
def _make_recv(ws, calibrator_holder, exp_holder):
    """
    calibrator_holder : list[Calibrator|None]
    exp_holder        : list[ExperimentRunner|None]
    Both are mutable via list so recv coroutine can update them.
    """
    async def recv_loop():
        try:
            async for raw in ws:
                try:
                    d = json.loads(raw)
                    msg_type = d.get('type')

                    if msg_type == 'settings':
                        STATE['settings'].update(d['settings'])
                        print(f"[WS] Settings updated: {STATE['settings']}")

                    elif msg_type == 'calibrate_start':
                        if CAL_OK:
                            cal = Calibrator()
                            calibrator_holder[0] = cal
                            status = cal.start()
                            await ws.send(json.dumps({'type': 'calibration', **status}))
                            print("[CAL] Calibration started")
                        else:
                            await ws.send(json.dumps(
                                {'type': 'error', 'message': 'calibration module not found'}))

                    elif msg_type == 'calibrate_load':
                        if CAL_OK:
                            cal = Calibrator.load()
                            if cal:
                                calibrator_holder[0] = cal
                                STATE['settings'].update(cal.to_settings())
                                await ws.send(json.dumps(
                                    {'type': 'calibration', 'state': 'loaded',
                                     'result': cal._result,
                                     'settings': cal.to_settings()}))
                            else:
                                await ws.send(json.dumps(
                                    {'type': 'calibration', 'state': 'not_found'}))

                    elif msg_type == 'exp_start':
                        if EXP_OK:
                            protocol = d.get('protocol', 'short')
                            runner = ExperimentRunner(protocol=protocol)
                            exp_holder[0] = runner
                            status = runner.start()
                            await ws.send(json.dumps({'type': 'experiment', **status}))
                            print(f"[EXP] Started protocol='{protocol}' "
                                  f"({runner.total_duration()}s)")
                        else:
                            await ws.send(json.dumps(
                                {'type': 'error', 'message': 'experiment module not found'}))

                    elif msg_type == 'exp_stop':
                        runner = exp_holder[0]
                        if runner and not runner.done:
                            runner._state = 'stopped'
                            runner.done   = True
                            path = runner.save()
                            await ws.send(json.dumps(
                                {'type': 'experiment', 'state': 'stopped',
                                 'saved': path, 'summary': runner._summary()}))

                    elif msg_type == 'exp_marker':
                        runner = exp_holder[0]
                        if runner and not runner.done:
                            label  = d.get('label', 'manual')
                            marker = runner.add_marker(label, STATE['bands'])
                            await ws.send(json.dumps(
                                {'type': 'exp_marker_ack', **marker}))
                            print(f"[EXP] Marker: {label} @ {marker['elapsed']}s")

                    elif msg_type == 'exp_protocols':
                        await ws.send(json.dumps({
                            'type': 'exp_protocols',
                            'protocols': {
                                k: [{'name': p.name, 'duration': p.duration,
                                     'phase_type': p.phase_type}
                                    for p in phases]
                                for k, phases in (PROTOCOLS.items() if EXP_OK else {})
                            }
                        }))

                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
    return recv_loop

# ── Handler ───────────────────────────────────────────────────────────────────
def make_handler(source, dat_file, trial, speed,
                 notch_hz=50, no_preprocess=False, model_path='models/vpattern_model.pt'):
    STATE['source'] = source
    detector = make_detector(model_path) if MODEL_OK else None

    # Try to auto-load saved calibration at startup
    _startup_cal = None
    if CAL_OK:
        _startup_cal = Calibrator.load()
        if _startup_cal:
            STATE['settings'].update(_startup_cal.to_settings())

    async def handler(ws, path='/'):
        peer = ws.remote_address
        print(f"[WS] Client connected: {peer}")

        calibrator_holder = [_startup_cal]
        exp_holder        = [None]          # active ExperimentRunner

        await ws.send(json.dumps({
            'type': 'state_snapshot',
            'channels': STATE['channels'],
            'bands': STATE['bands'],
            'settings': STATE['settings'],
            'source': STATE['source'],
            'calibrated': _startup_cal is not None,
        }))

        recv_loop = _make_recv(ws, calibrator_holder, exp_holder)
        recv_task = asyncio.create_task(recv_loop())

        async def stream_with_cal(stream_coro):
            """Wrap a stream coroutine to feed bands into active calibrator."""
            # We can't intercept mid-stream easily, so calibration push
            # is done inside each stream via the calibrator_holder reference.
            # Pass calibrator_holder to streams that support it.
            await stream_coro

        try:
            if source == 'deap':
                if not dat_file:
                    await ws.send(json.dumps({'type':'error','message':'--file not specified'}))
                    return
                await stream_deap(ws, dat_file, trial, speed,
                                  notch_hz=notch_hz, no_preprocess=no_preprocess,
                                  detector=detector, cal_holder=calibrator_holder,
                                  exp_holder=exp_holder)
            elif source == 'mental':
                if not dat_file:
                    await ws.send(json.dumps({'type':'error','message':'--file not specified'}))
                    return
                await stream_mental(ws, dat_file, speed,
                                    detector=detector, cal_holder=calibrator_holder,
                                    exp_holder=exp_holder)
            elif source == 'emotiv':
                await stream_emotiv(ws, detector=detector, cal_holder=calibrator_holder,
                                    exp_holder=exp_holder)
            elif source == 'muse':
                await stream_muse(ws, detector=detector, cal_holder=calibrator_holder,
                                  exp_holder=exp_holder)
            elif source == 'tgam':
                await stream_tgam(ws, serial_port=STATE.get('serial_port'),
                                  detector=detector, cal_holder=calibrator_holder,
                                  exp_holder=exp_holder)
            else:
                await stream_sim(ws, detector=detector, cal_holder=calibrator_holder,
                                 exp_holder=exp_holder)
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass
            print(f"[WS] Client disconnected: {peer}")
    return handler

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='EEG WebSocket Server — Brain Concentration Visualization')
    parser.add_argument('--source', choices=['sim','deap','mental','emotiv','muse','tgam'],
                        default='sim', help='Data source')
    parser.add_argument('--serial-port', default=None,
                        help='Serial port for TGAM (e.g. COM3 / /dev/ttyUSB0)')
    parser.add_argument('--file',  default=None, help='Data file path')
    parser.add_argument('--trial', type=int,   default=0,   help='DEAP trial index (0-39)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--port',  type=int,   default=8765, help='WebSocket port')
    parser.add_argument('--notch', type=int,   default=50,  choices=[50,60],
                        help='Power-line notch frequency: 50 (EU, default) or 60 (US/JP)')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Disable bandpass/notch/blink filters (raw signal)')
    parser.add_argument('--model', default='models/vpattern_model.pt',
                        help='Path to trained V-pattern model (default: models/vpattern_model.pt)')
    parser.add_argument('--emotiv-id',     default=None, help='Emotiv Cortex API Client ID')
    parser.add_argument('--emotiv-secret', default=None, help='Emotiv Cortex API Client Secret')
    args = parser.parse_args()

    if not NUMPY_OK and args.source in ('deap','mental'):
        print("[ERROR] numpy/scipy required. pip install numpy scipy")
        return
    if not PANDAS_OK and args.source == 'mental':
        print("[ERROR] pandas required. pip install pandas")
        return

    if args.serial_port:
        STATE['serial_port'] = args.serial_port
    if args.emotiv_id:
        STATE['emotiv_client_id']  = args.emotiv_id
    if args.emotiv_secret:
        STATE['emotiv_client_sec'] = args.emotiv_secret

    handler = make_handler(args.source, args.file, args.trial, args.speed,
                           notch_hz=args.notch, no_preprocess=args.no_preprocess,
                           model_path=args.model)

    print("=" * 60)
    print(" EEG WebSocket Server")
    print(f"  URL        : ws://localhost:{args.port}")
    print(f"  Source     : {args.source.upper()}")
    if args.file:
        print(f"  File       : {args.file}")
    if args.source in ('deap','mental'):
        print(f"  Speed      : {args.speed}x")
    if args.source == 'deap':
        print(f"  Trial      : {args.trial}")
    if not args.no_preprocess:
        print(f"  Preprocess : bandpass(1-45Hz) + notch({args.notch}Hz) + blink rejection")
    else:
        print(f"  Preprocess : DISABLED (raw signal)")
    print("=" * 60)
    if args.source == 'emotiv':
        print(" NOTE: Requires Emotiv App running + Client ID/Secret")
        print("       Edit CLIENT_ID / CLIENT_SECRET in stream_emotiv()")
    if args.source == 'muse':
        print(" NOTE: Requires Muse S paired via Bluetooth + muselsl running")
        print("       pip install muselsl pylsl  →  muselsl stream")
    if args.source == 'tgam':
        port = args.serial_port or '(auto)'
        print(f" NOTE: TGAM serial port: {port}")
        print("       pip install pyserial")
        print("       Windows: --serial-port COM3 | Linux: --serial-port /dev/ttyUSB0")
    print(" Open mockup_3d.html in browser")
    print(" Press Ctrl+C to stop")
    print("=" * 60)

    async def serve():
        async with websockets.serve(handler, 'localhost', args.port):
            await asyncio.Future()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\n[WS] Server stopped")

if __name__ == '__main__':
    main()
