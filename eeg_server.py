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
from typing import Optional

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
              'beta': 50.0, 'gamma': 30.0, 'concentration': 50.0},
    'settings': {'thLow': 30, 'thHigh': 70, 'dt': 3, 'slope': 1.5},
    'source': 'sim'
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

_FILTERS = None  # initialized on first use

def get_filters():
    global _FILTERS
    if _FILTERS is None:
        _FILTERS = _make_filters(FS)
    return _FILTERS

def preprocess(eeg_14ch, notch_hz=50, blink_thresh_uv=80.0):
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
    (b_bp, a_bp), (b_n50, a_n50), (b_n60, a_n60) = get_filters()
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
    # Blinks appear as large-amplitude, short-duration (~200ms) spikes
    blink_win  = max(1, int(FS * 0.2))   # 200 ms window
    blink_count   = 0
    rejected_samp = 0

    for fi in FRONTAL_IDX:
        sig = clean[fi]
        t = 0
        while t < n_samples - blink_win:
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

    rejected_ratio = round(rejected_samp / (n_samples * len(FRONTAL_IDX) + 1e-9), 3)
    return clean, {'blinks': blink_count, 'rejected_ratio': rejected_ratio}

def bandpower(signal, fs, fmin, fmax):
    nperseg = min(len(signal), fs * 2)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.mean(psd[mask])) if mask.any() else 0.0

def compute_frame(eeg_14ch, fs=FS, apply_preprocess=True, notch_hz=50):
    # Signal processing pipeline
    artifacts = {'blinks': 0, 'rejected_ratio': 0.0}
    if apply_preprocess and NUMPY_OK:
        eeg_14ch, artifacts = preprocess(eeg_14ch, notch_hz=notch_hz)

    ratios, ch_bands = [], []
    for ch in eeg_14ch:
        b_delta = bandpower(ch, fs, 1,  4)
        b_theta = bandpower(ch, fs, 4,  8)
        b_alpha = bandpower(ch, fs, 8,  12)
        b_beta  = bandpower(ch, fs, 12, 30)
        b_gamma = bandpower(ch, fs, 30, 45)
        ratios.append(b_beta / (b_alpha + 1e-9))
        ch_bands.append((b_delta, b_theta, b_alpha, b_beta, b_gamma))

    mn, mx = min(ratios), max(ratios)
    norm = [(r-mn)/(mx-mn)*100 for r in ratios] if mx > mn else [50.0]*14

    avg = lambda i: float(np.mean([ch_bands[c][i] for c in range(14)]))
    g_alpha, g_beta = avg(2), avg(3)
    conc = float(np.clip((g_beta/(g_alpha+1e-9)-0.3)*40+50, 0, 100))

    return {
        'type': 'eeg',
        'channels': [round(v, 2) for v in norm],
        'artifacts': artifacts,
        'bands': {
            'delta': round(avg(0)*1e6, 2),
            'theta': round(avg(1)*1e6, 2),
            'alpha': round(g_alpha*1e6, 2),
            'beta':  round(g_beta*1e6,  2),
            'gamma': round(avg(4)*1e6, 2),
            'concentration': round(conc, 2)
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
async def stream_deap(ws, dat_file, trial=0, speed=1.0, notch_hz=50, no_preprocess=False):
    print(f"[DEAP] Loading {dat_file}, trial={trial}, speed={speed}x")
    with open(dat_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

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
        frame = compute_frame(seg, notch_hz=notch_hz,
                              apply_preprocess=not no_preprocess)
        frame['progress']  = round(start/n, 3)
        frame['timestamp'] = start/FS
        update_state(frame['channels'], frame['bands'])
        try:
            await ws.send(json.dumps(frame))
            await asyncio.sleep(interval)
        except websockets.exceptions.ConnectionClosed:
            break

    await ws.send(json.dumps({'type': 'done', 'trial': trial}))
    print(f"[DEAP] Trial {trial} finished")

# ── Mental State CSV Source ───────────────────────────────────────────────────
async def stream_mental(ws, csv_file, speed=1.0):
    """
    Kaggle EEG Brainwave Dataset - Mental State
    Columns: attention, mediation, delta, theta, lowAlpha, highAlpha,
             lowBeta, highBeta, lowGamma, highGamma, label
    label: 0=focused, 1=unfocused, 2=drowsy  (or string variants)
    """
    if not PANDAS_OK:
        await ws.send(json.dumps({'type':'error','message':'pandas required: pip install pandas'}))
        return

    print(f"[MENTAL] Loading {csv_file}, speed={speed}x")
    df = pd.read_csv(csv_file)
    print(f"[MENTAL] Columns: {list(df.columns)}")
    print(f"[MENTAL] Rows: {len(df)}")

    # Normalize column names (lowercase, strip spaces)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect band power columns
    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_delta = find_col(['delta'])
    col_theta = find_col(['theta'])
    col_alpha = find_col(['highalpha', 'alpha'])
    col_beta  = find_col(['highbeta',  'beta'])
    col_gamma = find_col(['highgamma', 'gamma'])
    col_label = find_col(['label'])

    if not all([col_delta, col_theta, col_alpha, col_beta]):
        await ws.send(json.dumps({'type':'error',
            'message':f'Cannot find band columns. Found: {list(df.columns)}'}))
        return

    # Detect label encoding
    label_names = {0:'FOCUSED', 1:'UNFOCUSED', 2:'DROWSY'}
    if col_label and df[col_label].dtype == object:
        uniq = df[col_label].str.strip().str.upper().unique()
        print(f"[MENTAL] Labels found: {uniq}")
        lmap = {}
        for u in uniq:
            if 'FOCUS' in u or 'CONCENTRAT' in u:  lmap[u] = 0
            elif 'RELAX' in u or 'UNFOCUS' in u:    lmap[u] = 1
            else:                                    lmap[u] = 2
        df['_label_int'] = df[col_label].str.strip().str.upper().map(lmap).fillna(1)
    elif col_label:
        df['_label_int'] = df[col_label]
    else:
        df['_label_int'] = 1

    n = len(df)
    await ws.send(json.dumps({
        'type': 'meta',
        'source': 'mental_state',
        'rows': n,
        'columns': list(df.columns),
        'duration': n * 0.5
    }))

    interval = 0.5 / speed  # ~2 rows/s at real speed

    for i, row in df.iterrows():
        delta = float(row[col_delta])
        theta = float(row[col_theta])
        alpha = float(row[col_alpha])
        beta  = float(row[col_beta])
        gamma = float(row[col_gamma]) if col_gamma else alpha * 0.5
        lbl   = int(row['_label_int'])

        # Normalize band powers to 0-100 range within session
        b_sum = delta + theta + alpha + beta + gamma + 1e-9
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

        try:
            await ws.send(json.dumps(frame))
            await asyncio.sleep(interval)
        except websockets.exceptions.ConnectionClosed:
            break

    await ws.send(json.dumps({'type': 'done', 'source': 'mental_state'}))
    print("[MENTAL] Finished playback")

# ── Simulation Source ─────────────────────────────────────────────────────────
async def stream_sim(ws):
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
        try:
            await ws.send(json.dumps({
                'type': 'eeg', 'channels': STATE['channels'],
                'bands': b, 'progress': -1, 'timestamp': round(t, 2)
            }))
            await asyncio.sleep(0.1)
        except websockets.exceptions.ConnectionClosed:
            break

# ── Emotiv EPOC X — Cortex API ────────────────────────────────────────────────
async def stream_emotiv(ws):
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
    CLIENT_ID  = 'YOUR_CLIENT_ID'    # replace after Emotiv developer registration
    CLIENT_SEC = 'YOUR_CLIENT_SECRET'

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

            # 1. requestAccess
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'requestAccess','id':1,
                'params':{'clientId': CLIENT_ID,'clientSecret': CLIENT_SEC}
            }))
            resp = json.loads(await cortex.recv())
            print(f"[EMOTIV] requestAccess: {resp.get('result',{}).get('accessGranted')}")

            # 2. authorize
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'authorize','id':2,
                'params':{'clientId': CLIENT_ID,'clientSecret': CLIENT_SEC,
                          'debit': 1}
            }))
            resp  = json.loads(await cortex.recv())
            token = resp.get('result',{}).get('cortexToken','')
            print(f"[EMOTIV] Token: {token[:20]}..." if token else "[EMOTIV] Auth failed")

            if not token:
                raise RuntimeError("Cortex authorization failed")

            # 3. queryHeadsets → pick first
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'queryHeadsets','id':3,'params':{}
            }))
            resp     = json.loads(await cortex.recv())
            headsets = resp.get('result', [])
            if not headsets:
                raise RuntimeError("No Emotiv headset found")
            headset_id = headsets[0]['id']
            print(f"[EMOTIV] Headset: {headset_id}")

            # 4. createSession
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'createSession','id':4,
                'params':{'cortexToken': token,'headset': headset_id,
                          'status':'active'}
            }))
            resp       = json.loads(await cortex.recv())
            session_id = resp.get('result',{}).get('id','')
            print(f"[EMOTIV] Session: {session_id}")

            # 5. subscribe EEG
            await cortex.send(json.dumps({
                'jsonrpc':'2.0','method':'subscribe','id':5,
                'params':{'cortexToken': token,'session': session_id,
                          'streams':['eeg']}
            }))
            resp = json.loads(await cortex.recv())
            cols = resp.get('result',{}).get('success',[{}])[0].get('cols',[])
            print(f"[EMOTIV] EEG columns: {cols}")

            await ws.send(json.dumps({'type':'status',
                'message':f'Emotiv {headset_id} connected. Streaming EEG...'}))

            # Map Cortex column order to EMOTIV_CHS order
            ch_order = []
            for name in EMOTIV_CHS:
                if name in cols:
                    ch_order.append(cols.index(name))
                else:
                    ch_order.append(None)

            # Stream EEG frames
            buf = {i: [] for i in range(14)}
            WINDOW = 32  # accumulate 32 samples (~0.25s) then emit

            async for raw in cortex:
                pkt = json.loads(raw)
                if 'eeg' not in pkt:
                    continue
                row = pkt['eeg']
                for ci, col_i in enumerate(ch_order):
                    if col_i is not None and col_i < len(row):
                        buf[ci].append(float(row[col_i]))

                if len(buf[0]) >= WINDOW:
                    seg = np.array([buf[i][:WINDOW] for i in range(14)])
                    frame = compute_frame(seg, fs=128)
                    update_state(frame['channels'], frame['bands'])
                    try:
                        await ws.send(json.dumps(frame))
                    except websockets.exceptions.ConnectionClosed:
                        return
                    for i in range(14):
                        buf[i] = buf[i][WINDOW:]

    except OSError:
        print("[EMOTIV] Cortex not running — falling back to simulation")
        await ws.send(json.dumps({'type':'status',
            'message':'Emotiv App not running (wss://localhost:6868). Falling back to simulation.',
            'fallback':'sim'}))
        await stream_sim(ws)
    except Exception as e:
        print(f"[EMOTIV] Error: {e} — falling back to simulation")
        await ws.send(json.dumps({'type':'status',
            'message':f'Emotiv error: {e}. Falling back to simulation.',
            'fallback':'sim'}))
        await stream_sim(ws)

# ── Settings receiver ─────────────────────────────────────────────────────────
async def recv_settings(ws):
    try:
        async for raw in ws:
            try:
                d = json.loads(raw)
                if d.get('type') == 'settings':
                    STATE['settings'].update(d['settings'])
                    print(f"[WS] Settings updated: {STATE['settings']}")
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass

# ── Handler ───────────────────────────────────────────────────────────────────
def make_handler(source, dat_file, trial, speed, notch_hz=50, no_preprocess=False):
    STATE['source'] = source

    async def handler(ws, path='/'):
        peer = ws.remote_address
        print(f"[WS] Client connected: {peer}")

        await ws.send(json.dumps({
            'type': 'state_snapshot',
            'channels': STATE['channels'],
            'bands': STATE['bands'],
            'settings': STATE['settings'],
            'source': STATE['source']
        }))

        recv_task = asyncio.create_task(recv_settings(ws))

        try:
            if source == 'deap':
                if not dat_file:
                    await ws.send(json.dumps({'type':'error','message':'--file not specified'}))
                    return
                await stream_deap(ws, dat_file, trial, speed,
                                  notch_hz=notch_hz, no_preprocess=no_preprocess)
            elif source == 'mental':
                if not dat_file:
                    await ws.send(json.dumps({'type':'error','message':'--file not specified'}))
                    return
                await stream_mental(ws, dat_file, speed)
            elif source == 'emotiv':
                await stream_emotiv(ws)
            else:
                await stream_sim(ws)
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
    parser.add_argument('--source', choices=['sim','deap','mental','emotiv'],
                        default='sim', help='Data source')
    parser.add_argument('--file',  default=None, help='Data file path')
    parser.add_argument('--trial', type=int,   default=0,   help='DEAP trial index (0-39)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--port',  type=int,   default=8765, help='WebSocket port')
    parser.add_argument('--notch', type=int,   default=50,  choices=[50,60],
                        help='Power-line notch frequency: 50 (EU, default) or 60 (US/JP)')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Disable bandpass/notch/blink filters (raw signal)')
    args = parser.parse_args()

    if not NUMPY_OK and args.source in ('deap','mental'):
        print("[ERROR] numpy/scipy required. pip install numpy scipy")
        return
    if not PANDAS_OK and args.source == 'mental':
        print("[ERROR] pandas required. pip install pandas")
        return

    handler = make_handler(args.source, args.file, args.trial, args.speed,
                           notch_hz=args.notch, no_preprocess=args.no_preprocess)

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
