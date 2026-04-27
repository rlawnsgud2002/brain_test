#!/usr/bin/env python3
"""
EEG WebSocket Server for Brain Concentration Visualization
Supports:
  --source sim    : Random simulation (default)
  --source deap   : DEAP dataset .dat file playback
  --source emotiv : Emotiv EPOC X via Cortex API (stub - requires Emotiv App)

Usage:
  pip install websockets numpy scipy
  python eeg_server.py --source sim
  python eeg_server.py --source deap --file data/s01.dat --trial 0 --speed 2.0
  python eeg_server.py --source emotiv

DEAP Dataset: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
  - Download s01.dat ~ s32.dat into ./data/ folder
  - Each file: 40 trials x 40 channels x 8064 samples @ 128Hz
"""

import asyncio
import json
import pickle
import argparse
import random
import math
import time
from typing import Optional

try:
    import numpy as np
    from scipy.signal import welch, butter, filtfilt
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False
    print("[WARN] numpy/scipy not found. Install with: pip install numpy scipy")

try:
    import websockets
    WS_OK = True
except ImportError:
    WS_OK = False
    print("[ERROR] websockets not found. Install with: pip install websockets")
    exit(1)


# ── DEAP Channel Mapping ──────────────────────────────────────────────────────
# DEAP 32 EEG channels (0-indexed): Fp1, AF3, F3, F7, FC5, FC1, C3, T7, CP5,
# CP1, P3, P7, PO3, O1, Oz, Pz, Fp2, AF4, Fz, F4, F8, FC6, FC2, Cz, C4, T8,
# CP6, CP2, P4, P8, PO4, O2
DEAP_CH = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1',
           'P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4',
           'F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']

# Emotiv EPOC X 14 channels → DEAP indices
EMOTIV_TO_DEAP = {
    'AF3':1,'AF4':17,'F7':3,'F3':2,'F4':19,'F8':20,
    'FC5':4,'FC6':21,'T7':7,'T8':25,'P7':11,'P8':29,
    'O1':13,'O2':31
}
EMOTIV_CHS = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2']
DEAP_IDX   = [EMOTIV_TO_DEAP[ch] for ch in EMOTIV_CHS]  # length 14

FS = 128  # DEAP sampling rate (Hz)

# ── Signal Processing ─────────────────────────────────────────────────────────
def bandpower(signal: 'np.ndarray', fs: int, fmin: float, fmax: float) -> float:
    """Welch PSD band power."""
    nperseg = min(len(signal), fs * 2)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.mean(psd[mask])) if mask.any() else 0.0

def compute_frame(eeg_14ch: 'np.ndarray', fs: int = FS) -> dict:
    """
    eeg_14ch: (14, n_samples) raw EEG
    Returns dict with channel normalized values (0-100) and band averages.
    """
    ratios = []
    ch_bands = []
    for ch in eeg_14ch:
        b_delta = bandpower(ch, fs, 1,  4)
        b_theta = bandpower(ch, fs, 4,  8)
        b_alpha = bandpower(ch, fs, 8,  12)
        b_beta  = bandpower(ch, fs, 12, 30)
        b_gamma = bandpower(ch, fs, 30, 45)
        ratio   = b_beta / (b_alpha + 1e-9)
        ratios.append(ratio)
        ch_bands.append((b_delta, b_theta, b_alpha, b_beta, b_gamma))

    # Normalize ratios to 0-100 across channels
    mn, mx = min(ratios), max(ratios)
    if mx > mn:
        norm = [(r - mn) / (mx - mn) * 100 for r in ratios]
    else:
        norm = [50.0] * 14

    # Global band averages
    avg = lambda i: float(np.mean([ch_bands[c][i] for c in range(14)]))
    g_alpha = avg(2)
    g_beta  = avg(3)
    conc = float(np.clip((g_beta / (g_alpha + 1e-9) - 0.3) * 40 + 50, 0, 100))

    return {
        'type': 'eeg',
        'channels': [round(v, 2) for v in norm],
        'bands': {
            'delta': round(avg(0) * 1e6, 2),  # scale to readable range
            'theta': round(avg(1) * 1e6, 2),
            'alpha': round(g_alpha * 1e6, 2),
            'beta':  round(g_beta  * 1e6, 2),
            'gamma': round(avg(4) * 1e6, 2),
            'concentration': round(conc, 2)
        }
    }

# ── DEAP Source ───────────────────────────────────────────────────────────────
async def stream_deap(ws, dat_file: str, trial: int = 0, speed: float = 1.0):
    print(f"[DEAP] Loading {dat_file}, trial={trial}, speed={speed}x")
    with open(dat_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # data['data']: (40, 40, 8064) — trials x channels x samples
    eeg_all = data['data'][trial, :32, :]   # 32 EEG channels
    labels  = data['labels'][trial]         # valence, arousal, dominance, liking
    emotiv  = eeg_all[DEAP_IDX, :]          # (14, 8064)

    n = emotiv.shape[1]
    win  = FS          # 1s window
    step = FS // 4     # 0.25s step → 4 frames/s
    interval = (step / FS) / speed

    print(f"[DEAP] Labels: valence={labels[0]:.1f} arousal={labels[1]:.1f} "
          f"dominance={labels[2]:.1f} liking={labels[3]:.1f}")
    await ws.send(json.dumps({
        'type': 'meta',
        'trial': trial,
        'labels': {'valence': float(labels[0]), 'arousal': float(labels[1]),
                   'dominance': float(labels[2]), 'liking': float(labels[3])},
        'duration': n / FS
    }))

    for start in range(0, n - win, step):
        seg = emotiv[:, start:start + win]
        frame = compute_frame(seg)
        frame['progress'] = round(start / n, 3)
        frame['timestamp'] = start / FS
        try:
            await ws.send(json.dumps(frame))
            await asyncio.sleep(interval)
        except websockets.exceptions.ConnectionClosed:
            break

    await ws.send(json.dumps({'type': 'done', 'trial': trial}))
    print(f"[DEAP] Trial {trial} finished")

# ── Simulation Source ─────────────────────────────────────────────────────────
async def stream_sim(ws):
    print("[SIM] Starting simulation stream")
    vals = [50.0] * 14
    t = 0.0
    while True:
        t += 0.1
        for i in range(14):
            vals[i] += (random.random() - 0.5) * 6
            vals[i] = max(5.0, min(95.0, vals[i]))
        avg = sum(vals) / 14
        focus = avg / 100
        try:
            await ws.send(json.dumps({
                'type': 'eeg',
                'channels': [round(v, 2) for v in vals],
                'bands': {
                    'delta': round(30 + random.gauss(0, 5), 2),
                    'theta': round(40 + random.gauss(0, 7), 2),
                    'alpha': round(60 - focus * 40 + random.gauss(0, 5), 2),
                    'beta':  round(30 + focus * 50 + random.gauss(0, 5), 2),
                    'gamma': round(20 + focus * 30 + random.gauss(0, 4), 2),
                    'concentration': round(avg, 2)
                },
                'progress': -1,
                'timestamp': t
            }))
            await asyncio.sleep(0.1)
        except websockets.exceptions.ConnectionClosed:
            break

# ── Emotiv EPOC X (Cortex API stub) ──────────────────────────────────────────
async def stream_emotiv(ws):
    """
    Emotiv EPOC X requires Emotiv App (EmotivPRO / EmotivLauncher) running
    and the Cortex SDK. This is a stub that guides setup.

    Full integration steps:
      1. Install Emotiv App: https://www.emotiv.com/emotiv-launcher/
      2. pip install cortex  (unofficial: https://github.com/Emotiv/cortex-v2-example)
      3. Replace this stub with Cortex WebSocket client code
         connecting to wss://localhost:6868

    For now, falls back to simulation.
    """
    print("[EMOTIV] Cortex API stub — falling back to simulation")
    print("[EMOTIV] To integrate: see https://github.com/Emotiv/cortex-v2-example")
    await ws.send(json.dumps({
        'type': 'status',
        'message': 'Emotiv Cortex API not configured. Running simulation.',
        'fallback': 'sim'
    }))
    await stream_sim(ws)

# ── Handler ───────────────────────────────────────────────────────────────────
def make_handler(source: str, dat_file: Optional[str], trial: int, speed: float):
    async def handler(ws, path='/'):
        peer = ws.remote_address
        print(f"[WS] Client connected: {peer}")
        try:
            if source == 'deap':
                if not dat_file:
                    await ws.send(json.dumps({'type':'error','message':'--file not specified'}))
                    return
                await stream_deap(ws, dat_file, trial, speed)
            elif source == 'emotiv':
                await stream_emotiv(ws)
            else:
                await stream_sim(ws)
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            print(f"[WS] Client disconnected: {peer}")
    return handler

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='EEG WebSocket Server — Brain Concentration Visualization')
    parser.add_argument('--source', choices=['sim', 'deap', 'emotiv'],
                        default='sim', help='Data source (default: sim)')
    parser.add_argument('--file',  default=None,
                        help='DEAP .dat file path (e.g. data/s01.dat)')
    parser.add_argument('--trial', type=int, default=0,
                        help='DEAP trial index 0-39 (default: 0)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='DEAP playback speed multiplier (default: 1.0)')
    parser.add_argument('--port',  type=int, default=8765,
                        help='WebSocket port (default: 8765)')
    args = parser.parse_args()

    if not NUMPY_OK and args.source == 'deap':
        print("[ERROR] numpy/scipy required for DEAP. pip install numpy scipy")
        return

    handler = make_handler(args.source, args.file, args.trial, args.speed)

    print("=" * 55)
    print(" EEG WebSocket Server")
    print(f"  URL    : ws://localhost:{args.port}")
    print(f"  Source : {args.source.upper()}")
    if args.source == 'deap':
        print(f"  File   : {args.file}")
        print(f"  Trial  : {args.trial} / Speed: {args.speed}x")
    print("=" * 55)
    print(" Open mockup_3d.html, select DEAP or EMOTIV EPOC X")
    print(" Press Ctrl+C to stop")
    print("=" * 55)

    async def serve():
        async with websockets.serve(handler, 'localhost', args.port):
            await asyncio.Future()  # run forever

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\n[WS] Server stopped")

if __name__ == '__main__':
    main()
