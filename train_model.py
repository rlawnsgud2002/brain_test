#!/usr/bin/env python3
"""
V-pattern detection model training script.

Generates V-pattern labels from DEAP or Mental State data,
trains EEGNet+LSTM, saves to models/vpattern_model.pt

Usage:
  pip install torch numpy scipy
  python train_model.py --source deap   --file data/s01.dat
  python train_model.py --source mental --file data/mental-state.csv
  python train_model.py --source deap   --file data/s01.dat --trials 0,1,2,3
  python train_model.py --help
"""

import os, sys, argparse, pickle, random
import numpy as np

# ── Check deps ────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
except ImportError:
    sys.exit("[ERROR] PyTorch required: pip install torch")

try:
    from scipy.signal import butter, filtfilt, iirnotch, welch
except ImportError:
    sys.exit("[ERROR] scipy required: pip install scipy")

from eeg_model import EEGNetLSTM
sys.path.insert(0, os.path.dirname(__file__))
from eeg_server import preprocess, bandpower, EMOTIV_CHS, EMOTIV_TO_DEAP, FS

# ── Config ────────────────────────────────────────────────────────────────────
WIN_SAMPLES = 128       # 1s window @ 128Hz
HOP_SAMPLES = 32        # 0.25s hop
SEQ_LEN     = 10        # LSTM sequence length (10 windows = 2.5s)
BATCH       = 32
EPOCHS      = 40
LR          = 1e-3
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# V-pattern labelling thresholds (on concentration 0-100)
DROP_TH   = 10   # minimum drop in concentration to qualify
RISE_TH   = 8    # minimum recovery after trough
DROP_WIN  = 40   # frames (~10s) to measure the drop
RISE_WIN  = 30   # frames (~7.5s) to measure the recovery

EMOTIV_IDX = [EMOTIV_TO_DEAP[ch] for ch in EMOTIV_CHS]


# ── Signal utils ──────────────────────────────────────────────────────────────
def compute_concentration(eeg_14ch):
    """Compute per-window concentration scalar from cleaned EEG."""
    alphas, betas = [], []
    for ch in eeg_14ch:
        alphas.append(bandpower(ch, FS, 8,  12))
        betas.append( bandpower(ch, FS, 12, 30))
    g_alpha = np.mean(alphas)
    g_beta  = np.mean(betas)
    return float(np.clip((g_beta / (g_alpha + 1e-9) - 0.3) * 40 + 50, 0, 100))


def label_vpattern(conc_series, smooth_w=4):
    """
    Generate binary V-pattern labels from concentration time series.
    Returns np.ndarray of shape (n,) with 1 at trough of V, else 0.
    """
    n = len(conc_series)
    # Smooth
    s = np.convolve(conc_series, np.ones(smooth_w)/smooth_w, mode='same')
    labels = np.zeros(n, dtype=np.float32)

    for i in range(DROP_WIN, n - RISE_WIN):
        trough = s[i]
        pre    = np.max(s[i - DROP_WIN: i])
        post   = np.max(s[i: i + RISE_WIN])
        drop   = pre - trough
        rise   = post - trough
        if drop >= DROP_TH and rise >= RISE_TH:
            labels[i] = 1.0

    return labels


# ── DEAP loader ───────────────────────────────────────────────────────────────
def load_deap(dat_file, trials=None, notch_hz=50):
    """
    Returns list of (eeg_14ch_windows, labels) tuples.
    eeg_14ch_windows : (n_windows, 14, WIN_SAMPLES)
    labels           : (n_windows,)
    """
    print(f"[DATA] Loading DEAP: {dat_file}")
    with open(dat_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    all_X, all_y = [], []
    n_trials = data['data'].shape[0]
    trial_list = trials if trials else list(range(n_trials))

    for t in trial_list:
        eeg_full = data['data'][t, :32, :]   # (32, 8064)
        emotiv   = eeg_full[EMOTIV_IDX, :]   # (14, 8064)

        # Preprocess full trial
        clean, _ = preprocess(emotiv, notch_hz=notch_hz)

        # Slide windows
        n = clean.shape[1]
        windows, concs = [], []
        for start in range(0, n - WIN_SAMPLES, HOP_SAMPLES):
            seg = clean[:, start: start + WIN_SAMPLES]
            windows.append(seg)
            concs.append(compute_concentration(seg))

        if not windows:
            continue

        windows = np.stack(windows)          # (n_win, 14, 128)
        concs   = np.array(concs)
        labels  = label_vpattern(concs)

        all_X.append(windows)
        all_y.append(labels)
        n_pos = int(labels.sum())
        print(f"  Trial {t:02d}: {len(labels)} windows, {n_pos} V-pattern ({n_pos/len(labels)*100:.1f}%)")

    return np.concatenate(all_X), np.concatenate(all_y)


# ── Mental State CSV loader ───────────────────────────────────────────────────
def load_mental(csv_file):
    """
    Returns (X, y) where X is (n, 14, WIN_SAMPLES) synthesised from band powers
    and y is (n,) V-pattern labels derived from concentration series.
    """
    try:
        import pandas as pd
    except ImportError:
        sys.exit("[ERROR] pandas required for mental source: pip install pandas")

    print(f"[DATA] Loading Mental State CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    df.columns = [c.strip().lower() for c in df.columns]

    def find(candidates):
        for c in candidates:
            if c in df.columns: return c
        return None

    col_alpha = find(['highalpha','alpha'])
    col_beta  = find(['highbeta', 'beta'])
    col_delta = find(['delta'])
    col_theta = find(['theta'])
    col_gamma = find(['highgamma','gamma'])

    if not all([col_alpha, col_beta]):
        sys.exit(f"[ERROR] Cannot find alpha/beta columns. Found: {list(df.columns)}")

    # Compute concentration per row
    alphas = df[col_alpha].values.astype(float)
    betas  = df[col_beta].values.astype(float)
    concs  = np.clip((betas / (alphas + 1e-9) - 0.3) * 40 + 50, 0, 100)
    labels = label_vpattern(concs)

    # Synthesise 14-channel windows from band power scalars
    SPATIAL_W = np.array([1.2,1.2, 1.1,1.1,1.1,1.1, 1.0,1.0,
                           0.8,0.8, 0.7,0.7, 0.6,0.6])
    n = len(df)
    X = np.zeros((n, 14, WIN_SAMPLES), dtype=np.float32)
    for i in range(n):
        a = alphas[i]; b = betas[i]
        ratio = b / (a + 1e-9)
        base  = float(np.clip((ratio - 0.3) * 40 + 50, 5, 95))
        for ci, w in enumerate(SPATIAL_W):
            # Simple sinusoidal surrogate at alpha/beta frequencies
            t_ax = np.linspace(0, 1, WIN_SAMPLES)
            sig  = (np.sqrt(a) * np.sin(2*np.pi*10*t_ax)     # alpha 10Hz
                  + np.sqrt(b) * np.sin(2*np.pi*20*t_ax)     # beta  20Hz
                  + np.random.randn(WIN_SAMPLES) * 2)
            sig  = sig * w
            X[i, ci] = sig

    n_pos = int(labels.sum())
    print(f"[DATA] {n} samples, {n_pos} V-pattern ({n_pos/n*100:.1f}%)")
    return X, labels


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class VPatternDataset(Dataset):
    """
    Wraps (X, y) arrays into sequence samples for LSTM.
    Each item: x_seq (SEQ_LEN, 1, 14, 128), label scalar.
    """
    def __init__(self, X, y, seq_len=SEQ_LEN):
        self.X   = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (n,1,14,128)
        self.y   = torch.tensor(y, dtype=torch.float32)
        self.seq = seq_len
        # Only use indices where a full sequence is available
        self.idx = list(range(seq_len - 1, len(X)))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        end   = self.idx[i] + 1
        start = end - self.seq
        x_seq = self.X[start:end]          # (seq_len, 1, 14, 128)
        label = self.y[self.idx[i]]
        return x_seq, label


# ── Training loop ─────────────────────────────────────────────────────────────
def train(X, y, output_path, epochs=EPOCHS, lr=LR, batch=BATCH, device=DEVICE):
    print(f"\n[TRAIN] Device: {device}")
    print(f"[TRAIN] Dataset: {len(X)} windows, "
          f"{int(y.sum())} positive ({y.mean()*100:.1f}%)")

    dataset = VPatternDataset(X, y)
    n_val   = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Class-weighted loss to handle imbalance
    pos_weight = torch.tensor([(1 - y.mean()) / (y.mean() + 1e-6)]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

    model = EEGNetLSTM(n_ch=14, n_time=WIN_SAMPLES, seq_len=SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for x_seq, labels in train_dl:
            x_seq  = x_seq.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x_seq)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for x_seq, labels in val_dl:
                x_seq  = x_seq.to(device)
                labels = labels.to(device)
                logits = model(x_seq)
                loss   = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = (torch.sigmoid(logits) > 0.5).float()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
        val_loss /= len(val_ds)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} "
                  f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save ──
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save({
        'model_state': best_state,
        'config': {'n_ch': 14, 'n_time': WIN_SAMPLES, 'seq_len': SEQ_LEN},
        'val_loss': best_val_loss
    }, output_path)
    print(f"\n[TRAIN] Saved → {output_path}  (val_loss={best_val_loss:.4f})")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Train EEGNet+LSTM V-pattern detector')
    parser.add_argument('--source', choices=['deap','mental'], default='deap')
    parser.add_argument('--file',   required=True, help='Data file path')
    parser.add_argument('--trials', default=None,
                        help='Comma-separated DEAP trial indices (default: all)')
    parser.add_argument('--notch',  type=int, default=50, choices=[50,60])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr',     type=float, default=LR)
    parser.add_argument('--batch',  type=int, default=BATCH)
    parser.add_argument('--output', default='models/vpattern_model.pt')
    parser.add_argument('--device', default=DEVICE)
    args = parser.parse_args()

    trials = None
    if args.trials:
        trials = [int(t) for t in args.trials.split(',')]

    if args.source == 'deap':
        X, y = load_deap(args.file, trials=trials, notch_hz=args.notch)
    else:
        X, y = load_mental(args.file)

    train(X, y, args.output,
          epochs=args.epochs, lr=args.lr, batch=args.batch, device=args.device)


if __name__ == '__main__':
    main()
