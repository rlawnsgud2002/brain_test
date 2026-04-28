"""
EEGNet + LSTM for V-pattern concentration dip detection.

Architecture:
  Input  : (batch, 14, 128)  — 14 channels × 1s window @ 128Hz
  EEGNet : spatial+temporal CNN → (batch, F)  feature vector per window
  LSTM   : sequence of F features (sliding ring buffer) → temporal context
  Output : P(V-pattern)  scalar 0-1

V-pattern definition:
  Concentration drops ≥ DROP_TH points over DROP_WIN seconds,
  then recovers ≥ RISE_TH points within RISE_WIN seconds.
  The trough window is labelled positive.

Fallback (no PyTorch):
  Rule-based detector using the concentration scalar history.
"""

# ── Rule-based detector (always available) ────────────────────────────────────
class VPatternRuleBased:
    """
    Sliding-window rule-based V-pattern detector.
    Works on concentration history (0-100 scalar per frame).
    """
    DROP_TH  = 18   # minimum concentration drop to qualify as V-start
    RISE_TH  = 12   # minimum recovery to confirm V
    DROP_WIN = 16   # frames to look back for the drop  (~4s @ 4fps)
    RISE_WIN = 12   # frames to look ahead for recovery (~3s)
    SMOOTH   = 4    # simple moving average window

    def __init__(self, maxlen=120):
        self._buf  = []        # concentration history
        self._maxlen = maxlen
        self.v_prob    = 0.0
        self.v_active  = False
        self._cooldown = 0

    def push(self, concentration: float) -> dict:
        self._buf.append(float(concentration))
        if len(self._buf) > self._maxlen:
            self._buf.pop(0)
        if self._cooldown > 0:
            self._cooldown -= 1
        return self._detect()

    def _smooth(self, arr):
        w = self.SMOOTH
        return [sum(arr[max(0,i-w):i+1]) / len(arr[max(0,i-w):i+1])
                for i in range(len(arr))]

    def _detect(self) -> dict:
        buf = self._buf
        if len(buf) < self.DROP_WIN + 2:
            return {'v_prob': 0.0, 'v_active': False, 'rule_based': True}

        s = self._smooth(buf)
        n = len(s)
        cur = s[-1]

        # Look for a trough in the last DROP_WIN frames
        window = s[max(0, n - self.DROP_WIN):]
        trough_val  = min(window)
        trough_idx  = window.index(trough_val)
        pre_peak    = max(window[:trough_idx + 1]) if trough_idx > 0 else window[0]
        post_window = window[trough_idx:]
        post_peak   = max(post_window)

        drop    = pre_peak - trough_val
        rise    = post_peak - trough_val
        at_bottom = trough_idx >= len(window) - 3  # trough is recent

        if self._cooldown > 0:
            prob = 0.0
            active = False
        elif drop >= self.DROP_TH and rise >= self.RISE_TH and at_bottom:
            # Full V confirmed
            prob   = min(1.0, (drop / 40.0 + rise / 30.0) / 2)
            active = True
            self._cooldown = self.RISE_WIN
        elif drop >= self.DROP_TH and at_bottom:
            # Drop confirmed, waiting for recovery
            prob   = min(0.6, drop / 40.0)
            active = True
        else:
            prob   = 0.0
            active = False

        self.v_prob   = round(prob, 3)
        self.v_active = active
        return {'v_prob': self.v_prob, 'v_active': self.v_active, 'rule_based': True}


# ── PyTorch EEGNet + LSTM ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


if TORCH_OK:
    class EEGNet(nn.Module):
        """
        Compact EEG feature extractor.
        Input : (batch, 1, n_ch, n_time)
        Output: (batch, feat_dim)

        Reference: Lawhern et al., EEGNet: A Compact CNN for EEG-Based BCIs, 2018
        """
        def __init__(self, n_ch=14, n_time=128, F1=8, D=2, F2=16, drop_p=0.25):
            super().__init__()
            self.n_ch   = n_ch
            self.n_time = n_time
            F2 = F1 * D

            # Block 1: temporal convolution
            self.conv1    = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
            self.bn1      = nn.BatchNorm2d(F1)
            # Block 1: depthwise spatial filter
            self.dw_conv  = nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False)
            self.bn2      = nn.BatchNorm2d(F1*D)
            self.pool1    = nn.AvgPool2d((1, 4))
            self.drop1    = nn.Dropout(drop_p)
            # Block 2: separable temporal convolution
            self.sep_conv = nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False)
            self.pw_conv  = nn.Conv2d(F2, F2, (1, 1),  bias=False)
            self.bn3      = nn.BatchNorm2d(F2)
            self.pool2    = nn.AvgPool2d((1, 8))
            self.drop2    = nn.Dropout(drop_p)

            feat = self._feat_dim()
            self.flat_dim = feat

        def _feat_dim(self):
            with torch.no_grad():
                x = torch.zeros(1, 1, self.n_ch, self.n_time)
                return self._forward_features(x).shape[1]

        def _forward_features(self, x):
            x = F.elu(self.bn1(self.conv1(x)))
            x = F.elu(self.bn2(self.dw_conv(x)))
            x = self.drop1(self.pool1(x))
            x = F.elu(self.bn3(self.pw_conv(self.sep_conv(x))))
            x = self.drop2(self.pool2(x))
            return x.flatten(1)

        def forward(self, x):
            return self._forward_features(x)


    class EEGNetLSTM(nn.Module):
        """
        EEGNet feature extractor + LSTM temporal classifier for V-pattern detection.

        Input : sequence of raw EEG windows (seq_len, batch, 1, n_ch, n_time)
        Output: (batch,) logit for P(V-pattern)
        """
        def __init__(self, n_ch=14, n_time=128, seq_len=10,
                     lstm_hidden=64, lstm_layers=2, drop_p=0.25):
            super().__init__()
            self.seq_len   = seq_len
            self.eegnet    = EEGNet(n_ch, n_time, drop_p=drop_p)
            feat           = self.eegnet.flat_dim
            self.lstm      = nn.LSTM(feat, lstm_hidden, lstm_layers,
                                     batch_first=True, dropout=drop_p)
            self.classifier = nn.Sequential(
                nn.Linear(lstm_hidden, 32),
                nn.ELU(),
                nn.Dropout(drop_p),
                nn.Linear(32, 1)
            )

        def forward(self, x_seq):
            # x_seq: (batch, seq_len, 1, n_ch, n_time)
            B, S, C, H, W = x_seq.shape
            x_flat = x_seq.view(B * S, C, H, W)
            feats  = self.eegnet(x_flat)              # (B*S, feat)
            feats  = feats.view(B, S, -1)             # (B, S, feat)
            out, _ = self.lstm(feats)                 # (B, S, hidden)
            logit  = self.classifier(out[:, -1, :])   # (B, 1)
            return logit.squeeze(1)                   # (B,)

        def predict_proba(self, x_seq):
            self.eval()
            with torch.no_grad():
                return torch.sigmoid(self(x_seq)).cpu().numpy()


    class VPatternML:
        """
        Online V-pattern detector using EEGNetLSTM.
        Maintains a ring buffer of EEG windows for LSTM input.
        Falls back to rule-based if model file not found.
        """
        def __init__(self, model_path='models/vpattern_model.pt',
                     n_ch=14, n_time=128, seq_len=10, device='cpu'):
            import os, numpy as np
            self.seq_len  = seq_len
            self.n_ch     = n_ch
            self.n_time   = n_time
            self.device   = torch.device(device)
            self._ring    = []          # list of (1, n_ch, n_time) tensors
            self.v_prob   = 0.0
            self.v_active = False
            self._fallback = VPatternRuleBased()

            self.model = EEGNetLSTM(n_ch, n_time, seq_len)
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state'])
                self.model.to(self.device)
                self.model.eval()
                self._model_loaded = True
                print(f"[ML] Loaded V-pattern model: {model_path}")
            else:
                self._model_loaded = False
                print(f"[ML] Model not found ({model_path}). Using rule-based fallback.")
                print(f"[ML] Train with: python train_model.py --source deap --file data/s01.dat")

        def push(self, eeg_window, concentration: float) -> dict:
            """
            eeg_window : np.ndarray (14, n_time) cleaned EEG
            concentration : scalar 0-100
            """
            import numpy as np

            # Always update rule-based (used as fallback or cross-check)
            rb = self._fallback.push(concentration)

            if not self._model_loaded:
                return {**rb, 'rule_based': True}

            # Accumulate ring buffer
            t = torch.tensor(eeg_window, dtype=torch.float32).unsqueeze(0)  # (1,14,128)
            self._ring.append(t)
            if len(self._ring) > self.seq_len:
                self._ring.pop(0)

            if len(self._ring) < self.seq_len:
                return {'v_prob': 0.0, 'v_active': False, 'rule_based': False}

            # (1, seq_len, 1, 14, 128)
            seq = torch.stack(self._ring, dim=0).unsqueeze(0).unsqueeze(2).to(self.device)
            prob = float(self.model.predict_proba(seq)[0])

            self.v_prob   = round(prob, 3)
            self.v_active = prob > 0.5
            return {'v_prob': self.v_prob, 'v_active': self.v_active, 'rule_based': False}


def make_detector(model_path='models/vpattern_model.pt', device='cpu'):
    """Factory: returns ML detector if PyTorch available, else rule-based."""
    if TORCH_OK:
        return VPatternML(model_path=model_path, device=device)
    print("[ML] PyTorch not installed. Using rule-based V-pattern detector.")
    print("[ML] Install with: pip install torch")
    return VPatternRuleBased()
