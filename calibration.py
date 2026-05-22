"""
Personal EEG Calibration Module

Protocol:
  Phase 1 (15s) — RELAX  : eyes closed, breathe slowly
  Phase 2 (15s) — FOCUS  : mental arithmetic or counting backwards
  → Computes personal baselines and auto-adjusts thresholds

Usage (standalone test):
  python calibration.py

Integration with server:
  from calibration import Calibrator
  cal = Calibrator()
  cal.push(bands_dict)           # feed EEG bands each frame
  if cal.done:
      thresholds = cal.thresholds()
      settings   = cal.to_settings()
"""

import json
import os
import time

CALIBRATION_SECS  = 30   # total calibration duration
RELAX_SECS        = 15   # first half: relax
FOCUS_SECS        = 15   # second half: focus
SAVE_PATH         = 'calibration.json'


class Calibrator:
    """
    Collects EEG band-power frames during relax and focus phases,
    then derives personalized thresholds for the visualization.

    State machine:
      idle → relax → focus → done
    """

    def __init__(self, relax_secs=RELAX_SECS, focus_secs=FOCUS_SECS,
                 save_path=SAVE_PATH):
        self.relax_secs = relax_secs
        self.focus_secs = focus_secs
        self.save_path  = save_path

        self._state       = 'idle'   # idle | relax | focus | done
        self._start_ts    = None
        self._phase_ts    = None

        self._relax_buf   = []       # list of band dicts
        self._focus_buf   = []

        self._result      = None     # filled after done
        self.done         = False

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Begin calibration. Call once when user initiates."""
        self._state    = 'relax'
        self._start_ts = time.time()
        self._phase_ts = time.time()
        self._relax_buf.clear()
        self._focus_buf.clear()
        self._result   = None
        self.done      = False
        return {'phase': 'relax', 'duration': self.relax_secs,
                'message': '눈을 감고 편안하게 호흡하세요.'}

    def push(self, bands: dict) -> dict:
        """
        Feed one EEG frame (bands dict with delta/theta/alpha/beta/gamma/concentration).
        Returns current calibration status dict.
        """
        if self._state == 'idle':
            return {'state': 'idle'}

        if self._state == 'done':
            return {'state': 'done', 'result': self._result}

        now     = time.time()
        elapsed = now - self._phase_ts

        if self._state == 'relax':
            self._relax_buf.append(bands)
            remaining = max(0.0, self.relax_secs - elapsed)

            if elapsed >= self.relax_secs:
                self._state    = 'focus'
                self._phase_ts = now
                return {'state': 'focus', 'remaining': 0,
                        'message': '집중하세요. 속으로 숫자를 세거나 암산하세요.',
                        'transition': True}

            return {'state': 'relax', 'remaining': round(remaining, 1),
                    'progress': round(elapsed / self.relax_secs, 2)}

        if self._state == 'focus':
            self._focus_buf.append(bands)
            remaining = max(0.0, self.focus_secs - elapsed)

            if elapsed >= self.focus_secs:
                self._state = 'done'
                self.done   = True
                self._result = self._compute()
                self._save()
                return {'state': 'done', 'result': self._result}

            return {'state': 'focus', 'remaining': round(remaining, 1),
                    'progress': round(elapsed / self.focus_secs, 2)}

        return {'state': self._state}

    def thresholds(self) -> dict:
        """Return computed thresholds (call after done=True)."""
        if not self._result:
            return {}
        return self._result['thresholds']

    def to_settings(self) -> dict:
        """
        Returns settings dict compatible with eeg_server STATE['settings'].
        Merge into server settings via WebSocket settings message.
        """
        if not self._result:
            return {}
        t = self._result['thresholds']
        return {
            'thLow':  t['th_low'],
            'thHigh': t['th_high'],
            'slope':  t['slope'],
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _avg(self, buf, key):
        # Treat missing keys as 0 so denominator stays consistent across keys
        if not buf:
            return 0.0
        return sum(b.get(key, 0) for b in buf) / len(buf)

    def _compute(self) -> dict:
        """
        Derive personalized metrics from relax/focus buffers.

        Thresholds:
          th_low  = relax_mean + (focus_mean - relax_mean) * 0.3
          th_high = relax_mean + (focus_mean - relax_mean) * 0.7
          slope   = 1.0 + (focus_mean - relax_mean) / 50
        """
        rb, fb = self._relax_buf, self._focus_buf
        n_relax = len(rb)
        n_focus = len(fb)
        if n_relax == 0 or n_focus == 0:
            print(f"[CAL] Warning: degenerate calibration — "
                  f"relax={n_relax} frames, focus={n_focus} frames. "
                  f"Thresholds will be clamped to defaults.")

        # Concentration scalars
        relax_conc = [b.get('concentration', 50) for b in rb]
        focus_conc = [b.get('concentration', 50) for b in fb]

        r_mean = sum(relax_conc) / max(1, len(relax_conc))
        f_mean = sum(focus_conc) / max(1, len(focus_conc))
        r_std  = _std(relax_conc)
        f_std  = _std(focus_conc)

        spread = f_mean - r_mean

        th_low  = round(r_mean + spread * 0.30, 1)
        th_high = round(r_mean + spread * 0.70, 1)
        slope   = round(max(0.5, min(3.0, 1.0 + spread / 50.0)), 2)

        # Clamp to sane range
        th_low  = max(10.0, min(45.0, th_low))
        th_high = max(55.0, min(90.0, th_high))

        # Band baselines
        bands_relax = {k: self._avg(rb, k)
                       for k in ('delta','theta','alpha','beta','gamma')}
        bands_focus = {k: self._avg(fb, k)
                       for k in ('delta','theta','alpha','beta','gamma')}

        # Engagement index: beta / (alpha + theta); clamp to sensible range to avoid extremes
        def engagement(bands):
            denom = bands['alpha'] + bands['theta'] + 1e-9
            ratio = bands['beta'] / denom
            return round(max(0.0, min(10.0, ratio)), 3)

        result = {
            'relax': {
                'n_frames':    n_relax,
                'conc_mean':   round(r_mean, 1),
                'conc_std':    round(r_std,  1),
                'bands':       {k: round(v, 2) for k, v in bands_relax.items()},
                'engagement':  engagement(bands_relax),
            },
            'focus': {
                'n_frames':    n_focus,
                'conc_mean':   round(f_mean, 1),
                'conc_std':    round(f_std,  1),
                'bands':       {k: round(v, 2) for k, v in bands_focus.items()},
                'engagement':  engagement(bands_focus),
            },
            'thresholds': {
                'th_low':   th_low,
                'th_high':  th_high,
                'slope':    slope,
                'spread':   round(spread, 1),
            },
            'quality': _quality_score(n_relax, n_focus, spread, r_std, f_std),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        return result

    def _save(self):
        try:
            parent = os.path.dirname(self.save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(self._result, f, indent=2, ensure_ascii=False)
            print(f"[CAL] Saved → {self.save_path}")
        except OSError as e:
            print(f"[CAL] Save failed: {e}")

    @classmethod
    def load(cls, path=SAVE_PATH):
        """Load previously saved calibration and return a done Calibrator."""
        cal = cls(save_path=path)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[CAL] Load failed ({path}): {e}")
            return None
        cal._result = data
        cal._state  = 'done'
        cal.done    = True
        print(f"[CAL] Loaded calibration from {path} "
              f"(recorded: {data.get('timestamp','?')})")
        return cal

    def summary(self) -> str:
        if not self._result:
            return "Calibration not done."
        r = self._result
        t = r['thresholds']
        return (
            f"캘리브레이션 완료\n"
            f"  휴식 집중도: {r['relax']['conc_mean']:.1f} ± {r['relax']['conc_std']:.1f}\n"
            f"  집중 집중도: {r['focus']['conc_mean']:.1f} ± {r['focus']['conc_std']:.1f}\n"
            f"  개인 임계값: Low={t['th_low']}  High={t['th_high']}  Slope={t['slope']}\n"
            f"  품질: {r['quality']['grade']} ({r['quality']['score']}/100)"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _std(vals):
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean)**2 for v in vals) / (len(vals) - 1)) ** 0.5

def _quality_score(n_relax, n_focus, spread, r_std, f_std) -> dict:
    """
    Heuristic quality score (0-100).
      - Frame count: need ≥ 30 frames per phase
      - Spread: larger relax-focus gap = better discriminability
      - Stability: lower std = cleaner signal
    """
    frame_score = min(100, (n_relax + n_focus) / 1.2)
    spread_score = min(100, max(0, spread * 2.5))
    noise_score  = max(0, 100 - (r_std + f_std) * 2)
    score = round(frame_score * 0.3 + spread_score * 0.5 + noise_score * 0.2)

    if score >= 80:   grade = 'A'
    elif score >= 60: grade = 'B'
    elif score >= 40: grade = 'C'
    else:             grade = 'D (재측정 권장)'

    return {'score': score, 'grade': grade,
            'frame_score': round(frame_score),
            'spread_score': round(spread_score),
            'noise_score': round(noise_score)}


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import random, math

    print("=== Calibration Simulation Test ===")
    cal = Calibrator(relax_secs=3, focus_secs=3)  # short for testing
    status = cal.start()
    print(f"Start: {status}")

    t = 0.0
    while not cal.done:
        phase_conc = 30 + random.gauss(0, 5) if cal._state == 'relax' \
                     else 65 + random.gauss(0, 7)
        bands = {
            'delta': 30 + random.gauss(0,3),
            'theta': 40 + random.gauss(0,5),
            'alpha': 60 - phase_conc*0.3 + random.gauss(0,4),
            'beta':  20 + phase_conc*0.5 + random.gauss(0,4),
            'gamma': 15 + phase_conc*0.2 + random.gauss(0,3),
            'concentration': phase_conc,
        }
        status = cal.push(bands)
        t += 0.1
        time.sleep(0.01)

    print(cal.summary())
    print(f"Settings for server: {cal.to_settings()}")
