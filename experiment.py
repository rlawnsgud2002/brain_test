"""
Experiment Protocol Engine

Defines structured EEG experiment sequences (relax/task/rest phases)
with behavioral marker logging.

Predefined protocols:
  'standard' : 2min baseline → 3min relax → 4min task × 2 → 2min relax
  'short'    : 30s baseline → 1min relax → 2min task × 2 → 1min relax  (for quick tests)
  'custom'   : supply your own phase list

Usage:
  from experiment import ExperimentRunner, PROTOCOLS
  runner = ExperimentRunner(protocol='short')
  runner.start()
  while not runner.done:
      status = runner.tick()       # call ~1Hz
      runner.add_marker('response', eeg_bands)
  runner.save('results/exp_001.json')
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ── Phase definition ──────────────────────────────────────────────────────────
@dataclass
class Phase:
    name:        str          # 'baseline' | 'relax' | 'task' | 'rest'
    duration:    float        # seconds
    instruction: str          # shown to participant
    color:       str = '#3060ff'   # UI accent color
    phase_type:  str = 'rest'      # 'relax' | 'task' | 'rest' | 'baseline'


# ── Built-in protocols ────────────────────────────────────────────────────────
PROTOCOLS = {
    'standard': [
        Phase('BASELINE', 60,  '조용히 앉아서 화면을 바라보세요.',
              '#204080', 'baseline'),
        Phase('RELAX',    120, '눈을 감고 편안하게 호흡하세요.',
              '#2060b0', 'relax'),
        Phase('TASK 1',   240, '제시된 문제를 집중해서 풀어보세요.',
              '#b04020', 'task'),
        Phase('REST',     60,  '잠시 쉬세요. 자유롭게 있어도 됩니다.',
              '#204060', 'rest'),
        Phase('TASK 2',   240, '다음 문제를 집중해서 풀어보세요.',
              '#b04020', 'task'),
        Phase('REST',     60,  '잠시 쉬세요.',
              '#204060', 'rest'),
        Phase('RELAX',    120, '눈을 감고 편안하게 호흡하세요.',
              '#2060b0', 'relax'),
    ],
    'short': [
        Phase('BASELINE', 30,  '준비하세요. 조용히 기다리세요.',
              '#204080', 'baseline'),
        Phase('RELAX',    60,  '눈을 감고 편안하게 쉬세요.',
              '#2060b0', 'relax'),
        Phase('TASK 1',   120, '집중 과제: 속으로 7씩 빼어 세세요. (100→93→86…)',
              '#b04020', 'task'),
        Phase('REST',     30,  '잠시 쉬세요.',
              '#204060', 'rest'),
        Phase('TASK 2',   120, '집중 과제: 소수를 순서대로 떠올리세요.',
              '#b04020', 'task'),
        Phase('RELAX',    60,  '눈을 감고 편안하게 쉬세요.',
              '#2060b0', 'relax'),
    ],
    'quick': [   # dev/test
        Phase('BASELINE', 10,  '대기 중…',     '#204080', 'baseline'),
        Phase('RELAX',    20,  '편안하게.',    '#2060b0', 'relax'),
        Phase('TASK',     30,  '집중 과제.',   '#b04020', 'task'),
        Phase('RELAX',    20,  '마무리.',      '#2060b0', 'relax'),
    ],
}


# ── Marker ────────────────────────────────────────────────────────────────────
@dataclass
class Marker:
    timestamp:   float
    elapsed:     float          # seconds since experiment start
    marker_type: str            # 'phase_start' | 'phase_end' | 'manual' | 'vpattern' | 'threshold'
    phase:       str
    phase_type:  str
    label:       str = ''
    concentration: float = 0.0
    bands:       dict = field(default_factory=dict)


# ── Runner ────────────────────────────────────────────────────────────────────
class ExperimentRunner:
    """
    State machine that steps through protocol phases.
    Call tick() at ~1Hz; it returns status dicts for the client.
    """

    def __init__(self, protocol='short', custom_phases: Optional[List[Phase]] = None):
        if custom_phases is not None:
            if not custom_phases:
                raise ValueError("custom_phases must contain at least one Phase")
            self.phases = custom_phases
        elif protocol in PROTOCOLS:
            self.phases = PROTOCOLS[protocol]
        else:
            raise ValueError(f"Unknown protocol '{protocol}'. Choose: {list(PROTOCOLS)}")

        self.protocol_name  = protocol
        self._phase_idx     = -1
        self._phase_start   = None
        self._exp_start     = None
        self.markers: List[Marker] = []
        self.done           = False
        self._state         = 'idle'
        self._last_eeg      = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> dict:
        self._exp_start  = time.time()
        self._phase_idx  = 0
        self._phase_start = time.time()
        self._state      = 'running'
        self.done        = False
        self.markers.clear()
        self._log_marker('phase_start')
        return self._status()

    def tick(self, eeg_bands: Optional[dict] = None) -> dict:
        """
        Call ~1Hz. Returns current status dict.
        Automatically advances phases when duration expires.
        """
        if self._state != 'running':
            return {'state': self._state}

        if eeg_bands:
            self._last_eeg = eeg_bands

        elapsed_phase = time.time() - self._phase_start
        phase = self.phases[self._phase_idx]

        if elapsed_phase >= phase.duration:
            return self._advance()

        remaining = phase.duration - elapsed_phase
        progress  = elapsed_phase / phase.duration if phase.duration > 0 else 1.0

        return {**self._status(),
                'remaining':       round(remaining, 1),
                'phase_progress':  round(progress, 3)}

    def add_marker(self, label: str, eeg_bands: Optional[dict] = None) -> dict:
        """Manual behavioral marker (e.g., stimulus onset, key press)."""
        m = self._log_marker('manual', label=label, bands=eeg_bands or self._last_eeg)
        return asdict(m)

    def vpattern_marker(self, v_prob: float, eeg_bands: Optional[dict] = None):
        """Auto-marker when V-pattern is detected by ML model."""
        self._log_marker('vpattern', label=f'prob={v_prob:.2f}',
                         bands=eeg_bands or self._last_eeg)

    def threshold_marker(self, direction: str, concentration: float):
        """Auto-marker when concentration crosses threshold."""
        self._log_marker('threshold', label=direction,
                         bands=self._last_eeg, concentration=concentration)

    def total_duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def elapsed(self) -> float:
        if self._exp_start is None:
            return 0.0
        return time.time() - self._exp_start

    def save(self, path='results/experiment.json') -> str:
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            data = {
                'protocol':   self.protocol_name,
                'start_time': time.strftime('%Y-%m-%dT%H:%M:%S',
                                            time.localtime(self._exp_start or time.time())),
                'duration':   round(self.elapsed(), 1),
                'phases':     [asdict(p) for p in self.phases],
                'markers':    [asdict(m) for m in self.markers],
                'summary':    self._summary(),
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[EXP] Saved → {path}")
            return path
        except OSError as e:
            print(f"[EXP] Save failed ({path}): {e}")
            return ''

    def export_csv(self, path='results/experiment.csv') -> str:
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            rows = ['elapsed_s,marker_type,phase,phase_type,label,concentration']
            for m in self.markers:
                label = str(m.label or '').replace(',', ';')  # guard commas in label
                rows.append(f"{m.elapsed:.2f},{m.marker_type},{m.phase},"
                            f"{m.phase_type},{label},{m.concentration:.1f}")
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(rows))
            print(f"[EXP] CSV → {path}")
            return path
        except OSError as e:
            print(f"[EXP] CSV export failed ({path}): {e}")
            return ''

    # ── Internal ──────────────────────────────────────────────────────────────

    def _current_phase(self) -> Phase:
        if 0 <= self._phase_idx < len(self.phases):
            return self.phases[self._phase_idx]
        # Sentinel when idx is out of range (post-done state) — avoids IndexError
        return Phase(name='', duration=0.0, instruction='')

    def _advance(self) -> dict:
        self._log_marker('phase_end')
        self._phase_idx += 1

        if self._phase_idx >= len(self.phases):
            self._state = 'done'
            self.done   = True
            return {'state': 'done',
                    'elapsed': round(self.elapsed(), 1),
                    'summary': self._summary()}

        self._phase_start = time.time()
        self._log_marker('phase_start')
        return {**self._status(), 'transition': True,
                'remaining': self.phases[self._phase_idx].duration,
                'phase_progress': 0.0}

    def _status(self) -> dict:
        if self._phase_idx < 0 or self._phase_idx >= len(self.phases):
            return {'state': self._state}
        p = self._current_phase()
        return {
            'state':       self._state,
            'phase_idx':   self._phase_idx,
            'phase_total': len(self.phases),
            'phase':       p.name,
            'phase_type':  p.phase_type,
            'instruction': p.instruction,
            'color':       p.color,
            'duration':    p.duration,
            'elapsed_exp': round(self.elapsed(), 1),
            'total_dur':   self.total_duration(),
        }

    def _log_marker(self, marker_type: str, label: str = '',
                    bands: Optional[dict] = None, concentration: float = 0.0) -> Marker:
        phase = self._current_phase() if self._phase_idx >= 0 else Phase('', 0, '', '', '')
        conc  = concentration or (bands or {}).get('concentration', 0.0)
        m = Marker(
            timestamp   = time.time(),
            elapsed     = round(self.elapsed(), 3),
            marker_type = marker_type,
            phase       = phase.name,
            phase_type  = phase.phase_type,
            label       = label,
            concentration = round(conc, 2),
            bands       = {k: round(v, 2) for k, v in (bands or {}).items()
                           if k in ('delta','theta','alpha','beta','gamma','concentration')},
        )
        self.markers.append(m)
        return m

    def _summary(self) -> dict:
        by_phase = {}
        for m in self.markers:
            if m.phase_type not in by_phase:
                by_phase[m.phase_type] = []
            if m.concentration > 0:
                by_phase[m.phase_type].append(m.concentration)

        stats = {}
        for ptype, concs in by_phase.items():
            if concs:
                mean = sum(concs) / len(concs)
                stats[ptype] = {
                    'mean_concentration': round(mean, 1),
                    'n_samples': len(concs),
                }

        n_manual  = sum(1 for m in self.markers if m.marker_type == 'manual')
        n_vpattern = sum(1 for m in self.markers if m.marker_type == 'vpattern')

        return {
            'phase_stats': stats,
            'n_markers':  len(self.markers),
            'n_manual':   n_manual,
            'n_vpattern': n_vpattern,
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import random

    print("=== Experiment Protocol Test (quick) ===")
    runner = ExperimentRunner(protocol='quick')
    status = runner.start()
    print(f"Start: phase={status['phase']}, total={runner.total_duration()}s")

    t = 0
    while not runner.done:
        bands = {
            'delta': 30 + random.gauss(0,3),
            'theta': 40 + random.gauss(0,5),
            'alpha': 50 + random.gauss(0,8),
            'beta':  45 + random.gauss(0,6),
            'gamma': 20 + random.gauss(0,4),
            'concentration': 40 + random.gauss(0,12),
        }
        status = runner.tick(bands)
        if status.get('transition'):
            print(f"  → Phase: {status['phase']} ({status['duration']}s)")
        if t % 5 == 0:
            runner.add_marker(f'sample_t{t}', bands)
        t += 1
        time.sleep(0.1)

    print(f"\nDone. Markers: {len(runner.markers)}")
    print(f"Summary: {json.dumps(runner._summary(), indent=2)}")
    saved = runner.save('results/test_exp.json')
    runner.export_csv('results/test_exp.csv')
    print(f"Saved: {saved}")
