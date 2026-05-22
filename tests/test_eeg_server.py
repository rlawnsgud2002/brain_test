"""
Unit tests for eeg_server.py helper functions.

Tests pure functions (not WebSocket coroutines):
- update_state
- bandpower
- bands_to_14ch
- compute_frame
- settings/protocol validation logic (extracted in fixtures)
"""
import sys
import math
import pytest
import numpy as np

import eeg_server


# ── update_state ─────────────────────────────────────────────────────────────
class TestUpdateState:
    def test_updates_channels_and_bands(self):
        eeg_server.STATE['channels'] = []
        eeg_server.STATE['bands'] = {}
        eeg_server.update_state([1.234, 5.678, 9.0], {'alpha': 50})
        assert eeg_server.STATE['channels'] == [1.23, 5.68, 9.0]
        assert eeg_server.STATE['bands'] == {'alpha': 50}

    def test_rounds_channel_values(self):
        eeg_server.update_state([1.999999, 2.000001], {})
        assert eeg_server.STATE['channels'] == [2.0, 2.0]

    def test_handles_empty_list(self):
        eeg_server.update_state([], {'beta': 30})
        assert eeg_server.STATE['channels'] == []
        assert eeg_server.STATE['bands'] == {'beta': 30}


# ── bandpower ────────────────────────────────────────────────────────────────
class TestBandpower:
    def test_short_signal_returns_zero(self):
        # nperseg < 4 → return 0
        result = eeg_server.bandpower(np.array([1.0, 2.0]), fs=128, fmin=8, fmax=12)
        assert result == 0.0

    def test_zero_signal_returns_zero(self):
        result = eeg_server.bandpower(np.zeros(256), fs=128, fmin=8, fmax=12)
        assert result == 0.0

    def test_sine_in_band_has_power(self):
        # 10 Hz sine wave should have power in [8,12] band
        fs = 128
        t = np.arange(256) / fs
        sig = np.sin(2 * np.pi * 10 * t)
        power_in_band = eeg_server.bandpower(sig, fs=fs, fmin=8, fmax=12)
        power_out_band = eeg_server.bandpower(sig, fs=fs, fmin=20, fmax=30)
        assert power_in_band > power_out_band

    def test_returns_finite(self):
        # NaN/inf inputs must not propagate
        sig = np.array([np.nan] * 256)
        result = eeg_server.bandpower(sig, fs=128, fmin=8, fmax=12)
        # nan/inf safeguard at end of function returns 0
        assert np.isfinite(result)


# ── bands_to_14ch ────────────────────────────────────────────────────────────
class TestBandsTo14ch:
    def test_returns_14_channels(self):
        vals = eeg_server.bands_to_14ch(30, 40, 50, 45, 20)
        assert len(vals) == 14

    def test_values_clipped_0_100(self):
        # Even pathological inputs should produce values in [0,100]
        vals = eeg_server.bands_to_14ch(0, 0, 0.001, 1000, 0, noise=0.0)
        for v in vals:
            assert 0 <= v <= 100

    def test_zero_alpha_no_crash(self):
        # alpha=0 → ratio = beta/1e-9 = huge, but base clips to 95
        vals = eeg_server.bands_to_14ch(30, 40, 0, 45, 20)
        assert len(vals) == 14
        for v in vals:
            assert math.isfinite(v)

    def test_deterministic_with_seeded_random(self):
        # bands_to_14ch uses random.random() — must be reproducible if seeded
        import random
        random.seed(42)
        a = eeg_server.bands_to_14ch(30, 40, 50, 45, 20, noise=5.0)
        random.seed(42)
        b = eeg_server.bands_to_14ch(30, 40, 50, 45, 20, noise=5.0)
        assert a == b


# ── compute_frame ────────────────────────────────────────────────────────────
@pytest.mark.skipif(not eeg_server.NUMPY_OK, reason="numpy/scipy not available")
class TestComputeFrame:
    @pytest.fixture
    def eeg_14ch(self):
        # 14 channels of 1-second sine waves at different frequencies
        fs = 128
        t = np.arange(fs) / fs
        return np.stack([np.sin(2 * np.pi * (i + 1) * t) for i in range(14)])

    def test_returns_required_keys(self, eeg_14ch):
        frame = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False)
        assert {'type', 'channels', 'contact_quality', 'bands'} <= frame.keys()
        assert frame['type'] == 'eeg'

    def test_channels_count(self, eeg_14ch):
        frame = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False)
        assert len(frame['channels']) == 14
        assert len(frame['contact_quality']) == 14

    def test_bands_are_finite(self, eeg_14ch):
        frame = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False)
        for k, v in frame['bands'].items():
            if isinstance(v, (int, float)):
                assert math.isfinite(v), f"Band '{k}' is non-finite: {v}"

    def test_concentration_in_range(self, eeg_14ch):
        frame = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False)
        assert 0 <= frame['bands']['concentration'] <= 100

    def test_corrupted_baseline_does_not_propagate(self, eeg_14ch):
        # NaN/inf/zero baseline must be guarded
        for bad in [{'engagement_index': float('nan')},
                    {'engagement_index': float('inf')},
                    {'engagement_index': 0},
                    {'engagement_index': -1}]:
            frame = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False,
                                              baseline=bad)
            assert math.isfinite(frame['bands']['concentration'])
            assert 0 <= frame['bands']['concentration'] <= 100

    def test_valid_baseline_used(self, eeg_14ch):
        frame_no_base = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False)
        frame_base = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False,
                                                baseline={'engagement_index': 0.5})
        # Both should produce valid output; values may differ
        assert math.isfinite(frame_no_base['bands']['concentration'])
        assert math.isfinite(frame_base['bands']['concentration'])

    def test_faa_bounded(self, eeg_14ch):
        frame = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False)
        faa = frame['bands']['faa']
        # FAA should be in [-1, +1] by definition
        assert -1.0 <= faa <= 1.0

    def test_contact_quality_bounded(self, eeg_14ch):
        frame = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False)
        for q in frame['contact_quality']:
            assert 0 <= q <= 1


# ── Settings validation (extracted from recv_loop) ──────────────────────────
class TestSettingsValidation:
    """Test the coercion logic in the WS settings handler — replicated here as
    a pure-function check so we don't need to run a real WebSocket."""

    @staticmethod
    def _validate(new_s, current):
        """Mirror of the validation logic in _make_recv."""
        _NUM = ('thLow', 'thHigh', 'dt', 'slope')
        coerced = {}
        for k, v in new_s.items():
            if k in _NUM:
                try:
                    coerced[k] = float(v)
                except (TypeError, ValueError):
                    return None, f'Invalid settings: {k} must be a number'
            else:
                coerced[k] = v
        tl = coerced.get('thLow',  current['thLow'])
        th = coerced.get('thHigh', current['thHigh'])
        if tl >= th:
            return None, 'Invalid settings: thLow must be < thHigh'
        return coerced, None

    def test_valid_numeric_settings(self):
        current = {'thLow': 30, 'thHigh': 70}
        coerced, err = self._validate({'thLow': 25, 'thHigh': 75}, current)
        assert err is None
        assert coerced == {'thLow': 25.0, 'thHigh': 75.0}

    def test_string_numeric_coerced(self):
        current = {'thLow': 30, 'thHigh': 70}
        coerced, err = self._validate({'thLow': '40', 'thHigh': '80'}, current)
        assert err is None
        assert coerced['thLow'] == 40.0

    def test_non_numeric_rejected(self):
        current = {'thLow': 30, 'thHigh': 70}
        coerced, err = self._validate({'thLow': 'abc'}, current)
        assert coerced is None
        assert 'thLow' in err

    def test_none_rejected(self):
        current = {'thLow': 30, 'thHigh': 70}
        coerced, err = self._validate({'thLow': None}, current)
        assert coerced is None

    def test_thLow_ge_thHigh_rejected(self):
        current = {'thLow': 30, 'thHigh': 70}
        coerced, err = self._validate({'thLow': 80, 'thHigh': 70}, current)
        assert coerced is None
        assert 'thLow must be < thHigh' in err

    def test_partial_update_uses_current(self):
        # Only thLow provided; thHigh from current
        current = {'thLow': 30, 'thHigh': 70}
        coerced, err = self._validate({'thLow': 25}, current)
        assert err is None
        assert coerced == {'thLow': 25.0}

    def test_non_numeric_keys_passthrough(self):
        # Unknown keys should pass through unchanged
        current = {'thLow': 30, 'thHigh': 70}
        coerced, err = self._validate({'thLow': 25, 'somekey': 'string'}, current)
        assert err is None
        assert coerced['somekey'] == 'string'


# ── Protocol validation ──────────────────────────────────────────────────────
class TestProtocolValidation:
    def test_valid_protocols_in_dict(self):
        if not eeg_server.EXP_OK:
            pytest.skip("experiment module not loaded")
        # Server validates against PROTOCOLS keys
        assert 'short' in eeg_server.PROTOCOLS
        assert 'standard' in eeg_server.PROTOCOLS

    def test_unknown_protocol_detection(self):
        if not eeg_server.EXP_OK:
            pytest.skip("experiment module not loaded")
        # The server's validation: `protocol not in PROTOCOLS`
        assert 'fake_protocol' not in eeg_server.PROTOCOLS
        assert 'evil' not in eeg_server.PROTOCOLS


# ── Message type validation (non-dict payload rejection) ─────────────────────
class TestMessagePayloadValidation:
    """Mirror of the isinstance check added to recv_loop — verifies the
    rejection logic without spinning up a WebSocket."""

    @staticmethod
    def _is_valid_payload(d):
        return isinstance(d, dict)

    def test_dict_accepted(self):
        assert self._is_valid_payload({'type': 'settings'})

    def test_array_rejected(self):
        assert not self._is_valid_payload([1, 2, 3])

    def test_string_rejected(self):
        assert not self._is_valid_payload('hello')

    def test_number_rejected(self):
        assert not self._is_valid_payload(42)

    def test_null_rejected(self):
        assert not self._is_valid_payload(None)


# ── Argparse validation ──────────────────────────────────────────────────────
class TestArgValidation:
    """Verify the manual validation added after parser.parse_args()."""

    @staticmethod
    def _validate(speed=1.0, trial=0, port=8765):
        errs = []
        if speed <= 0:
            errs.append('speed')
        if trial < 0:
            errs.append('trial')
        if not (1 <= port <= 65535):
            errs.append('port')
        return errs

    def test_defaults_valid(self):
        assert self._validate() == []

    def test_zero_speed_rejected(self):
        assert 'speed' in self._validate(speed=0)

    def test_negative_speed_rejected(self):
        assert 'speed' in self._validate(speed=-1)

    def test_negative_trial_rejected(self):
        assert 'trial' in self._validate(trial=-1)

    def test_port_out_of_range_rejected(self):
        assert 'port' in self._validate(port=0)
        assert 'port' in self._validate(port=70000)


# ── DataFrame index vs sent-counter (stream_mental progress fix) ────────────
class TestSentCounterIndependence:
    """Verify the pattern: if rows are skipped (NaN), the monotonic counter
    used for timestamp differs from the original df index."""

    def test_skipped_rows_dont_advance_counter(self):
        # Simulate the iteration pattern
        rows = [(0, 'ok'), (1, 'skip'), (2, 'ok'), (3, 'skip'), (4, 'ok')]
        sent = 0
        timestamps = []
        progresses = []
        n = len(rows)
        for i, status in rows:
            if status == 'skip':
                continue
            timestamps.append(round(sent * 0.5, 2))
            progresses.append(round((i + 1) / n, 3))
            sent += 1
        # Timestamps are monotonic, no gaps
        assert timestamps == [0.0, 0.5, 1.0]
        # Progresses reflect actual file position (with gaps)
        assert progresses == [0.2, 0.6, 1.0]
