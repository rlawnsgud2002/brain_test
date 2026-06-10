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

    def test_quality_scalar_present_and_bounded(self, eeg_14ch):
        # v3 bridge consumes a single `quality` scalar alongside contact_quality
        frame = eeg_server.compute_frame(eeg_14ch, fs=128, apply_preprocess=False)
        assert 'quality' in frame
        assert 0.0 <= frame['quality'] <= 1.0
        # contact_quality must still be present (not replaced by the scalar)
        assert 'contact_quality' in frame

    def test_quality_scalar_on_invalid_input(self):
        # Empty/wrong-shape input still returns a quality field (schema consistency)
        frame = eeg_server.compute_frame(np.empty((0, 0)), fs=128)
        assert 'quality' in frame
        assert 0.0 <= frame['quality'] <= 1.0


# ── quality helpers (channel_quality / signal_quality_scalar) ────────────────
class TestSignalQualityScalar:
    def test_empty_returns_neutral(self):
        assert eeg_server.signal_quality_scalar([]) == 0.5

    def test_all_perfect_is_one(self):
        assert eeg_server.signal_quality_scalar([1.0] * 14) == 1.0

    def test_emg_warn_reduces(self):
        base = eeg_server.signal_quality_scalar([1.0] * 14)
        emg  = eeg_server.signal_quality_scalar([1.0] * 14, emg_warn=True)
        assert emg < base
        assert abs(emg - 0.7) < 1e-6

    def test_rejected_ratio_reduces(self):
        q = eeg_server.signal_quality_scalar([1.0] * 14, rejected_ratio=0.5)
        assert abs(q - 0.5) < 1e-6

    def test_clipped_0_1(self):
        assert eeg_server.signal_quality_scalar([5.0] * 4) <= 1.0
        assert eeg_server.signal_quality_scalar([-3.0] * 4) >= 0.0

    def test_nan_rejected_ratio_ignored(self):
        q = eeg_server.signal_quality_scalar([1.0] * 4, rejected_ratio=float('nan'))
        assert q == 1.0

    def test_varies_with_input(self):
        # Sim relies on per-frame noisy contact_quality producing a varying scalar
        a = eeg_server.signal_quality_scalar([0.9] * 14)
        b = eeg_server.signal_quality_scalar([0.8] * 14)
        assert a != b


@pytest.mark.skipif(not eeg_server.NUMPY_OK, reason="numpy not available")
class TestChannelQuality:
    def test_length_matches_input(self):
        eeg = np.random.randn(8, 256)
        assert len(eeg_server.channel_quality(eeg)) == 8

    def test_values_bounded(self):
        eeg = np.random.randn(4, 256) * 20
        for q in eeg_server.channel_quality(eeg):
            assert 0.0 <= q <= 1.0

    def test_flat_channel_low_quality(self):
        # A flat (disconnected) channel → near-zero quality
        eeg = np.zeros((1, 256))
        assert eeg_server.channel_quality(eeg)[0] <= 0.1


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


# ── _SimState ─────────────────────────────────────────────────────────────────
class TestSimState:
    def test_step_returns_14_channels(self):
        sim = eeg_server._SimState()
        channels, bands = sim.step()
        assert len(channels) == 14

    def test_channels_in_range(self):
        sim = eeg_server._SimState()
        for _ in range(20):
            channels, _ = sim.step()
            for v in channels:
                assert 5.0 <= v <= 95.0, f"channel out of range: {v}"

    def test_bands_has_engagement_index(self):
        sim = eeg_server._SimState()
        _, bands = sim.step()
        assert 'engagement_index' in bands

    def test_bands_has_faa(self):
        sim = eeg_server._SimState()
        _, bands = sim.step()
        assert 'faa' in bands
        assert -0.5 <= bands['faa'] <= 0.5

    def test_bands_has_all_required_keys(self):
        sim = eeg_server._SimState()
        _, bands = sim.step()
        required = {'delta', 'theta', 'alpha', 'beta', 'gamma',
                    'concentration', 'engagement_index', 'faa'}
        assert required <= set(bands.keys())

    def test_concentration_in_range(self):
        sim = eeg_server._SimState()
        for _ in range(30):
            _, bands = sim.step()
            assert 0.0 <= bands['concentration'] <= 100.0

    def test_focus_transitions_smoothly(self):
        """Low-pass filter means focus changes < 0.5 per step."""
        sim = eeg_server._SimState()
        prev_lp = sim._focus_lp
        max_jump = 0.0
        for _ in range(100):
            sim.step()
            jump = abs(sim._focus_lp - prev_lp)
            max_jump = max(max_jump, jump)
            prev_lp = sim._focus_lp
        assert max_jump < 0.5


# ── stream_sim frame fields ───────────────────────────────────────────────────
class TestStreamSimFrame:
    """Verify _SimState produces all fields that stream_sim includes in frames."""

    def test_sim_bands_include_engagement_index_and_faa(self):
        sim = eeg_server._SimState()
        _, bands = sim.step()
        assert 'engagement_index' in bands
        assert 'faa' in bands

    def test_sim_channels_count_matches_spatial_weights(self):
        sim = eeg_server._SimState()
        channels, _ = sim.step()
        assert len(channels) == len(eeg_server.SPATIAL_W)


# ── TGAM parse_payload ────────────────────────────────────────────────────────
class TestTgamParsePayload:
    """Verify parse_payload skips unknown TGCP codes without corrupting state."""

    def _make_parser(self):
        """Return (parse_payload, state_dict) with captured closure variables."""
        attention = [50.0]
        meditation = [50.0]
        poor_signal = [200]
        bands_raw = [0] * 8
        raw_buf = []

        def parse_payload(payload):
            i = 0
            while i < len(payload):
                code = payload[i]; i += 1
                if code == 0x02:
                    poor_signal[0] = payload[i]; i += 1
                elif code == 0x04:
                    attention[0] = float(payload[i]); i += 1
                elif code == 0x05:
                    meditation[0] = float(payload[i]); i += 1
                elif code == 0x16:
                    i += 1
                elif code == 0x80:
                    vlen = payload[i] if i < len(payload) else 0
                    i += 1
                    if vlen >= 2 and i + 1 < len(payload):
                        raw_buf.append(int.from_bytes(payload[i:i+2], 'big', signed=True))
                    i += vlen
                elif code == 0x83:
                    vlen = payload[i] if i < len(payload) else 0
                    i += 1
                    for b in range(8):
                        if i + 2 < len(payload):
                            bands_raw[b] = int.from_bytes(payload[i:i+3], 'big')
                        i += 3
                else:
                    if code < 0x80:
                        i += 1
                    elif i < len(payload):
                        i += 1 + payload[i]

        return parse_payload, {'attention': attention, 'meditation': meditation,
                               'poor_signal': poor_signal, 'bands_raw': bands_raw,
                               'raw_buf': raw_buf}

    def test_known_code_attention(self):
        parse, state = self._make_parser()
        parse([0x04, 75])
        assert state['attention'][0] == 75.0

    def test_known_code_poor_signal(self):
        parse, state = self._make_parser()
        parse([0x02, 0])
        assert state['poor_signal'][0] == 0

    def test_unknown_single_byte_code_skips_value(self):
        """Unknown code 0x03 (< 0x80): skip its 1-byte value, then parse next code."""
        parse, state = self._make_parser()
        # [0x03, 0xFF, 0x04, 90] → unknown(0x03 + value 0xFF) → attention=90
        parse([0x03, 0xFF, 0x04, 90])
        assert state['attention'][0] == 90.0

    def test_unknown_multi_byte_code_skips_length_and_value(self):
        """Unknown code 0x85 (>= 0x80): next byte is length, skip length bytes."""
        parse, state = self._make_parser()
        # [0x85, 3, 0xAA, 0xBB, 0xCC, 0x04, 42] → skip 3 bytes → attention=42
        parse([0x85, 3, 0xAA, 0xBB, 0xCC, 0x04, 42])
        assert state['attention'][0] == 42.0

    def test_multiple_known_codes_in_sequence(self):
        parse, state = self._make_parser()
        parse([0x02, 50, 0x04, 80, 0x05, 60])
        assert state['poor_signal'][0] == 50
        assert state['attention'][0] == 80.0
        assert state['meditation'][0] == 60.0

    def test_unknown_code_does_not_prevent_subsequent_parsing(self):
        """After an unknown code, subsequent known codes must still be parsed."""
        parse, state = self._make_parser()
        parse([0x06, 0x00, 0x05, 33])   # 0x06 unknown + value → meditation=33
        assert state['meditation'][0] == 33.0

    def test_band_power_packet_skips_length_byte(self):
        """0x83 = ASIC_EEG_POWER: [code][len=0x18][8×3-byte values].
        The length byte (0x18) must be skipped, else it bleeds into band 0."""
        parse, state = self._make_parser()
        band_bytes = []
        for v in range(1, 9):
            band_bytes += [0x00, 0x00, v]   # 3-byte big-endian == v
        parse([0x83, 0x18] + band_bytes)
        assert state['bands_raw'] == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_attention_parsed_after_band_powers(self):
        """Regression: the 0x83 length-byte off-by-one used to misalign every
        byte after the band block, corrupting attention/meditation that follow."""
        parse, state = self._make_parser()
        band_bytes = []
        for v in range(1, 9):
            band_bytes += [0x00, 0x00, v]
        # Realistic packet: poor_signal, ASIC_EEG_POWER, attention, meditation
        parse([0x02, 0x00, 0x83, 0x18] + band_bytes + [0x04, 10, 0x05, 12])
        assert state['poor_signal'][0] == 0
        assert state['attention'][0] == 10.0
        assert state['meditation'][0] == 12.0
        assert state['bands_raw'] == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_raw_eeg_skips_length_byte(self):
        """0x80 = RAW: [code][len=0x02][2-byte big-endian int16].
        Length byte must be skipped, else it becomes the value's high byte."""
        parse, state = self._make_parser()
        parse([0x80, 0x02, 0x01, 0x00])   # value 0x0100 == 256
        assert state['raw_buf'] == [256]


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
