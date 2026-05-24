"""Unit tests for calibration.Calibrator."""
import json
import os
import time
import pytest

from calibration import Calibrator, _std, _quality_score


# ── Helpers ───────────────────────────────────────────────────────────────────
def _bands(conc=50, delta=30, theta=40, alpha=50, beta=45, gamma=20):
    return {'delta': delta, 'theta': theta, 'alpha': alpha,
            'beta': beta, 'gamma': gamma, 'concentration': conc}


def _drive(cal, conc_relax=30, conc_focus=70, frames_per_phase=20):
    """Push frames through both phases at a faster-than-real-time rate
    by monkey-patching the internal phase timer."""
    cal.start()
    # Simulate RELAX phase
    for _ in range(frames_per_phase):
        cal.push(_bands(conc=conc_relax))
    # Force transition to FOCUS
    cal._phase_ts = time.time() - cal.relax_secs - 0.1
    cal.push(_bands(conc=conc_relax))
    # Simulate FOCUS phase
    for _ in range(frames_per_phase):
        cal.push(_bands(conc=conc_focus))
    # Force transition to DONE
    cal._phase_ts = time.time() - cal.focus_secs - 0.1
    cal.push(_bands(conc=conc_focus))
    return cal


# ── _std helper ───────────────────────────────────────────────────────────────
class TestStd:
    def test_empty_returns_zero(self):
        assert _std([]) == 0.0

    def test_single_returns_zero(self):
        assert _std([42.0]) == 0.0

    def test_known_values(self):
        # std of [1,2,3,4,5] using N-1 = sqrt(2.5) ≈ 1.5811
        assert abs(_std([1, 2, 3, 4, 5]) - 1.5811388300841898) < 1e-9

    def test_identical_values(self):
        assert _std([5, 5, 5, 5]) == 0.0


# ── _quality_score helper ─────────────────────────────────────────────────────
class TestQualityScore:
    def test_returns_required_keys(self):
        q = _quality_score(n_relax=50, n_focus=50, spread=30, r_std=5, f_std=5)
        assert {'score', 'grade', 'frame_score', 'spread_score', 'noise_score'} <= q.keys()

    def test_high_quality_grade_a(self):
        q = _quality_score(n_relax=100, n_focus=100, spread=40, r_std=2, f_std=2)
        assert q['grade'] == 'A'
        assert q['score'] >= 80

    def test_low_quality_grade_d(self):
        q = _quality_score(n_relax=5, n_focus=5, spread=2, r_std=30, f_std=30)
        assert q['grade'].startswith('D')

    def test_score_bounded_0_to_100(self):
        # Pathological inputs should not push score outside 0–100
        for spread in (-50, 0, 200):
            for std in (0, 100):
                q = _quality_score(50, 50, spread, std, std)
                assert 0 <= q['score'] <= 100


# ── Calibrator state machine ─────────────────────────────────────────────────
class TestCalibratorLifecycle:
    def test_initial_state_is_idle(self):
        cal = Calibrator(save_path='/tmp/test_cal_init.json')
        assert cal._state == 'idle'
        assert cal.done is False
        assert cal._result is None

    def test_push_during_idle_does_not_crash(self):
        cal = Calibrator(save_path='/tmp/test_cal_idle.json')
        result = cal.push(_bands())
        assert result['state'] == 'idle'

    def test_start_transitions_to_relax(self):
        cal = Calibrator(save_path='/tmp/test_cal_start.json')
        status = cal.start()
        assert cal._state == 'relax'
        assert status['phase'] == 'relax'
        assert status['duration'] == cal.relax_secs

    def test_start_clears_previous_buffers(self):
        cal = Calibrator(save_path='/tmp/test_cal_clear.json')
        cal._relax_buf.append(_bands())
        cal._focus_buf.append(_bands())
        cal.start()
        assert cal._relax_buf == []
        assert cal._focus_buf == []

    def test_relax_to_focus_transition(self):
        cal = Calibrator(save_path='/tmp/test_cal_r2f.json')
        cal.start()
        cal.push(_bands(conc=30))
        # Force time forward
        cal._phase_ts = time.time() - cal.relax_secs - 0.1
        status = cal.push(_bands(conc=30))
        assert cal._state == 'focus'
        assert status.get('transition') is True

    def test_focus_to_done_transition(self, tmp_path):
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal)
        assert cal._state == 'done'
        assert cal.done is True
        assert cal._result is not None

    def test_push_after_done_returns_result(self, tmp_path):
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal)
        status = cal.push(_bands())
        assert status['state'] == 'done'
        assert 'result' in status


# ── Compute output structure ─────────────────────────────────────────────────
class TestComputeOutput:
    def test_result_has_required_top_keys(self, tmp_path):
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal)
        r = cal._result
        assert {'relax', 'focus', 'thresholds', 'quality', 'timestamp'} <= r.keys()

    def test_thresholds_have_required_fields(self, tmp_path):
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal)
        t = cal._result['thresholds']
        assert {'th_low', 'th_high', 'slope', 'spread'} <= t.keys()

    def test_threshold_low_clamped(self, tmp_path):
        # Extreme inputs should still produce clamped thresholds
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal, conc_relax=5, conc_focus=10)  # very low spread
        t = cal._result['thresholds']
        assert 10.0 <= t['th_low'] <= 45.0
        assert 55.0 <= t['th_high'] <= 90.0

    def test_threshold_low_lt_high(self, tmp_path):
        # Even with reversed inputs (focus<relax), output must respect bounds
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal, conc_relax=80, conc_focus=20)
        t = cal._result['thresholds']
        # With reversed inputs, clamp still produces low<high
        assert t['th_low'] < t['th_high']

    def test_slope_clamped(self, tmp_path):
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal, conc_relax=0, conc_focus=100)  # extreme spread
        t = cal._result['thresholds']
        assert 0.5 <= t['slope'] <= 3.0

    def test_to_settings_keys(self, tmp_path):
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal)
        s = cal.to_settings()
        assert set(s.keys()) == {'thLow', 'thHigh', 'slope'}

    def test_to_settings_empty_before_done(self):
        cal = Calibrator(save_path='/tmp/test_cal_nodone.json')
        assert cal.to_settings() == {}

    def test_thresholds_empty_before_done(self):
        cal = Calibrator(save_path='/tmp/test_cal_nothr.json')
        assert cal.thresholds() == {}


# ── Edge cases ───────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_avg_empty_buffer_returns_zero(self):
        cal = Calibrator(save_path='/tmp/test_cal_avg.json')
        assert cal._avg([], 'alpha') == 0.0

    def test_avg_missing_key_treated_as_zero(self):
        cal = Calibrator(save_path='/tmp/test_cal_miss.json')
        # bands lacking 'gamma' key
        bufs = [{'alpha': 50}, {'alpha': 60}]
        # _avg sums b.get(key,0); with no gamma -> 0
        assert cal._avg(bufs, 'gamma') == 0.0
        assert cal._avg(bufs, 'alpha') == 55.0

    def test_concentration_missing_defaults_50(self, tmp_path):
        # Without 'concentration' field, default is 50
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        cal.start()
        bands_no_conc = {'delta': 30, 'theta': 40, 'alpha': 50, 'beta': 45, 'gamma': 20}
        for _ in range(20):
            cal.push(bands_no_conc)
        cal._phase_ts = time.time() - cal.relax_secs - 0.1
        cal.push(bands_no_conc)
        for _ in range(20):
            cal.push(bands_no_conc)
        cal._phase_ts = time.time() - cal.focus_secs - 0.1
        cal.push(bands_no_conc)
        # With concentration=50 both phases, spread=0 → thresholds at clamp
        assert cal.done is True
        assert cal._result['thresholds']['spread'] == 0.0


# ── Persistence ──────────────────────────────────────────────────────────────
class TestSaveLoad:
    def test_save_writes_valid_json(self, tmp_path):
        path = tmp_path / 'cal.json'
        cal = Calibrator(save_path=str(path))
        _drive(cal)
        assert path.exists()
        data = json.loads(path.read_text())
        assert 'thresholds' in data

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = Calibrator.load(path=str(tmp_path / 'missing.json'))
        assert result is None

    def test_load_valid_file(self, tmp_path):
        path = tmp_path / 'cal.json'
        cal = Calibrator(save_path=str(path))
        _drive(cal)
        # Reload
        loaded = Calibrator.load(path=str(path))
        assert loaded is not None
        assert loaded.done is True
        assert loaded.to_settings() == cal.to_settings()

    def test_load_corrupted_json_returns_none(self, tmp_path):
        path = tmp_path / 'corrupt.json'
        path.write_text('{not valid json')
        result = Calibrator.load(path=str(path))
        assert result is None

    def test_load_empty_dict_returns_none(self, tmp_path):
        # Valid JSON but wrong schema — must reject, not raise KeyError later
        path = tmp_path / 'empty.json'
        path.write_text('{}')
        result = Calibrator.load(path=str(path))
        assert result is None

    def test_load_array_returns_none(self, tmp_path):
        # Root must be a dict, not an array
        path = tmp_path / 'arr.json'
        path.write_text('[1, 2, 3]')
        result = Calibrator.load(path=str(path))
        assert result is None

    def test_load_missing_thresholds_keys_returns_none(self, tmp_path):
        # 'thresholds' exists but missing required sub-keys (th_low/th_high/slope)
        path = tmp_path / 'partial.json'
        path.write_text('{"thresholds": {"th_low": 30}}')
        result = Calibrator.load(path=str(path))
        assert result is None

    def test_save_is_atomic(self, tmp_path):
        # After _save() succeeds, no .tmp file should be left behind
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal)
        assert (tmp_path / 'cal.json').exists()
        assert not (tmp_path / 'cal.json.tmp').exists()

    def test_save_creates_parent_directory(self, tmp_path):
        nested = tmp_path / 'sub' / 'dir' / 'cal.json'
        cal = Calibrator(save_path=str(nested))
        _drive(cal)
        assert nested.exists()


# ── Summary ──────────────────────────────────────────────────────────────────
class TestSummary:
    def test_summary_before_done(self):
        cal = Calibrator(save_path='/tmp/test_cal_summ.json')
        assert 'not done' in cal.summary().lower()

    def test_summary_after_done(self, tmp_path):
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        _drive(cal)
        s = cal.summary()
        assert 'Low=' in s and 'High=' in s


# ── Degenerate buffer warning ────────────────────────────────────────────────
class TestDegenerateBuffer:
    def test_empty_relax_buf_produces_clamped_thresholds(self, tmp_path, capsys):
        # If the relax buffer is empty, _compute should warn and still produce valid output
        cal = Calibrator(save_path=str(tmp_path / 'cal.json'))
        cal._relax_buf = []
        cal._focus_buf = [{'concentration': 70, 'delta': 30, 'theta': 40,
                           'alpha': 50, 'beta': 45, 'gamma': 20}]
        result = cal._compute()
        captured = capsys.readouterr()
        assert 'Warning' in captured.out or 'degenerate' in captured.out.lower()
        t = result['thresholds']
        assert 10.0 <= t['th_low'] <= 45.0
        assert 55.0 <= t['th_high'] <= 90.0
