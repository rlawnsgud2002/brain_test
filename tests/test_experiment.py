"""Unit tests for experiment.ExperimentRunner."""
import json
import time
import pytest

from experiment import ExperimentRunner, Phase, PROTOCOLS, Marker


def _bands(conc=50):
    return {'delta': 30, 'theta': 40, 'alpha': 50, 'beta': 45,
            'gamma': 20, 'concentration': conc}


# ── Construction ─────────────────────────────────────────────────────────────
class TestConstruction:
    def test_default_protocol_loads(self):
        runner = ExperimentRunner()
        assert runner.protocol_name == 'short'
        assert runner.phases == PROTOCOLS['short']

    @pytest.mark.parametrize('proto', list(PROTOCOLS.keys()))
    def test_all_builtin_protocols(self, proto):
        runner = ExperimentRunner(protocol=proto)
        assert len(runner.phases) > 0

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match='Unknown protocol'):
            ExperimentRunner(protocol='nonexistent')

    def test_custom_phases_overrides(self):
        custom = [Phase('A', 5, 'first', '#000', 'task'),
                  Phase('B', 10, 'second', '#fff', 'rest')]
        runner = ExperimentRunner(custom_phases=custom)
        assert runner.phases == custom

    def test_initial_state(self):
        runner = ExperimentRunner()
        assert runner._state == 'idle'
        assert runner.done is False
        assert runner._phase_idx == -1
        assert runner.markers == []


# ── total_duration ───────────────────────────────────────────────────────────
class TestTotalDuration:
    def test_short_protocol(self):
        runner = ExperimentRunner(protocol='short')
        # short: 30+60+120+30+120+60 = 420s
        assert runner.total_duration() == 420

    def test_custom_phases(self):
        runner = ExperimentRunner(custom_phases=[
            Phase('A', 7, '', '', ''), Phase('B', 13, '', '', '')])
        assert runner.total_duration() == 20

    def test_empty_custom_phases_raises(self):
        # Empty list must not silently fall back to default protocol
        with pytest.raises(ValueError, match='at least one Phase'):
            ExperimentRunner(custom_phases=[])


# ── start ────────────────────────────────────────────────────────────────────
class TestStart:
    def test_start_initializes_state(self):
        runner = ExperimentRunner(protocol='quick')
        status = runner.start()
        assert runner._state == 'running'
        assert runner._phase_idx == 0
        assert status['phase'] == runner.phases[0].name

    def test_start_logs_phase_start_marker(self):
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        assert len(runner.markers) == 1
        assert runner.markers[0].marker_type == 'phase_start'
        assert runner.markers[0].phase == runner.phases[0].name


# ── tick ─────────────────────────────────────────────────────────────────────
class TestTick:
    def test_tick_before_start_returns_idle(self):
        runner = ExperimentRunner()
        result = runner.tick()
        assert result['state'] == 'idle'

    def test_tick_returns_progress(self):
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        status = runner.tick(_bands())
        assert 'remaining' in status
        assert 'phase_progress' in status
        assert 0.0 <= status['phase_progress'] <= 1.0

    def test_tick_advances_phase_after_duration(self):
        runner = ExperimentRunner(custom_phases=[
            Phase('A', 1, '', '', 'task'),
            Phase('B', 1, '', '', 'rest')])
        runner.start()
        # Force past duration
        runner._phase_start = time.time() - 2
        status = runner.tick()
        assert status.get('transition') is True
        assert runner._phase_idx == 1

    def test_tick_after_all_phases_done(self):
        runner = ExperimentRunner(custom_phases=[
            Phase('A', 1, '', '', 'task')])
        runner.start()
        runner._phase_start = time.time() - 2
        status = runner.tick()
        assert status['state'] == 'done'
        assert runner.done is True

    def test_tick_stores_last_eeg(self):
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        bands = _bands(conc=75)
        runner.tick(bands)
        assert runner._last_eeg == bands


# ── Markers ──────────────────────────────────────────────────────────────────
class TestMarkers:
    def test_add_manual_marker(self):
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        n_before = len(runner.markers)
        m = runner.add_marker('response', _bands(conc=80))
        assert len(runner.markers) == n_before + 1
        assert m['marker_type'] == 'manual'
        assert m['label'] == 'response'
        assert m['concentration'] == 80.0

    def test_add_marker_uses_last_eeg_if_none(self):
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        runner.tick(_bands(conc=65))
        m = runner.add_marker('label')
        assert m['concentration'] == 65.0

    def test_vpattern_marker(self):
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        runner.vpattern_marker(0.85, _bands(conc=40))
        v_markers = [m for m in runner.markers if m.marker_type == 'vpattern']
        assert len(v_markers) == 1
        assert 'prob=0.85' in v_markers[0].label

    def test_threshold_marker(self):
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        runner.tick(_bands(conc=50))
        runner.threshold_marker('rising', 72.5)
        t_markers = [m for m in runner.markers if m.marker_type == 'threshold']
        assert len(t_markers) == 1
        assert t_markers[0].concentration == 72.5

    def test_marker_only_includes_known_band_keys(self):
        # Garbage keys must NOT leak into stored marker bands
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        bands = {'delta': 30, 'alpha': 50, 'evil_key': 'inject', 'concentration': 60}
        runner.add_marker('x', bands)
        m = runner.markers[-1]
        assert 'evil_key' not in m.bands


# ── Full lifecycle ───────────────────────────────────────────────────────────
class TestLifecycle:
    def test_run_through_all_phases(self):
        phases = [Phase(f'P{i}', 0.5, '', '', 'task') for i in range(3)]
        runner = ExperimentRunner(custom_phases=phases)
        runner.start()
        for _ in range(len(phases)):
            runner._phase_start = time.time() - 1
            runner.tick()
        assert runner.done is True
        # Each phase: phase_start + phase_end = 2 markers × 3 phases = 6
        assert len(runner.markers) == 6

    def test_done_freezes_state(self):
        runner = ExperimentRunner(custom_phases=[Phase('A', 0.1, '', '', '')])
        runner.start()
        runner._phase_start = time.time() - 1
        runner.tick()
        before_markers = len(runner.markers)
        # Subsequent ticks should not advance or add markers
        runner.tick()
        runner.tick()
        assert len(runner.markers) == before_markers


# ── Summary ──────────────────────────────────────────────────────────────────
class TestSummary:
    def test_summary_has_expected_keys(self):
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        s = runner._summary()
        assert {'phase_stats', 'n_markers', 'n_manual', 'n_vpattern'} <= s.keys()

    def test_summary_counts_markers(self):
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        runner.add_marker('a', _bands())
        runner.add_marker('b', _bands())
        runner.vpattern_marker(0.7, _bands())
        s = runner._summary()
        assert s['n_manual'] == 2
        assert s['n_vpattern'] == 1


# ── Persistence ──────────────────────────────────────────────────────────────
class TestPersistence:
    def test_save_writes_valid_json(self, tmp_path):
        path = tmp_path / 'exp.json'
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        runner.add_marker('m1', _bands(conc=60))
        runner.save(str(path))
        assert path.exists()
        data = json.loads(path.read_text())
        assert data['protocol'] == 'quick'
        assert len(data['markers']) >= 1

    def test_save_no_dir_uses_cwd(self, tmp_path, monkeypatch):
        # Path with no directory component
        monkeypatch.chdir(tmp_path)
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        runner.save('flat.json')
        assert (tmp_path / 'flat.json').exists()

    def test_export_csv_valid(self, tmp_path):
        path = tmp_path / 'exp.csv'
        runner = ExperimentRunner(protocol='quick')
        runner.start()
        runner.add_marker('m1', _bands(conc=60))
        runner.export_csv(str(path))
        text = path.read_text()
        lines = text.strip().split('\n')
        # Header + at least one data row
        assert len(lines) >= 2
        assert lines[0].startswith('elapsed_s,marker_type')


# ── Phase sentinel ───────────────────────────────────────────────────────────
class TestPhaseSentinel:
    def test_current_phase_after_done_does_not_crash(self):
        runner = ExperimentRunner(custom_phases=[Phase('A', 0.1, '', '', '')])
        runner.start()
        runner._phase_start = time.time() - 1
        runner.tick()
        # Phase_idx now == 1 (past last); _current_phase should return sentinel
        p = runner._current_phase()
        assert p.name == ''
        assert p.duration == 0.0
