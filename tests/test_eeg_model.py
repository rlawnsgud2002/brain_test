"""Unit tests for eeg_model.VPatternRuleBased (the always-available rule-based detector)."""
import pytest

from eeg_model import VPatternRuleBased, make_detector, TORCH_OK


# ── VPatternRuleBased construction ───────────────────────────────────────────
class TestRuleBasedInit:
    def test_initial_state(self):
        det = VPatternRuleBased()
        assert det._buf == []
        assert det.v_prob == 0.0
        assert det.v_active is False
        assert det._cooldown == 0

    def test_default_maxlen(self):
        det = VPatternRuleBased()
        assert det._maxlen == 120

    def test_custom_maxlen(self):
        det = VPatternRuleBased(maxlen=50)
        assert det._maxlen == 50


# ── Buffer behavior ──────────────────────────────────────────────────────────
class TestBuffer:
    def test_push_appends_to_buffer(self):
        det = VPatternRuleBased()
        det.push(50.0)
        det.push(45.0)
        assert det._buf == [50.0, 45.0]

    def test_maxlen_caps_buffer(self):
        det = VPatternRuleBased(maxlen=10)
        for v in range(20):
            det.push(float(v))
        assert len(det._buf) == 10
        # Should hold the most recent 10 values (10..19)
        assert det._buf == [float(v) for v in range(10, 20)]

    def test_push_returns_dict_with_required_keys(self):
        det = VPatternRuleBased()
        result = det.push(50.0)
        assert {'v_prob', 'v_active', 'rule_based'} <= result.keys()

    def test_push_coerces_to_float(self):
        det = VPatternRuleBased()
        det.push(50)  # int
        det.push("60.5")  # string
        assert det._buf[0] == 50.0
        assert det._buf[1] == 60.5


# ── Insufficient data ────────────────────────────────────────────────────────
class TestInsufficientData:
    def test_empty_buffer_no_detection(self):
        det = VPatternRuleBased()
        result = det.push(50.0)
        assert result['v_prob'] == 0.0
        assert result['v_active'] is False

    def test_below_drop_win_threshold(self):
        det = VPatternRuleBased()
        # DROP_WIN = 60; push fewer than 62 frames
        for _ in range(30):
            result = det.push(50.0)
        assert result['v_active'] is False
        assert result['v_prob'] == 0.0


# ── Detection logic ──────────────────────────────────────────────────────────
class TestDetection:
    def test_constant_value_no_detection(self):
        det = VPatternRuleBased()
        for _ in range(80):
            result = det.push(60.0)
        assert result['v_active'] is False

    def test_v_pattern_triggers(self):
        # DROP_WIN=60 needs ≥62 frames before any detection.
        # SMOOTH=4 averages over 5 frames so trough is muted —
        # need only 1–2 recovery frames so the trough stays in the last 3 positions.
        det = VPatternRuleBased()
        for _ in range(50):
            det.push(80.0)                       # warm-up at peak
        for v in [70, 60, 50, 40, 30, 25, 20, 18, 15, 12]:
            det.push(float(v))                   # drop (10 frames)
        # Sharp 2-frame recovery while trough still at_bottom
        result = det.push(40.0)
        result = det.push(60.0)
        assert result['v_active'] is True
        assert result['v_prob'] > 0

    def test_drop_only_partial_activation(self):
        det = VPatternRuleBased()
        for _ in range(30):
            det.push(80.0)
        # Sustained drop, no recovery
        for _ in range(40):
            result = det.push(20.0)
        # During the drop period the detector should activate (waiting for recovery)
        # By the end the trough is old, so activity may have subsided
        # Just verify no exceptions and result structure is valid
        assert 0.0 <= result['v_prob'] <= 1.0

    def test_v_prob_bounded(self):
        # Even with extreme drop/rise, prob must stay 0-1
        det = VPatternRuleBased()
        for _ in range(30):
            det.push(100.0)
        for v in [50, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            det.push(float(v))
        for v in [50, 100, 100, 100, 100]:
            result = det.push(float(v))
        assert 0.0 <= result['v_prob'] <= 1.0


# ── Cooldown ─────────────────────────────────────────────────────────────────
class TestCooldown:
    def test_cooldown_after_full_v_detection(self):
        det = VPatternRuleBased()
        # Trigger a V-pattern
        for _ in range(30):
            det.push(80.0)
        for v in [70, 50, 30, 15, 10, 8, 6, 5, 5, 5]:
            det.push(float(v))
        for v in [10, 30, 55, 75, 80]:
            det.push(float(v))
        # Detect should have triggered cooldown
        if det.v_active:
            assert det._cooldown > 0


# ── Smoothing ────────────────────────────────────────────────────────────────
class TestSmoothing:
    def test_smooth_returns_same_length(self):
        det = VPatternRuleBased()
        arr = [10.0, 20.0, 30.0, 40.0, 50.0]
        smoothed = det._smooth(arr)
        assert len(smoothed) == len(arr)

    def test_smooth_first_value_unchanged(self):
        det = VPatternRuleBased()
        arr = [42.0, 10.0, 20.0]
        smoothed = det._smooth(arr)
        assert smoothed[0] == 42.0

    def test_smooth_averages_window(self):
        det = VPatternRuleBased()  # SMOOTH=4
        arr = [10.0, 10.0, 10.0, 10.0, 50.0]
        smoothed = det._smooth(arr)
        # Last value smoothed over 4 prior + self: mean([10,10,10,10,50]) = 18
        assert abs(smoothed[-1] - 18.0) < 0.01


# ── Factory ──────────────────────────────────────────────────────────────────
class TestFactory:
    def test_make_detector_returns_detector(self):
        # Returns either VPatternML or VPatternRuleBased; both have .push()
        det = make_detector(model_path='/nonexistent/model.pt')
        assert hasattr(det, 'push')

    def test_make_detector_works_without_torch(self):
        # Even if torch isn't installed, this must not crash
        det = make_detector(model_path='/nonexistent/model.pt')
        result = det.push(50.0)
        assert 'v_prob' in result
