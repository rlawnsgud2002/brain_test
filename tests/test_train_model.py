"""
Unit tests for train_model.py validation logic.

Since train_model.py imports torch at module level, we can't import it directly
without torch installed. Instead we replicate the validation helpers and verify
the same logic that ships in the script.
"""
import numpy as np
import pytest


# ── Trial argument parsing (mirror of train_model.main()) ────────────────────
class TestTrialArgParsing:
    @staticmethod
    def _parse(s):
        try:
            trials = [int(t.strip()) for t in s.split(',') if t.strip()]
        except ValueError:
            return None
        if any(t < 0 for t in trials):
            return 'negative'
        return trials

    def test_valid_trials(self):
        assert self._parse('0,1,2,3') == [0, 1, 2, 3]

    def test_whitespace_tolerated(self):
        assert self._parse('0, 1, 2') == [0, 1, 2]

    def test_trailing_comma_tolerated(self):
        assert self._parse('0,1,2,') == [0, 1, 2]

    def test_empty_string_returns_empty(self):
        assert self._parse('') == []

    def test_invalid_token_returns_none(self):
        assert self._parse('0,1,x') is None

    def test_decimal_rejected(self):
        assert self._parse('0,1.5,2') is None

    def test_negative_detected(self):
        assert self._parse('-1,0,1') == 'negative'


# ── Class balance check (mirror of train()) ──────────────────────────────────
class TestPosWeightEdgeCases:
    @staticmethod
    def _check(y):
        if len(y) == 0:
            return 'empty'
        y_mean = float(y.mean())
        if y_mean == 0.0: return 'all_negative'
        if y_mean == 1.0: return 'all_positive'
        return 'mixed'

    def test_all_negative_detected(self):
        assert self._check(np.zeros(100)) == 'all_negative'

    def test_all_positive_detected(self):
        assert self._check(np.ones(100)) == 'all_positive'

    def test_balanced_detected(self):
        y = np.concatenate([np.zeros(50), np.ones(50)])
        assert self._check(y) == 'mixed'

    def test_imbalanced_still_mixed(self):
        # 99% negative still counts as mixed
        y = np.concatenate([np.zeros(99), np.ones(1)])
        assert self._check(y) == 'mixed'

    def test_empty_dataset(self):
        assert self._check(np.array([])) == 'empty'


# ── NaN drop in alpha/beta (mirror of load_mental) ───────────────────────────
class TestNaNDropMental:
    @staticmethod
    def _drop_nan(alphas, betas):
        valid = np.isfinite(alphas) & np.isfinite(betas)
        return alphas[valid], betas[valid], int((~valid).sum())

    def test_no_nan_no_drop(self):
        a, b, dropped = self._drop_nan(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert dropped == 0
        assert len(a) == 3

    def test_nan_in_alpha_dropped(self):
        a, b, dropped = self._drop_nan(
            np.array([1, np.nan, 3], dtype=float),
            np.array([4.0, 5, 6]))
        assert dropped == 1
        assert list(a) == [1.0, 3.0]
        assert list(b) == [4.0, 6.0]

    def test_inf_dropped(self):
        a, b, dropped = self._drop_nan(
            np.array([1, np.inf, 3], dtype=float),
            np.array([4.0, 5, 6]))
        assert dropped == 1

    def test_both_nan_dropped(self):
        a, b, dropped = self._drop_nan(
            np.array([np.nan, np.nan]),
            np.array([np.nan, 5]))
        assert dropped == 2
        assert len(a) == 0


# ── Empty concatenation guard (mirror of load_deap) ──────────────────────────
class TestEmptyConcatenation:
    def test_empty_list_concat_fails(self):
        # numpy raises ValueError when concatenating empty list
        with pytest.raises(ValueError):
            np.concatenate([])

    def test_single_array_concat_works(self):
        result = np.concatenate([np.array([1, 2, 3])])
        assert list(result) == [1, 2, 3]
