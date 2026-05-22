"""
Pytest configuration — adds project root to sys.path so tests can import
calibration, experiment, eeg_model directly without installation.
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
