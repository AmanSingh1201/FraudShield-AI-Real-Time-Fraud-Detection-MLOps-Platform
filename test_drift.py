import numpy as np

from src.fraudshield.monitoring.drift import psi, classify


def test_psi_zero_for_identical_distributions():
    rng = np.random.RandomState(0)
    ref = rng.normal(0, 1, 5000)
    val = psi(ref, ref.copy())
    assert val < 0.01


def test_psi_high_for_shifted_distribution():
    rng = np.random.RandomState(0)
    ref = rng.normal(0, 1, 5000)
    shifted = rng.normal(5, 1, 5000)
    val = psi(ref, shifted)
    assert val > 0.25


def test_classify_thresholds():
    assert classify(0.05) == "OK"
    assert classify(0.15) == "WARNING"
    assert classify(0.30) == "CRITICAL"
