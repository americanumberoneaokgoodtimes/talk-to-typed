import numpy as np
import pytest

from stt import config, robust


def test_auto_tune_settings_short():
    res = robust.auto_tune_settings(10.0)
    assert res["win_sec"] == 5.0
    assert res["step_sec"] == 1.25
    assert res["overlap_pct"] == 75
    assert res["low_hz"] == 120.0
    assert res["blank_penalty"] == 0.7
    assert len(res["reasoning"]) > 0


def test_auto_tune_settings_medium():
    res = robust.auto_tune_settings(60.0)
    assert res["win_sec"] == 5.0
    assert res["step_sec"] == 1.75
    assert res["overlap_pct"] == 65
    assert res["blank_penalty"] == 0.6


def test_auto_tune_settings_long():
    res = robust.auto_tune_settings(180.0)
    assert res["win_sec"] == 6.0
    assert res["step_sec"] == 3.0
    assert res["overlap_pct"] == 50
    assert res["low_hz"] == 130.0
    assert res["blank_penalty"] == 0.5


def test_score_coherence():
    # Empty
    score, reason = robust.score_coherence("", 5.0)
    assert score == 0.0
    assert reason == "empty"

    # Too fast hallucination (>5 wps)
    fast_text = "one two three four five six seven eight nine ten"
    score, reason = robust.score_coherence(fast_text, 1.0)
    assert score < 0
    assert "hallucination" in reason

    # Repetitive single token
    rep_text = "the the the the the the"
    score, reason = robust.score_coherence(rep_text, 3.0)
    assert score <= 0.05
    assert "repetitive token" in reason

    # Repetitive bigram (where single token dom_ratio < 0.5)
    rep_bi = "thank you ok thank you sir"
    score, reason = robust.score_coherence(rep_bi, 3.0)
    assert score <= 0.05
    assert "repetitive phrase" in reason

    # Coherent text
    good_text = "The quick brown fox jumps over the lazy dog"
    score, reason = robust.score_coherence(good_text, 5.0)
    assert score > 1.0
    assert reason == "coherent"


def test_pregain():
    # Empty array
    empty = np.array([], dtype=np.float32)
    assert robust.pregain(empty) == 1.0

    # Very quiet signal (amplified up to max_gain)
    zeros = np.zeros(1024, dtype=np.float32)
    assert robust.pregain(zeros, max_gain=50.0) == 50.0

    # Loud signal (never attenuates below 1.0)
    loud = np.ones(2048, dtype=np.float32) * 0.8
    gain = robust.pregain(loud, target=0.1)
    assert gain == 1.0


def test_variants():
    cfg = dict(config.DEFAULTS)
    for lvl in [1, 2, 3, 4, 5]:
        v = robust._variants(lvl, cfg)
        assert isinstance(v, list)
        assert len(v) >= 1
        for variant in v:
            assert "denoise" in variant
            assert "vad" in variant
