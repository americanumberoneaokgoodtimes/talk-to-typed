"""Unit and lifecycle tests for stt/mic.py audio capture backends and process safeguards."""
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from stt import config, mic


def test_put_bounded_overflow_drops_oldest():
    q = queue.Queue(maxsize=3)
    mic._put_bounded(q, 1)
    mic._put_bounded(q, 2)
    mic._put_bounded(q, 3)
    assert q.qsize() == 3

    # Adding 4th item drops oldest (1)
    mic._put_bounded(q, 4)
    assert q.qsize() == 3
    items = [q.get(), q.get(), q.get()]
    assert items == [2, 3, 4]


def test_put_bounded_sentinel_never_blocks_on_full():
    q = queue.Queue(maxsize=2)
    mic._put_bounded(q, "a")
    mic._put_bounded(q, "b")
    assert q.full()

    # Sentinel None should force room and enqueue None without blocking
    mic._put_bounded(q, None)
    first = q.get_nowait()
    second = q.get_nowait()
    assert second is None or first is None


def test_pid_is_alive():
    current_pid = os.getpid()
    assert mic._pid_is_alive(current_pid) is True
    # Negative and absurdly large PIDs
    assert mic._pid_is_alive(-1) is False
    assert mic._pid_is_alive(99999999) is False


def test_detect_structure():
    cfg = config.load()
    res = mic.detect(cfg)
    assert "pulse" in res
    assert "termux" in res
    assert "windows" in res
    for backend, (ok, detail) in res.items():
        assert isinstance(ok, bool)
        assert isinstance(detail, str)


def test_list_dshow_devices_parser(monkeypatch):
    mock_stderr = (
        "[dshow @ 000001] DirectShow video devices (some may be both video and audio devices)\n"
        "[dshow @ 000001]  \"Integrated Camera\"\n"
        "[dshow @ 000001] DirectShow audio devices\n"
        "[dshow @ 000001]  \"Microphone Array (Realtek(R) Audio)\"\n"
        "[dshow @ 000001]     Alternative name \"@device_cm_{...}\"\n"
        "[dshow @ 000001]  \"Stereo Mix (Realtek(R) Audio)\"\n"
    )

    def mock_run(*args, **kwargs):
        m = MagicMock()
        m.stderr = mock_stderr
        m.returncode = 0
        return m

    monkeypatch.setattr(subprocess, "run", mock_run)
    devs = mic.list_dshow_audio_devices()
    assert devs == [
        "Microphone Array (Realtek(R) Audio)",
        "Stereo Mix (Realtek(R) Audio)",
    ]


def test_sounddevice_source_lifecycle():
    cfg = config.load()
    src = mic.SoundDeviceSource(cfg)
    assert src.stream is None
    # Verify stopping when unstarted is safe and non-blocking
    src.stop()
    assert src.q.get_nowait() is None


def test_windows_dshow_source_init():
    cfg = config.load()
    src = mic.WindowsDShowSource(cfg)
    assert src.rec is None
    src.stop()
    assert src.rec is None


def test_pulse_source_port_release():
    cfg = config.load()
    cfg["pulse_port"] = 4720
    mic._release_pulse_port(4720)
    # Releasing an unbound port should be completely silent and not raise
    mic._release_pulse_port(None)
    mic._release_pulse_port(9999)
