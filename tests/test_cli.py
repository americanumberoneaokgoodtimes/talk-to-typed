import sys
import tempfile
from pathlib import Path

import pytest
from stt import cli


def test_format_timestamp_srt():
    assert cli.format_timestamp_srt(0.0) == "00:00:00,000"
    assert cli.format_timestamp_srt(1.234) == "00:00:01,234"
    assert cli.format_timestamp_srt(65.5) == "00:01:05,500"
    assert cli.format_timestamp_srt(3661.123) == "01:01:01,123"


def test_format_timestamp_vtt():
    assert cli.format_timestamp_vtt(0.0) == "00:00:00.000"
    assert cli.format_timestamp_vtt(1.234) == "00:00:01.234"
    assert cli.format_timestamp_vtt(65.5) == "00:01:05.500"
    assert cli.format_timestamp_vtt(3661.123) == "01:01:01.123"


def test_format_srt():
    triples = [
        (0.5, 2.5, "Hello world"),
        (3.0, 5.2, "This is a test transcript")
    ]
    srt_out = cli.format_srt(triples)
    assert "1\n00:00:00,500 --> 00:00:02,500\nHello world" in srt_out
    assert "2\n00:00:03,000 --> 00:00:05,200\nThis is a test transcript" in srt_out


def test_format_vtt():
    triples = [
        (0.5, 2.5, "Hello world"),
        (3.0, 5.2, "This is a test transcript")
    ]
    vtt_out = cli.format_vtt(triples)
    assert vtt_out.startswith("WEBVTT\n")
    assert "1\n00:00:00.500 --> 00:00:02.500\nHello world" in vtt_out
    assert "2\n00:00:03.000 --> 00:00:05.200\nThis is a test transcript" in vtt_out


def test_collect_media_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        sub = tmp_p / "subdir"
        sub.mkdir()
        (tmp_p / "audio1.wav").touch()
        (tmp_p / "audio2.mp3").touch()
        (tmp_p / "notes.txt").touch()
        (sub / "audio3.flac").touch()

        # Non-recursive on tmp_p
        paths_flat = cli.collect_media_paths([str(tmp_p)], recursive=False)
        flat_names = [Path(p).name for p in paths_flat]
        assert "audio1.wav" in flat_names
        assert "audio2.mp3" in flat_names
        assert "notes.txt" not in flat_names
        assert "audio3.flac" not in flat_names

        # Recursive on tmp_p
        paths_rec = cli.collect_media_paths([str(tmp_p)], recursive=True)
        rec_names = [Path(p).name for p in paths_rec]
        assert "audio1.wav" in rec_names
        assert "audio2.mp3" in rec_names
        assert "audio3.flac" in rec_names
        assert "notes.txt" not in rec_names
