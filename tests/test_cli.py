import sys
import tempfile
from pathlib import Path

import pytest
from stt import cli


def test_format_timestamp():
    assert cli.format_timestamp(0.0) == "00:00.000"
    assert cli.format_timestamp(1.234) == "00:01.234"
    assert cli.format_timestamp(65.5) == "01:05.500"
    assert cli.format_timestamp(3661.123) == "01:01:01.123"


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


def test_color_enabled_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert not cli._color_enabled()

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert cli._color_enabled()


def test_color_help_formatter_plain_when_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    formatter = cli.ColorHelpFormatter("stt")
    help_out = formatter.format_help()
    assert "\033[" not in help_out


def test_color_help_formatter_colorized_when_force_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    formatter = cli.ColorHelpFormatter("stt file")
    dummy_text = (
        "usage: stt file [-h] [--dynamic]\n\n"
        "ACOUSTIC ENHANCEMENT:\n"
        "  --dynamic                   run multi-model ensemble (default: false)\n\n"
        "EXAMPLES & USAGE:\n"
        "  stt file audio.mp3\n"
        "      Transcribe audio file.\n"
    )
    colorized = formatter._colorize(dummy_text)
    assert "\033[1;36mACOUSTIC ENHANCEMENT:\033[0m" in colorized
    assert "\033[1;36m--dynamic\033[0m" in colorized
    assert "\033[1;33mEXAMPLES & USAGE:\033[0m" in colorized
    assert "\033[1;32mstt file\033[0m" in colorized


def test_cli_help_sections_and_examples(capsys, monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    # stt file --help
    with pytest.raises(SystemExit):
        cli.main(["file", "--help"])
    captured = capsys.readouterr().out
    assert "POSITIONAL ARGUMENTS:" in captured
    assert "INPUT & OUTPUT OPTIONS:" in captured
    assert "ACOUSTIC ENHANCEMENT & SUBFLOOR RECOVERY:" in captured
    assert "NOISE SUPPRESSION & AGGRESSIVENESS:" in captured
    assert "MULTILINGUAL & TRANSLATION:" in captured
    assert "HARDWARE & ENGINE CONTROLS:" in captured
    assert "DIAGNOSTICS & WIZARD:" in captured
    assert "EXAMPLES & USAGE:" in captured
    assert "stt file audio.wav --dynamic" in captured

    # stt mic --help
    with pytest.raises(SystemExit):
        cli.main(["mic", "--help"])
    captured_mic = capsys.readouterr().out
    assert "MICROPHONE CAPTURE MODES:" in captured_mic
    assert "STREAMING & OUTPUT OPTIONS:" in captured_mic
    assert "ACOUSTIC ENHANCEMENT & SUBFLOOR RECOVERY:" in captured_mic
    assert "MICROPHONE HARDWARE & DRIVER CONTROLS:" in captured_mic
    assert "EXAMPLES & USAGE:" in captured_mic
    assert "stt mic --whisper" in captured_mic


def test_cli_declick_flags():
    ap = cli._build_parser()
    args_default = ap.parse_args(["file", "dummy.wav"])
    assert args_default.declick is True
    assert args_default.declick_threshold == 4.8

    args_nodeclick = ap.parse_args(["file", "dummy.wav", "--no-declick"])
    assert args_nodeclick.declick is False

    args_custom = ap.parse_args(["file", "dummy.wav", "--declick", "--declick-threshold", "3.5"])
    assert args_custom.declick is True
    assert args_custom.declick_threshold == 3.5

    # Also test mic parser
    args_mic = ap.parse_args(["mic", "--no-declick", "--declick-threshold", "5.2"])
    assert args_mic.declick is False
    assert args_mic.declick_threshold == 5.2


