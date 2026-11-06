"""Unit tests for --range time specification parsing, slice decoding, and timestamp offsetting."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from stt import audio, cli


class TestTimeSpecParsing:
    def test_tagged_formats(self):
        assert audio.parse_time_spec("00h00m07s635z") == pytest.approx(7.635)
        assert audio.parse_time_spec("07s635z") == pytest.approx(7.635)
        assert audio.parse_time_spec("1m30s") == pytest.approx(90.0)
        assert audio.parse_time_spec("2h") == pytest.approx(7200.0)
        assert audio.parse_time_spec("500ms") == pytest.approx(0.5)
        assert audio.parse_time_spec("500z") == pytest.approx(0.5)
        assert audio.parse_time_spec("10s") == pytest.approx(10.0)
        assert audio.parse_time_spec("01h30m15s500ms") == pytest.approx(5415.5)

    def test_clock_formats(self):
        assert audio.parse_time_spec("00:07.635") == pytest.approx(7.635)
        assert audio.parse_time_spec("01:30") == pytest.approx(90.0)
        assert audio.parse_time_spec("01:02:03.500") == pytest.approx(3723.5)
        assert audio.parse_time_spec("00:00:10") == pytest.approx(10.0)

    def test_raw_seconds(self):
        assert audio.parse_time_spec("7.635") == pytest.approx(7.635)
        assert audio.parse_time_spec("120") == pytest.approx(120.0)
        assert audio.parse_time_spec("0") == pytest.approx(0.0)

    def test_invalid_specs(self):
        with pytest.raises(ValueError):
            audio.parse_time_spec("")
        with pytest.raises(ValueError):
            audio.parse_time_spec("invalid")
        with pytest.raises(ValueError):
            audio.parse_time_spec("-5s")


class TestTimeRangeParsing:
    def test_standard_ranges(self):
        start, end = audio.parse_time_range("00h00m07s635z-00h00m10s000z")
        assert start == pytest.approx(7.635)
        assert end == pytest.approx(10.0)

        start, end = audio.parse_time_range("7s-10s")
        assert start == pytest.approx(7.0)
        assert end == pytest.approx(10.0)

        start, end = audio.parse_time_range("01:30-02:00")
        assert start == pytest.approx(90.0)
        assert end == pytest.approx(120.0)

    def test_open_ended_ranges(self):
        start, end = audio.parse_time_range("-8s")
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(8.0)

        start, end = audio.parse_time_range("7s-")
        assert start == pytest.approx(7.0)
        assert end is None

    def test_invalid_ranges(self):
        with pytest.raises(ValueError, match="(?i)empty"):
            audio.parse_time_range("")
        with pytest.raises(ValueError, match="must be greater than start time"):
            audio.parse_time_range("10s-5s")
        with pytest.raises(ValueError, match="must be greater than start time"):
            audio.parse_time_range("5s-5s")
        with pytest.raises(ValueError):
            audio.parse_time_range("a-b-c")


class TestAudioSliceDecode:
    @patch("subprocess.run")
    def test_ffmpeg_arguments_with_range(self, mock_run):
        fake_res = MagicMock()
        fake_audio = np.zeros(16000 * 2, dtype=np.float32).tobytes()
        fake_res.stdout = fake_audio
        fake_res.stderr = b""
        fake_res.returncode = 0
        mock_run.return_value = fake_res

        samples = audio.decode("dummy.mp3", start_sec=7.635, end_sec=10.0)
        assert samples is not None

        call_args = mock_run.call_args[0][0]
        assert "-ss" in call_args
        idx_ss = call_args.index("-ss")
        assert call_args[idx_ss + 1] == "7.635"
        assert "-t" in call_args
        idx_t = call_args.index("-t")
        assert pytest.approx(float(call_args[idx_t + 1])) == 2.365


class TestCliRangeIntegration:
    @patch("stt.audio.decode")
    @patch("stt.robust.transcribe")
    @patch("stt.asr.offline")
    def test_cli_offsets_timestamps(self, mock_offline, mock_transcribe, mock_decode, capsys):
        # 16000 samples = 1 sec
        mock_decode.return_value = np.zeros(16000, dtype=np.float32)
        mock_offline.return_value = MagicMock()
        mock_transcribe.return_value = [(0.5, 0.9, "test utterance")]

        cfg = {"models_dir": ".", "aggressiveness": 0, "threads": 1, "model": "fake"}
        args = MagicMock()
        args.interactive = False
        args.paths = ["dummy.mp3"]
        args.range = "07s-10s"
        args.all = False
        args.dynamic = False
        args.auto = False
        args.subnoise = False
        args.aggressiveness = None
        args.robust = False
        args.blank_penalty = None
        args.lang = None
        args.lang_source = None
        args.lang_target = None
        args.translate = False
        args.threads = 1
        args.model = None
        args.jsonl = False
        args.srt = False
        args.vtt = False
        args.sidecar = False
        args.timestamps = True
        args.out = None
        args.verbose = False
        args.recursive = False

        with patch("stt.cli.collect_media_paths", return_value=["dummy.mp3"]):
            cli.cmd_file(args, cfg)

        out, _ = capsys.readouterr()
        # Segment (0.5, 0.9) offset by 7.0s must display [00:07.500 -> 00:07.900]
        assert "00:07.500 -> 00:07.900" in out
        assert "test utterance" in out
