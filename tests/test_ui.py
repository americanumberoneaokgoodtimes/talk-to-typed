import http.client
import json
import threading
import time
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from stt import config, robust, ui


@pytest.fixture(scope="module")
def ui_server():
    cfg = dict(config.DEFAULTS)
    # Pick an ephemeral port
    server = ui.ThreadedHTTPServer(("127.0.0.1", 0), ui.STTRequestHandler)
    server.cfg = cfg
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    yield f"127.0.0.1:{port}"

    server.shutdown()
    server.server_close()


def test_ui_index(ui_server):
    host, port = ui_server.split(":")
    conn = http.client.HTTPConnection(host, int(port))
    conn.request("GET", "/")
    res = conn.getresponse()
    assert res.status == 200
    content = res.read().decode("utf-8")
    assert "<!DOCTYPE html>" in content
    assert "talk-to-typed" in content
    conn.close()


def test_ui_api_status(ui_server):
    host, port = ui_server.split(":")
    conn = http.client.HTTPConnection(host, int(port))
    conn.request("GET", "/api/status")
    res = conn.getresponse()
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert "version" in data
    assert "models_dir" in data
    conn.close()


def test_ui_api_transcribe_invalid_path(ui_server):
    host, port = ui_server.split(":")
    conn = http.client.HTTPConnection(host, int(port))
    payload = json.dumps({"server_path": "/invalid/nonexistent/path.wav"})
    conn.request("POST", "/api/transcribe_path", body=payload, headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    assert res.status == 400
    conn.close()


def test_ui_api_audio_and_security_guards(ui_server, tmp_path):
    # 1. Valid audio file streaming + HTTP Range seeking
    test_audio = tmp_path / "sample.wav"
    test_audio.write_bytes(b"RIFF....WAVEfmt ....data....")

    host, port = ui_server.split(":")
    conn = http.client.HTTPConnection(host, int(port))
    conn.request("GET", f"/api/audio?path={test_audio}")
    res = conn.getresponse()
    assert res.status == 200
    assert res.getheader("Content-Type") == "audio/wav"
    body = res.read()
    assert body == b"RIFF....WAVEfmt ....data...."
    conn.close()

    conn = http.client.HTTPConnection(host, int(port))
    conn.request("GET", f"/api/audio?path={test_audio}", headers={"Range": "bytes=0-3"})
    res = conn.getresponse()
    assert res.status == 206
    assert res.read() == b"RIFF"
    conn.close()

    # 2. Non-media extension blocked (prevents arbitrary text/credential reads)
    secret_txt = tmp_path / "secrets.txt"
    secret_txt.write_text("TOP_SECRET", encoding="utf-8")
    conn = http.client.HTTPConnection(host, int(port))
    conn.request("GET", f"/api/audio?path={secret_txt}")
    res = conn.getresponse()
    assert res.status == 403
    res.read()
    conn.close()

    # 3. Sensitive dot-directory blocked even with media extension
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    spoof_wav = ssh_dir / "id_rsa.wav"
    spoof_wav.write_bytes(b"RIFF....WAVE")
    conn = http.client.HTTPConnection(host, int(port))
    conn.request("GET", f"/api/audio?path={spoof_wav}")
    res = conn.getresponse()
    assert res.status == 403
    res.read()
    conn.close()

    # 4. Cross-origin DNS-rebinding / CSRF blocked
    conn = http.client.HTTPConnection(host, int(port))
    conn.request(
        "GET",
        f"/api/audio?path={test_audio}",
        headers={"Origin": "https://evil.example"},
    )
    res = conn.getresponse()
    assert res.status == 403
    res.read()
    conn.close()

    # 5. Nonexistent valid media extension returns 404
    conn = http.client.HTTPConnection(host, int(port))
    conn.request("GET", "/api/audio?path=/does/not/exist.wav")
    res = conn.getresponse()
    assert res.status == 404
    res.read()
    conn.close()


def test_ui_api_transcribe_path_sse_no_shadowing(ui_server, tmp_path):
    """Verify SSE transcription runs cleanly without 'asr' UnboundLocalError and emits burial diagnostics."""
    wav_path = tmp_path / "clip.wav"
    pcm = (np.sin(2.0 * np.pi * 220.0 * np.arange(16000) / 16000.0) * 8000).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())

    def fake_transcribe(rec, samples, cfg, **kwargs):
        robust._set_last_segment_diagnostics([{
            "start": 0.1,
            "end": 0.9,
            "location": "[00:00.100 -> 00:00.900]",
            "snr_db": -1.4,
            "burial_below_floor_db": 1.4,
            "depth_vs_foreground_db": -16.2,
            "layer": "subnoise-buried",
            "recovery_source": "gap-subnoise",
        }])
        return [(0.1, 0.9, "buried test phrase")]

    host, port = ui_server.split(":")
    with patch("stt.asr.get_offline", return_value=object()), patch(
        "stt.robust.transcribe", side_effect=fake_transcribe
    ):
        conn = http.client.HTTPConnection(host, int(port))
        payload = json.dumps({"server_path": str(wav_path), "mode": "standard", "level": 2})
        conn.request(
            "POST",
            "/api/transcribe_path",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Origin": f"http://127.0.0.1:{port}",
            },
        )
        res = conn.getresponse()
        assert res.status == 200
        raw_sse = res.read().decode("utf-8")
        conn.close()

    assert "event: error" not in raw_sse
    assert "event: segment" in raw_sse
    assert "buried test phrase" in raw_sse
    assert "subnoise-buried" in raw_sse
    assert "event: done" in raw_sse
