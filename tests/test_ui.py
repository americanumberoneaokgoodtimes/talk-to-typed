import http.client
import json
import threading
import time
from pathlib import Path

import pytest
from stt import config, ui


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
    payload = json.dumps({"server_path": "/invalid/nonexistent/path"})
    conn.request("POST", "/api/transcribe_path", body=payload, headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    assert res.status == 400
    conn.close()


def test_ui_api_audio(ui_server, tmp_path):
    # Test valid audio file streaming
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

    # Test nonexistent audio path
    conn = http.client.HTTPConnection(host, int(port))
    conn.request("GET", "/api/audio?path=/does/not/exist.wav")
    res = conn.getresponse()
    assert res.status == 404
    conn.close()
