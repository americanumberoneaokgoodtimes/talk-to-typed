"""Live microphone backends: PulseAudio (OpenSL ES source) and Termux:API chunks."""
import atexit
import collections
import os
import queue
import shutil
import sys
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import asr, audio, robust

BLOCK = 512  # samples per VAD window (32 ms)


def _which(cfg: dict, name: str) -> str | None:
    # Only Termux's pulseaudio ships module-sles-source (the phone mic); Kali's does not.
    t = Path(cfg["termux_bin"]) / name
    if name == "pulseaudio":
        return str(t) if t.exists() else None
    return shutil.which(name) or (str(Path(cfg["termux_bin"]) / name) if (Path(cfg["termux_bin"]) / name).exists() else None)


def _kill(p: subprocess.Popen | None):
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(3)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


# ---------------------------------------------------------------- PulseAudio

class PulseSource:
    """Private PulseAudio server with module-sles-source plus a parec reader."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.server = f"tcp:127.0.0.1:{cfg['pulse_port']}"
        self.env = dict(os.environ, PULSE_SERVER=self.server)
        self.pa: subprocess.Popen | None = None
        self.rec: subprocess.Popen | None = None

    def _sources(self) -> str:
        pactl = _which(self.cfg, "pactl")
        if not pactl:
            return ""
        r = subprocess.run([pactl, "list", "short", "sources"], env=self.env, capture_output=True, text=True, timeout=5)
        return r.stdout if r.returncode == 0 else ""

    def start(self):
        if not _which(self.cfg, "parec"):
            raise RuntimeError("parec not found")
        if "OpenSL_ES_source" not in self._sources():
            pa = _which(self.cfg, "pulseaudio")
            if not pa:
                raise RuntimeError("pulseaudio binary not found")
            self.pa = subprocess.Popen(
                [pa, "-n", "--daemonize=no", "--exit-idle-time=-1", "--use-pid-file=no", "--log-target=null",
                 "--load=module-native-protocol-tcp port=%d auth-anonymous=1 listen=127.0.0.1" % self.cfg["pulse_port"],
                 "--load=module-sles-source"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            )
            for _ in range(30):
                time.sleep(0.2)
                if self.pa.poll() is not None:
                    raise RuntimeError("pulseaudio exited during startup")
                if "OpenSL_ES_source" in self._sources():
                    break
            else:
                self.stop()
                raise RuntimeError("OpenSL_ES_source did not appear (mic permission?)")
        self.rec = subprocess.Popen(
            [_which(self.cfg, "parec"), "--format=s16le", "--rate=16000", "--channels=1", "--raw",
             "--device=OpenSL_ES_source", "--latency-msec=20"],
            env=self.env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, bufsize=0,
        )

    def blocks(self):
        """Yield float32 blocks of BLOCK samples.

        A reader thread drains parec into an unbounded queue: if decoding runs slower than real
        time the pipe must never fill up, or parec drops audio.
        """
        q: queue.Queue = queue.Queue()
        stdout = self.rec.stdout

        def reader():
            nbytes = BLOCK * 2
            buf = b""
            while True:
                d = stdout.read(nbytes - len(buf))
                if not d:
                    q.put(None)
                    return
                buf += d
                if len(buf) == nbytes:
                    q.put(np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0)
                    buf = b""

        threading.Thread(target=reader, daemon=True).start()
        self.backlog = q.qsize
        while True:
            b = q.get()
            if b is None:
                return
            yield b

    def stop(self):
        _kill(self.rec)
        _kill(self.pa)
        self.rec = self.pa = None


def capture_pulse(cfg: dict, seconds: float) -> np.ndarray:
    src = PulseSource(cfg)
    out = []
    try:
        src.start()
        n = 0
        for b in src.blocks():
            out.append(b)
            n += len(b)
            if n >= seconds * audio.SR:
                break
    finally:
        src.stop()
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def run_pulse(cfg: dict, args, emit, src=None):
    rec = asr.streaming(cfg, args.model, args.threads)
    vad = audio.make_vad(dict(cfg, vad_threshold=cfg['mic_vad_threshold'], vad_min_speech=cfg['mic_vad_min_speech']))
    src = src or PulseSource(cfg)
    src.start()
    agc = audio.AGC(cfg['mic_agc_target'], cfg['mic_agc_max_gain']) if cfg['mic_agc'] else None
    end_blocks = max(4, int(cfg['vad_min_silence'] * audio.SR / BLOCK))
    preroll = collections.deque(maxlen=max(1, int(cfg['mic_preroll'] * audio.SR / BLOCK)))
    stream = None
    silent = 0
    last = ""
    started = 0.0
    tail = np.zeros(int(0.4 * audio.SR), dtype=np.float32)

    def finish():
        nonlocal stream, last, silent
        stream.accept_waveform(audio.SR, tail)
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        text = rec.get_result(stream).strip()
        if args.partial:
            print("\r\033[K", end="", flush=True)
        if text:
            emit(text, started, time.time())
        stream = None
        last = ""
        silent = 0
        vad.reset()
        preroll.clear()

    raw, nblk = [], 0
    try:
        for b in src.blocks():
            if args.save:
                raw.append(b)
            nblk += 1
            if args.debug and nblk % 31 == 0:
                print(f"\r\033[K[lvl] rms {float(np.sqrt(np.mean(b*b))):.4f} gain {agc.gain if agc else 1:.1f} vad {int(vad.is_speech_detected())} utt {stream is not None}", file=sys.stderr, flush=True)
            if agc:
                b = agc(b)
            vad.accept_waveform(b)
            speech = vad.is_speech_detected()
            if stream is None:
                preroll.append(b)
                if speech:
                    stream = rec.create_stream()
                    started = time.time()
                    stream.accept_waveform(audio.SR, np.concatenate(preroll))
                    preroll.clear()
                else:
                    continue
            else:
                stream.accept_waveform(audio.SR, b)
                silent = 0 if speech else silent + 1
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            if args.partial:
                t = rec.get_result(stream).strip()
                if t != last:
                    last = t
                    print("\r\033[K" + t, end="", flush=True)
            if silent >= end_blocks or time.time() - started > 30:
                finish()
        if stream is not None:
            finish()
    finally:
        src.stop()
        if args.save and raw:
            import soundfile as sf
            sf.write(args.save, np.concatenate(raw), audio.SR)
            print(f"[stt] saved raw audio to {args.save}", file=sys.stderr)


def run_pulse_utterance(cfg: dict, args, emit, src=None):
    """Live mic -> VAD utterances -> (denoise/normalize) -> offline recognizer. Better for whispering and noise.

    Text appears when you pause (no partials), but Parakeet is both faster than the streaming model on this CPU
    and more accurate. Recognition uses the raw audio; AGC only helps the VAD hear quiet starts.
    """
    distant = getattr(args, "distant", False)
    level = args.aggressiveness if args.aggressiveness is not None else (5 if distant else 3 if args.robust or args.whisper else int(cfg["aggressiveness"]))
    level = max(level, 1) if (args.whisper or distant) else level
    rec = asr.offline(cfg, args.model, args.threads)
    thr = min(cfg["mic_vad_threshold"], 0.25) if distant else cfg["mic_vad_threshold"]
    vad = audio.make_vad(dict(cfg, vad_threshold=thr, vad_min_speech=cfg["mic_vad_min_speech"]))
    src = src or PulseSource(cfg)
    src.start()
    blocks = src.blocks()
    floor = None
    if not getattr(args, "no_calibrate", False) and cfg["mic_calibrate_seconds"] > 0:
        # Measure the room noise first; the AGC then knows its floor immediately instead of learning it.
        print("[stt] Calibrating, please don't speak ...", file=sys.stderr, flush=True)
        need, got = int(cfg["mic_calibrate_seconds"] * audio.SR / BLOCK), []
        for b0 in blocks:
            got.append(b0)
            if len(got) >= need:
                break
        if got:
            prof = audio.NoiseProfile(np.concatenate(got))
            floor = prof.rms
            print(f"[stt] OK, noise profile applied (room noise RMS {floor:.4f})", file=sys.stderr, flush=True)
    agc = (audio.AGC(cfg["mic_agc_target"], cfg["mic_agc_max_gain"], 0.05 if distant else 0.02, floor) if cfg["mic_agc"] else None)
    end_blocks = max(4, int(cfg["vad_min_silence"] * audio.SR / BLOCK))
    n_pre = max(1, int((1.5 if distant else 1.0) * max(cfg["mic_preroll"], 1.0) * audio.SR / BLOCK))
    preroll = collections.deque(maxlen=n_pre)
    buf, active, silent, started = [], False, 0, 0.0
    pad = np.zeros(int(0.3 * audio.SR), dtype=np.float32)
    raw, nblk = [], 0

    def finish():
        nonlocal buf, active, silent
        x = np.concatenate([pad] + buf + [pad])
        buf, active, silent = [], False, 0
        vad.reset()
        preroll.clear()
        for s, e, text in robust.transcribe(rec, x, cfg, level):
            emit(text, started + s, started + e)

    try:
        for b0 in blocks:
            if args.save:
                raw.append(b0)
            nblk += 1
            b = agc(b0) if agc else b0
            if args.debug and nblk % 31 == 0:
                print(f"\r\033[K[lvl] rms {float(np.sqrt(np.mean(b0*b0))):.4f} gain {agc.gain if agc else 1:.1f} vad {int(vad.is_speech_detected())} utt {active}", file=sys.stderr, flush=True)
            vad.accept_waveform(b)
            speech = vad.is_speech_detected()
            if not active:
                preroll.append(b0)
                if speech:
                    active, started, buf = True, time.time(), list(preroll)
                    preroll.clear()
                continue
            buf.append(b0)
            silent = 0 if speech else silent + 1
            if silent >= end_blocks or time.time() - started > 30:
                finish()
        if active:
            finish()
    finally:
        src.stop()
        if args.save and raw:
            import soundfile as sf
            sf.write(args.save, np.concatenate(raw), audio.SR)
            print(f"[stt] saved raw audio to {args.save}", file=sys.stderr)


# ---------------------------------------------------------------- Termux:API

def _tmx(cfg: dict) -> str:
    p = _which(cfg, "termux-microphone-record")
    if not p:
        raise RuntimeError("termux-microphone-record not found")
    return p


def record_chunk(cfg: dict, seconds: int) -> Path | None:
    """Record one chunk via Termux:API; returns the file path (caller deletes it)."""
    tmp = Path(cfg["termux_tmp"])
    tmp.mkdir(parents=True, exist_ok=True)
    f = tmp / f"chunk-{os.getpid()}-{time.monotonic_ns()}.m4a"
    cmd = _tmx(cfg)
    r = subprocess.run([cmd, "-f", str(f), "-l", str(seconds), "-e", "aac", "-r", "16000", "-c", "1"],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or "error" in (r.stdout + r.stderr).lower():
        raise RuntimeError((r.stdout + r.stderr).strip() or "termux-microphone-record failed")
    deadline = time.time() + seconds + 6
    time.sleep(0.5)
    while time.time() < deadline:
        info = subprocess.run([cmd, "-i"], capture_output=True, text=True, timeout=30).stdout
        if '"isRecording": false' in info:
            break
        time.sleep(0.2)
    else:
        subprocess.run([cmd, "-q"], capture_output=True)
    return f if f.exists() else None


def capture_termux(cfg: dict, seconds: int) -> np.ndarray:
    f = record_chunk(cfg, int(seconds))
    if f is None:
        return np.zeros(0, np.float32)
    try:
        return audio.decode(str(f))
    finally:
        f.unlink(missing_ok=True)


def run_termux(cfg: dict, args, emit):
    level = args.aggressiveness if args.aggressiveness is not None else (3 if args.robust else int(cfg["aggressiveness"]))
    rec = asr.offline(cfg, args.model, args.threads)
    q: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()

    def worker():
        while True:
            item = q.get()
            if item is None:
                return
            samples, t0 = item
            for s, e, text in robust.transcribe(rec, samples, cfg, level):
                emit(text, t0 + s, t0 + e)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    n = int(cfg["mic_chunk_seconds"])
    try:
        while not stop.is_set():
            t0 = time.time()
            samples = capture_termux(cfg, n)
            if len(samples) and float(np.sqrt(np.mean(samples**2))) > 1e-4:
                q.put((samples, t0))
    finally:
        try:
            subprocess.run([_tmx(cfg), "-q"], capture_output=True, timeout=5)
        except Exception:
            pass
        q.put(None)
        th.join(timeout=60)
        for f in Path(cfg["termux_tmp"]).glob(f"chunk-{os.getpid()}-*"):
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------- detection

def detect(cfg: dict) -> dict:
    """Return {backend: (ok, detail)} without recording."""
    res = {}
    pa, parec = _which(cfg, "pulseaudio"), _which(cfg, "parec")
    res["pulse"] = (bool(pa and parec), f"pulseaudio={pa} parec={parec}")
    cmd = _which(cfg, "termux-microphone-record")
    ok, detail = False, "termux-microphone-record not found"
    if cmd:
        try:
            r = subprocess.run([cmd, "-i"], capture_output=True, text=True, timeout=30)
            ok = '"isRecording"' in r.stdout
            detail = cmd if ok else (r.stdout + r.stderr).strip()[:100]
        except Exception as e:
            detail = str(e)
    res["termux"] = (ok, detail)
    return res
