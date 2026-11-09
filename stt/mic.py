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

# Port management for PulseAudio
_PULSE_PORT_RANGE = range(4714, 4725)  # Try ports 4714-4724
_PULSE_LOCK_DIR = Path(tempfile.gettempdir()) / "stt_pulse_locks"
_PULSE_LOCK_DIR.mkdir(parents=True, exist_ok=True)


def _is_termux_env(cfg: dict) -> bool:
    """Return True when running inside Android Termux or Kali Userland with Termux binaries."""
    tbin = cfg.get("termux_bin", "/data/data/com.termux/files/usr/bin")
    return bool("TERMUX_VERSION" in os.environ or (tbin and Path(tbin).exists()))


def which(cfg: dict, name: str) -> str | None:
    t = Path(cfg["termux_bin"]) / name
    if name == "pulseaudio" and _is_termux_env(cfg):
        # On Android/Kali Userland, only Termux's pulseaudio ships module-sles-source.
        return str(t) if t.exists() else None
    return shutil.which(name) or (str(t) if t.exists() else None)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False


def _put_bounded(q: queue.Queue, item) -> None:
    """Enqueue an audio block into a bounded queue without deadlocking; drop oldest frame on overflow."""
    if item is None:
        while True:
            try:
                q.put_nowait(None)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def _kill(p: subprocess.Popen | None):
    if p is None or p.poll() is not None:
        return
    try:
        p.terminate()
        p.wait(3)
    except subprocess.TimeoutExpired:
        try:
            p.kill()
            p.wait(2)
        except OSError:
            pass
    except OSError:
        pass


def _acquire_pulse_port(cfg: dict) -> int:
    """Find and lock an available PulseAudio port."""
    import socket

    for port in _PULSE_PORT_RANGE:
        lock_file = _PULSE_LOCK_DIR / f"port_{port}.lock"
        for _attempt in range(2):
            try:
                fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(str(os.getpid()))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    try:
                        sock.bind(("127.0.0.1", port))
                        cfg["pulse_port"] = port
                        return port
                    except OSError:
                        lock_file.unlink(missing_ok=True)
                        break
            except FileExistsError:
                try:
                    with open(lock_file) as f:
                        pid = int(f.read().strip())
                    if _pid_is_alive(pid):
                        break
                except (ValueError, OSError):
                    pass
                lock_file.unlink(missing_ok=True)
    raise RuntimeError("No available PulseAudio ports in range 4714-4724")


def _release_pulse_port(port: int | None):
    """Release the port lock."""
    if port is None:
        return
    lock_file = _PULSE_LOCK_DIR / f"port_{port}.lock"
    try:
        lock_file.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------- PulseAudio

class PulseSource:
    """PulseAudio capture: Termux private SLES server on Android, or native PulseAudio/PipeWire on Linux."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._use_termux_sles = _is_termux_env(cfg)
        self._port: int | None = _acquire_pulse_port(cfg) if self._use_termux_sles else None
        self.server = f"tcp:127.0.0.1:{self._port}" if self._port is not None else None
        self.env = dict(os.environ, PULSE_SERVER=self.server) if self.server else dict(os.environ)
        self.pa: subprocess.Popen | None = None
        self.rec: subprocess.Popen | None = None
        atexit.register(self._cleanup)
        if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
            signal.signal(signal.SIGTERM, self._signal_handler)
        if signal.getsignal(signal.SIGINT) is signal.SIG_DFL:
            signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._cleanup()
        sys.exit(1)

    def _cleanup(self):
        _kill(self.rec)
        _kill(self.pa)
        self.rec = self.pa = None
        _release_pulse_port(self._port)

    def _sources(self) -> str:
        pactl = which(self.cfg, "pactl")
        if not pactl:
            return ""
        try:
            r = subprocess.run([pactl, "list", "short", "sources"], env=self.env, capture_output=True, text=True, timeout=5)
            return r.stdout if r.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def start(self):
        parec = which(self.cfg, "parec")
        if not parec:
            raise RuntimeError("parec not found")

        cmd = [parec, "--format=s16le", "--rate=16000", "--channels=1", "--raw", "--latency-msec=20"]
        if self._use_termux_sles:
            if "OpenSL_ES_source" not in self._sources():
                pa = which(self.cfg, "pulseaudio")
                if not pa:
                    raise RuntimeError("pulseaudio binary not found")
                try:
                    self.pa = subprocess.Popen(
                        [
                            pa, "-n", "--daemonize=no", "--exit-idle-time=-1", "--use-pid-file=no", "--log-target=null",
                            "--load=module-native-protocol-tcp port=%d auth-anonymous=1 listen=127.0.0.1" % self._port,
                            "--load=module-sles-source",
                        ],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                    )
                except OSError as e:
                    raise RuntimeError(f"failed to start pulseaudio: {e}")
                for _ in range(30):
                    time.sleep(0.2)
                    if self.pa.poll() is not None:
                        raise RuntimeError("pulseaudio exited during startup")
                    if "OpenSL_ES_source" in self._sources():
                        break
                else:
                    self.stop()
                    raise RuntimeError("OpenSL_ES_source did not appear (mic permission?)")
            cmd.append("--device=OpenSL_ES_source")

        try:
            self.rec = subprocess.Popen(
                cmd,
                env=self.env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, bufsize=0,
            )
        except OSError as e:
            self.stop()
            raise RuntimeError(f"failed to start parec: {e}")

    def blocks(self):
        """Yield float32 blocks of BLOCK samples via a bounded reader queue."""
        q: queue.Queue = queue.Queue(maxsize=1024)
        stdout = self.rec.stdout

        def reader():
            nbytes = BLOCK * 2
            buf = b""
            while True:
                try:
                    d = stdout.read(nbytes - len(buf))
                    if not d:
                        _put_bounded(q, None)
                        return
                    buf += d
                    if len(buf) == nbytes:
                        _put_bounded(q, np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0)
                        buf = b""
                except (OSError, ValueError):
                    _put_bounded(q, None)
                    return

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
        _release_pulse_port(self._port)


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
    if not args.jsonl:
        print("[stt] Loading streaming model...", file=sys.stderr, flush=True)
    rec = asr.streaming(cfg, args.model, args.threads)
    if not args.jsonl:
        print("\r\033[K[stt] Model loaded", file=sys.stderr, flush=True)
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
    do_declick = getattr(args, "declick", True)
    thresh_arg = getattr(args, "declick_threshold", None)
    thresh = float(4.8 if thresh_arg is None else thresh_arg)
    declicker = audio.StreamingDeclicker(sr=audio.SR, threshold_sigma=thresh) if do_declick else None
    try:
        for b in src.blocks():
            if declicker:
                b = declicker.process(b)
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
    """Live mic -> VAD utterances -> (denoise/normalize) -> offline recognizer. Better for whispering and noise."""
    distant = getattr(args, "distant", False)
    level = args.aggressiveness if args.aggressiveness is not None else (5 if distant else 3 if args.robust or args.whisper else int(cfg["aggressiveness"]))
    level = max(level, 1) if (args.whisper or distant) else level
    bp_arg = getattr(args, "blank_penalty", None)
    blank_penalty = float(cfg.get("blank_penalty", 0.6) if bp_arg is None else bp_arg)
    if not args.jsonl:
        print("[stt] Loading offline model...", file=sys.stderr, flush=True)
    rec = asr.get_offline(cfg, args.model, args.threads, blank_penalty=blank_penalty)
    if not args.jsonl:
        print("\r\033[K[stt] Model loaded", file=sys.stderr, flush=True)
    if distant:
        thr = cfg.get("mic_vad_threshold_distant", 0.20)
    elif args.whisper:
        thr = cfg.get("mic_vad_threshold_whisper", 0.25)
    else:
        thr = cfg["mic_vad_threshold"]
    vad = audio.make_vad(dict(cfg, vad_threshold=thr, vad_min_speech=cfg["mic_vad_min_speech"]))
    src = src or PulseSource(cfg)
    src.start()
    blocks = src.blocks()
    floor = None
    prof = None
    if not getattr(args, "no_calibrate", False) and cfg["mic_calibrate_seconds"] > 0:
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
    do_declick = getattr(args, "declick", True)
    thresh_arg = getattr(args, "declick_threshold", None)
    thresh = float(4.8 if thresh_arg is None else thresh_arg)
    declicker = audio.StreamingDeclicker(sr=audio.SR, threshold_sigma=thresh) if do_declick else None

    def finish():
        nonlocal buf, active, silent
        x = np.concatenate([pad] + buf + [pad])
        buf, active, silent = [], False, 0
        vad.reset()
        preroll.clear()
        for s, e, text in robust.transcribe(rec, x, cfg, level, noise=prof):
            emit(text, started + s, started + e)

    try:
        for b0 in blocks:
            if declicker:
                b0 = declicker.process(b0)
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


def run_pulse_subnoise(cfg: dict, args, emit, src=None):
    """Continuous sliding ring-buffer formant scan on live mic input for sub-noise speech.

    Uses zero-phase Butterworth HPF, hum notching, whisper formant lifting, noise-bounded AGC,
    and parallel multi-stream batch decoding (`asr.transcribe_batch_detailed`).
    """
    bp_arg = getattr(args, "blank_penalty", None)
    blank_penalty = float(cfg.get("blank_penalty", 0.65) if bp_arg is None else bp_arg)
    is_dynamic = bool(getattr(args, "dynamic", False) or getattr(args, "auto", False))
    if not args.jsonl:
        print("[stt] Loading offline model...", file=sys.stderr, flush=True)
    rec = asr.get_offline(
        cfg,
        args.model,
        args.threads,
        blank_penalty=blank_penalty,
        decoding_method="modified_beam_search",
        max_active_paths=16,
    )
    if not args.jsonl:
        print("\r\033[K[stt] Model loaded", file=sys.stderr, flush=True)
    src = src or PulseSource(cfg)
    src.start()

    low_arg = getattr(args, "rumble_cutoff", None)
    boost_arg = getattr(args, "formant_boost", None)
    low_hz_default = float(120.0 if low_arg is None else low_arg)
    boost_db_default = float(3.0 if boost_arg is None else boost_arg)
    window_sec = float(getattr(args, "window_size", None) or 5.0)
    step_sec = float(getattr(args, "step_size", None) or 1.5)

    step_blocks = max(1, int(step_sec * audio.SR / BLOCK))
    max_blocks = max(step_blocks + 1, int(window_sec * audio.SR / BLOCK))

    ring = collections.deque(maxlen=max_blocks)
    block_iter = src.blocks()
    stop_event = threading.Event()

    task_q: queue.Queue = queue.Queue(maxsize=3)
    emitted_words_history: list[str] = []
    warned_no_denoiser = False

    def inference_worker():
        nonlocal emitted_words_history, warned_no_denoiser
        while not stop_event.is_set():
            try:
                item = task_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return

            window_audio, t_capture = item
            dur = len(window_audio) / audio.SR

            rms = float(np.sqrt(np.mean(window_audio ** 2)))
            if rms < 5e-5:
                continue

            prof = audio.profile_audio(window_audio)
            eff_low_hz = prof.hpf_cutoff_hz if (is_dynamic and low_arg is None) else low_hz_default
            eff_boost_db = prof.formant_boost_db if (is_dynamic and boost_arg is None) else boost_db_default

            # 1. Zero-phase Butterworth HPF + Hum Notching + Formant Lifting (Rule 4)
            hp = audio.butterworth_highpass(window_audio, cutoff_hz=eff_low_hz, order=4)
            if prof.notch_freqs:
                hp = audio.harmonic_notch_filter(hp, freqs=prof.notch_freqs, bw_hz=6.0)
            if eff_boost_db > 0.0:
                if prof.snr_db <= 6.0:
                    hp = audio.whisper_formant_filter(hp, boost_db=eff_boost_db)
                else:
                    hp = audio.speech_formant_filter(hp, boost_db=eff_boost_db)

            if not audio.evaluate_window_pregate(hp, max(prof.noise_floor_rms, 1e-5)):
                continue

            # 2. Candidate enhancements + Noise-Bounded AGC (never explodes stationary hiss to 0.8)
            candidates: list[tuple[str, np.ndarray]] = []
            try:
                cfg_g = dict(cfg, robust_denoiser="gtcrn_simple.onnx")
                gtcrn_arr = audio.denoise(hp, cfg_g)
                candidates.append(("hybrid", audio.spectral_crossover_blend(gtcrn_arr, hp)))
                candidates.append(("gtcrn", gtcrn_arr))
            except Exception as e:
                if not warned_no_denoiser:
                    print(f"[stt] Warning: GTCRN denoiser unavailable: {e}", file=sys.stderr)
            try:
                cfg_d4 = dict(cfg, robust_denoiser="dpdfnet4.onnx")
                candidates.append(("dp4", audio.denoise(hp, cfg_d4, attenuation_limit_db=15.0)))
            except Exception as e:
                if not warned_no_denoiser:
                    print(f"[stt] Warning: DPDFNet4 denoiser unavailable: {e}", file=sys.stderr)
            candidates.append(("hp", hp))
            if len(candidates) == 1 and not warned_no_denoiser:
                print("[stt] Warning: No neural denoisers available; using high-pass only", file=sys.stderr)
                warned_no_denoiser = True

            chunks = [
                audio.noise_bounded_agc(arr, noise_floor_rms=max(prof.noise_floor_rms, 1e-5))
                for _, arr in candidates
            ]
            # 3. Parallel multi-stream batch decode (Rule 4)
            details = asr.transcribe_batch_detailed(rec, chunks)

            best_text = ""
            best_score = -1.0
            for (name, _), det in zip(candidates, details):
                txt = (det.get("text") or "").strip()
                if not txt:
                    continue
                if det.get("zero_skip_ratio", 0.0) > 0.60 and len(det.get("tokens", [])) >= 3:
                    continue
                sc, _ = robust.score_coherence(txt, dur)
                conf = float(det.get("confidence", 0.5))
                weighted = sc * (0.6 + 0.4 * conf)
                if weighted > best_score:
                    best_score = weighted
                    best_text = txt

            if best_text and best_score > 0.5:
                curr_words = best_text.split()
                k_max = min(len(emitted_words_history), len(curr_words))
                overlap = 0
                for k in range(k_max, 0, -1):
                    hist_slice = [w.lower().strip(".,?!;:'\"") for w in emitted_words_history[-k:]]
                    curr_slice = [w.lower().strip(".,?!;:'\"") for w in curr_words[:k]]
                    if hist_slice == curr_slice:
                        overlap = k
                        break

                new_words = curr_words[overlap:]
                if new_words:
                    clean_text = " ".join(new_words)
                    t_start = t_capture - dur
                    t_end = t_capture
                    if prof.snr_db <= 6.0 and not args.jsonl:
                        print(
                            f"[stt] Live subnoise speech detected (SNR {prof.snr_db:+.1f} dB vs mic noise floor)",
                            file=sys.stderr,
                            flush=True,
                        )
                    emit(clean_text, t_start, t_end)
                    emitted_words_history.extend(new_words)
                    if len(emitted_words_history) > 60:
                        emitted_words_history = emitted_words_history[-30:]

    worker_th = threading.Thread(target=inference_worker, daemon=True)
    worker_th.start()

    blocks_since_step = 0
    raw = []
    do_declick = getattr(args, "declick", True)
    thresh_arg = getattr(args, "declick_threshold", None)
    thresh = float(4.8 if thresh_arg is None else thresh_arg)
    declicker = audio.StreamingDeclicker(sr=audio.SR, threshold_sigma=thresh) if do_declick else None
    try:
        for b in block_iter:
            if declicker:
                b = declicker.process(b)
            if args.save:
                raw.append(b)
            ring.append(b)
            blocks_since_step += 1
            if blocks_since_step >= step_blocks and len(ring) >= max_blocks // 2:
                blocks_since_step = 0
                window_data = np.concatenate(list(ring))
                _put_bounded(task_q, (window_data, time.time()))
    finally:
        stop_event.set()
        src.stop()
        # Drain full queue non-blockingly so sentinel put never deadlocks on Ctrl+C!
        while True:
            try:
                task_q.get_nowait()
            except queue.Empty:
                break
        try:
            task_q.put_nowait(None)
        except queue.Full:
            pass
        worker_th.join(timeout=2.0)
        if args.save and raw:
            import soundfile as sf
            sf.write(args.save, np.concatenate(raw), audio.SR)
            print(f"[stt] saved raw audio to {args.save}", file=sys.stderr)


# ---------------------------------------------------------------- Termux:API

def _tmx(cfg: dict) -> str:
    p = which(cfg, "termux-microphone-record")
    if not p:
        raise RuntimeError("termux-microphone-record not found")
    return p


def record_chunk(cfg: dict, seconds: int) -> Path | None:
    """Record one chunk via Termux:API; returns the file path (caller deletes it)."""
    tmp = Path(cfg["termux_tmp"])
    tmp.mkdir(parents=True, exist_ok=True)
    f = tmp / f"chunk-{os.getpid()}-{time.monotonic_ns()}.m4a"
    cmd = _tmx(cfg)

    max_retries = 3
    base_timeout = 30
    last_error = "unknown error"
    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (2 ** attempt)
            r = subprocess.run([cmd, "-f", str(f), "-l", str(seconds), "-e", "aac", "-r", "16000", "-c", "1"],
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower():
                break
            last_error = (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired:
            last_error = f"timeout after {timeout}s"
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"termux-microphone-record failed after {max_retries} attempts: {last_error}")

    deadline = time.time() + seconds + 30
    time.sleep(0.5)
    while time.time() < deadline:
        try:
            info = subprocess.run([cmd, "-i"], capture_output=True, text=True, timeout=10).stdout
            if '"isRecording": false' in info:
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        try:
            subprocess.run([cmd, "-q"], capture_output=True, timeout=5)
        except Exception:
            pass
    return f if f.exists() else None


def capture_termux(cfg: dict, seconds: int) -> np.ndarray:
    f = record_chunk(cfg, int(seconds))
    if f is None:
        return np.zeros(0, np.float32)
    try:
        decoded = audio.decode(str(f))
        return decoded if decoded is not None else np.zeros(0, np.float32)
    finally:
        f.unlink(missing_ok=True)


def run_termux(cfg: dict, args, emit):
    _tmx_cmd = _tmx(cfg)
    subnoise = bool(getattr(args, "subnoise", False) or getattr(args, "dynamic", False) or getattr(args, "auto", False))
    level = args.aggressiveness if args.aggressiveness is not None else (3 if args.robust else int(cfg["aggressiveness"]))
    bp_arg = getattr(args, "blank_penalty", None)
    blank_penalty = float(cfg.get("blank_penalty", 0.6) if bp_arg is None else bp_arg)
    if not args.jsonl:
        print("[stt] Loading offline model...", file=sys.stderr, flush=True)
    rec = asr.get_offline(cfg, args.model, args.threads, blank_penalty=blank_penalty)
    if not args.jsonl:
        print("\r\033[K[stt] Model loaded", file=sys.stderr, flush=True)
    q: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()

    def worker():
        while True:
            item = q.get()
            if item is None:
                return
            samples, t0 = item
            if subnoise:
                for s, e, text in robust.subnoise_transcribe(rec, samples, cfg):
                    emit(text, t0 + s, t0 + e)
            else:
                for s, e, text in robust.transcribe(rec, samples, cfg, level):
                    emit(text, t0 + s, t0 + e)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    n = int(cfg["mic_chunk_seconds"])
    try:
        while not stop.is_set():
            t0 = time.time()
            samples = capture_termux(cfg, n)
            if len(samples) and float(np.sqrt(np.mean(samples ** 2))) > 1e-4:
                _put_bounded(q, (samples, t0))
    finally:
        stop.set()
        try:
            subprocess.run([_tmx_cmd, "-q"], capture_output=True, timeout=5)
        except Exception:
            pass
        _put_bounded(q, None)
        th.join(timeout=60)
        for f in Path(cfg["termux_tmp"]).glob(f"chunk-{os.getpid()}-*"):
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------- Windows (DirectShow / sounddevice)

def list_dshow_audio_devices() -> list[str]:
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg:
        return []
    try:
        p = subprocess.run(
            [ffmpeg, "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=5,
        )
        lines = (p.stderr or "").splitlines()
        audio_section = False
        devices = []
        for line in lines:
            if "DirectShow audio devices" in line:
                audio_section = True
                continue
            if "DirectShow video devices" in line:
                audio_section = False
            if audio_section and '"' in line:
                dev = line.split('"')[1]
                if dev and dev not in devices and not dev.startswith("@device"):
                    devices.append(dev)
        return devices
    except Exception:
        return []


class WindowsDShowSource:
    """Windows audio capture using ffmpeg with DirectShow."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rec: subprocess.Popen | None = None
        self._stderr_lines: collections.deque[str] = collections.deque(maxlen=32)
        self.backlog = lambda: 0
        atexit.register(self.stop)
        if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
            signal.signal(signal.SIGTERM, lambda s, f: (self.stop(), sys.exit(1)))
        if signal.getsignal(signal.SIGINT) is signal.SIG_DFL:
            signal.signal(signal.SIGINT, lambda s, f: (self.stop(), sys.exit(1)))

    def start(self):
        ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not ffmpeg:
            raise RuntimeError("ffmpeg binary not found on PATH")
        device = self.cfg.get("windows_mic_device", "default")
        if device == "default":
            dev_names = list_dshow_audio_devices()
            device_arg = f"audio={dev_names[0]}" if dev_names else "audio=default"
        else:
            device_arg = f"audio={device}"

        cmd = [
            ffmpeg, "-v", "error", "-nostdin",
            "-f", "dshow", "-i", device_arg,
            "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "-",
        ]
        try:
            self.rec = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as e:
            raise RuntimeError(f"failed to start ffmpeg dshow: {e}")

        # Drain stderr continuously in a daemon thread so pipe buffer never fills & deadlocks stdout (Rule 3)
        stderr_pipe = self.rec.stderr
        if stderr_pipe is not None:
            def _drain_stderr():
                try:
                    for raw_line in iter(stderr_pipe.readline, b""):
                        line = raw_line.decode(errors="replace").strip()
                        if line:
                            self._stderr_lines.append(line)
                except Exception:
                    pass

            threading.Thread(target=_drain_stderr, daemon=True).start()

        time.sleep(0.3)
        if self.rec.poll() is not None:
            err = "; ".join(self._stderr_lines)
            raise RuntimeError(f"ffmpeg dshow capture exited: {err}")

    def blocks(self):
        q: queue.Queue = queue.Queue(maxsize=1024)
        stdout = self.rec.stdout

        def reader():
            nbytes = BLOCK * 2
            buf = b""
            while True:
                try:
                    d = stdout.read(nbytes - len(buf))
                    if not d:
                        _put_bounded(q, None)
                        return
                    buf += d
                    if len(buf) == nbytes:
                        _put_bounded(q, np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0)
                        buf = b""
                except Exception:
                    _put_bounded(q, None)
                    return

        threading.Thread(target=reader, daemon=True).start()
        self.backlog = q.qsize
        while True:
            b = q.get()
            if b is None:
                return
            yield b

    def stop(self):
        _kill(self.rec)
        self.rec = None


class SoundDeviceSource:
    """Cross-platform / Windows capture using sounddevice (PortAudio/WASAPI)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.stream = None
        self.q: queue.Queue = queue.Queue(maxsize=1024)
        self.backlog = self.q.qsize
        atexit.register(self.stop)

    def start(self):
        import sounddevice as sd
        device = None if self.cfg.get("windows_mic_device", "default") == "default" else self.cfg.get("windows_mic_device")

        def callback(indata, frames, time_info, status):
            _put_bounded(self.q, indata.flatten().copy())

        self.stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            blocksize=BLOCK,
            dtype="float32",
            device=device,
            callback=callback,
        )
        self.stream.start()

    def blocks(self):
        while True:
            b = self.q.get()
            if b is None:
                return
            yield b

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        _put_bounded(self.q, None)


def get_windows_source(cfg: dict):
    backend = cfg.get("windows_backend", "auto")
    if backend == "sounddevice":
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            in_devs = [d for d in devs if d.get("max_input_channels", 0) > 0]
            if not in_devs:
                raise RuntimeError("No input devices found for sounddevice")
        except Exception as e:
            raise RuntimeError(f"sounddevice unavailable: {e}")
        return SoundDeviceSource(cfg)
    elif backend == "dshow":
        return WindowsDShowSource(cfg)
    else:
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            in_devs = [d for d in devs if d.get("max_input_channels", 0) > 0]
            if in_devs:
                return SoundDeviceSource(cfg)
        except Exception:
            pass
        return WindowsDShowSource(cfg)


# ---------------------------------------------------------------- detection

def detect(cfg: dict) -> dict:
    """Return {backend: (ok, detail)} without recording."""
    res = {}
    pa, parec = which(cfg, "pulseaudio"), which(cfg, "parec")
    if _is_termux_env(cfg):
        res["pulse"] = (bool(pa and parec), f"pulseaudio={pa} parec={parec}")
    else:
        # On native Linux (PulseAudio / PipeWire), `parec` works directly without spawning Termux SLES server
        pulse_ok = bool(parec and sys.platform != "win32")
        res["pulse"] = (pulse_ok, f"parec={parec} (system PulseAudio/PipeWire)" if pulse_ok else f"pulseaudio={pa} parec={parec}")

    cmd = which(cfg, "termux-microphone-record")
    ok, detail = False, "termux-microphone-record not found"
    if cmd:
        try:
            r = subprocess.run([cmd, "-i"], capture_output=True, text=True, timeout=30)
            ok = '"isRecording"' in r.stdout
            detail = cmd if ok else (r.stdout + r.stderr).strip()[:100]
        except Exception as e:
            detail = str(e)
    res["termux"] = (ok, detail)

    has_sd = False
    sd_detail = ""
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        in_devs = [d for d in devs if d.get("max_input_channels", 0) > 0]
        if in_devs:
            has_sd = True
            sd_detail = f"sounddevice ({in_devs[0].get('name')})"
    except Exception:
        pass

    dshow_devs = list_dshow_audio_devices()
    dshow_ok = bool((shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")) and (dshow_devs or sys.platform == "win32"))

    if has_sd:
        res["windows"] = (True, sd_detail)
    elif dshow_ok:
        dev_str = dshow_devs[0] if dshow_devs else "DirectShow default"
        res["windows"] = (True, f"ffmpeg dshow: {dev_str}")
    else:
        res["windows"] = (False, "ffmpeg dshow or sounddevice not found")

    return res
