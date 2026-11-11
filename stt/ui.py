"""Web front-end for talk-to-typed / stt.

Features:
- File picker, folder picker, and drag-and-drop input.
- Real-time streaming transcription text box.
- Dynamic auto-tuning, sub-noise formant scanning, and denoising controls.
- Complete keyboard hotkeys for hands-free navigation.
- Zero external dependencies (pure Python standard library http.server).
"""
import http.server
import json
import os
import shutil
import socketserver
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

from . import __version__, asr, audio, config, robust


def _parse_multipart(content_type: str, raw_body: bytes) -> list[dict]:
    """Parse multipart/form-data body into [{name, filename, data}].

    Uses raw boundary splitting so binary audio payloads are never decoded as text.
    """
    boundary = None
    for param in content_type.split(";"):
        param = param.strip()
        if param.lower().startswith("boundary="):
            boundary = param[9:].strip('"\'')
            break
    if not boundary:
        return []

    sep = ("--" + boundary).encode("latin-1")
    parts = []
    for raw_part in raw_body.split(sep):
        stripped = raw_part.strip(b"\r\n")
        if not stripped or stripped == b"--":
            continue
        if b"\r\n\r\n" in stripped:
            headers_raw, body = stripped.split(b"\r\n\r\n", 1)
        elif b"\n\n" in stripped:
            headers_raw, body = stripped.split(b"\n\n", 1)
        else:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]
        name = filename = None
        for hline in headers_raw.decode("latin-1", errors="replace").splitlines():
            if ":" not in hline:
                continue
            hname, hval = hline.split(":", 1)
            if hname.strip().lower() == "content-disposition":
                for p in hval.split(";"):
                    p = p.strip()
                    if p.lower().startswith("name="):
                        name = p[5:].strip("\"'")
                    elif p.lower().startswith("filename="):
                        filename = p[9:].strip("\"'")
        if name:
            parts.append({"name": name, "filename": filename, "data": body})
    return parts

# Thread-safe abort flag shared across concurrent request threads
_transcription_abort = threading.Event()

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>talk-to-typed — Offline Speech-to-Text</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24' fill='none' stroke='%233b82f6' stroke-width='2'><path d='M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z'/><path d='M19 10v2a7 7 0 0 1-14 0v-2'/><line x1='12' x2='12' y1='19' y2='22'/></svg>">
<style>
  :root {
    --bg-dark: #0f1117;
    --bg-card: #181c24;
    --bg-card-hover: #202632;
    --border: #28303f;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.25);
    --accent-hover: #2563eb;
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --code-bg: #090b10;
  }

  body.light-theme {
    --bg-dark: #f8fafc;
    --bg-card: #ffffff;
    --bg-card-hover: #f1f5f9;
    --border: #e2e8f0;
    --accent: #2563eb;
    --accent-glow: rgba(37, 99, 235, 0.12);
    --accent-hover: #1d4ed8;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --success: #059669;
    --warning: #d97706;
    --danger: #dc2626;
    --code-bg: #f1f5f9;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-main);
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }

  header {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-main);
  }
  .badge {
    background: var(--accent-glow);
    color: var(--accent);
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid var(--accent);
  }
  .header-actions {
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .container {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* Sidebar Controls */
  aside {
    width: 360px;
    background: var(--bg-card);
    border-right: 1px solid var(--border);
    padding: 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  .section-title {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
  }
  .control-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  label {
    font-size: 0.85rem;
    color: var(--text-main);
    display: flex;
    justify-content: space-between;
    font-weight: 500;
  }
  .label-desc {
    font-size: 0.75rem;
    color: var(--text-muted);
  }
  select, input[type="text"], input[type="number"] {
    background: var(--code-bg);
    border: 1px solid var(--border);
    color: var(--text-main);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.9rem;
    width: 100%;
  }
  select:focus, input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-glow);
  }
  .slider-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  input[type="range"] {
    flex: 1;
    accent-color: var(--accent);
  }
  .val-badge {
    font-family: monospace;
    font-size: 0.85rem;
    background: var(--border);
    padding: 2px 6px;
    border-radius: 4px;
    min-width: 48px;
    text-align: center;
  }
  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    cursor: pointer;
  }

  /* Window overlap canvas */
  .win-canvas-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  #windowCanvas {
    width: 100%;
    border-radius: 6px;
    display: block;
    background: var(--code-bg);
    border: 1px solid var(--border);
  }
  .win-canvas-info {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: monospace;
    text-align: center;
  }

  /* Transcription error message */
  .error-banner {
    background: rgba(239,68,68,0.12);
    border: 1px solid var(--danger);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: var(--danger);
    display: none;
  }

  /* Main Workspace */
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 24px;
    gap: 20px;
    overflow-y: auto;
  }

  /* Dropzone */
  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    background: var(--bg-card);
    transition: all 0.2s ease;
    cursor: pointer;
    position: relative;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--accent);
    background: var(--bg-card-hover);
    box-shadow: 0 0 20px var(--accent-glow);
  }
  .dropzone p {
    color: var(--text-muted);
    margin-top: 8px;
    font-size: 0.9rem;
  }
  .file-buttons {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 14px;
  }

  /* Primary Button */
  .btn {
    background: var(--accent);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: background 0.2s;
  }
  .btn:hover { background: var(--accent-hover); }
  .btn-outline {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-main);
  }
  .btn-outline:hover {
    background: var(--border);
  }
  .btn-danger {
    background: var(--danger);
  }
  .btn-sm {
    padding: 4px 10px;
    font-size: 0.78rem;
  }

  /* Server Path Row */
  .path-bar {
    display: flex;
    gap: 10px;
    margin-top: 12px;
  }

  /* Reasoning Box */
  .reasoning-box {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.85rem;
    line-height: 1.5;
    display: none;
  }
  .reasoning-box h4 {
    color: var(--accent);
    margin-bottom: 6px;
    font-size: 0.88rem;
  }
  .reasoning-box ul {
    list-style-type: none;
    padding-left: 0;
  }
  .reasoning-box li {
    margin-bottom: 4px;
    color: var(--text-muted);
  }
  .reasoning-box li strong { color: var(--text-main); }

  /* Progress Bar */
  .progress-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    display: none;
  }
  .progress-bar-bg {
    background: var(--code-bg);
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
    margin: 8px 0;
  }
  .progress-bar-fill {
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    height: 100%;
    width: 0%;
    transition: width 0.15s ease;
  }
  .progress-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-muted);
    font-family: monospace;
  }

  /* Output Area */
  .output-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .output-header {
    padding: 12px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255,255,255,0.02);
  }
  .output-header h3 {
    font-size: 0.95rem;
    font-weight: 600;
  }
  .output-actions {
    display: flex;
    gap: 8px;
  }
  textarea#output {
    flex: 1;
    background: var(--code-bg);
    border: none;
    color: var(--text-main);
    padding: 18px;
    font-size: 0.95rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    line-height: 1.6;
    resize: none;
    min-height: 250px;
  }
  textarea#output:focus { outline: none; }
  .output-footer {
    padding: 10px 18px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-muted);
    background: rgba(255,255,255,0.02);
  }

  /* Audio Sync & Interactive Cues */
  .audio-wrap {
    padding: 10px 18px;
    background: var(--code-bg);
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .audio-wrap-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-muted);
  }
  .audio-player {
    width: 100%;
    height: 36px;
    outline: none;
  }
  .view-switcher {
    display: inline-flex;
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px;
    margin-right: 8px;
  }
  .view-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    padding: 3px 10px;
    font-size: 0.78rem;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 500;
  }
  .view-btn.active {
    background: var(--accent);
    color: white;
  }
  .segments-list {
    flex: 1;
    overflow-y: auto;
    padding: 14px 18px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-height: 250px;
    background: var(--code-bg);
  }
  .segment-item {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    background: rgba(255,255,255,0.015);
    border-left: 3px solid transparent;
    transition: all 0.15s ease;
  }
  .segment-item:hover {
    background: var(--bg-card-hover);
    border-left-color: var(--accent-hover);
  }
  .segment-item.active {
    background: var(--accent-glow);
    border-left-color: var(--accent);
  }
  .seg-time {
    font-family: monospace;
    font-size: 0.78rem;
    color: var(--accent);
    flex-shrink: 0;
    user-select: none;
  }
  .seg-text {
    color: var(--text-main);
    line-height: 1.5;
    font-size: 0.93rem;
  }

  /* Modal */
  .modal-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7);
    display: none;
    justify-content: center;
    align-items: center;
    z-index: 1000;
  }
  .modal {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    width: 500px;
    max-width: 90vw;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
  }
  .modal h3 {
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .hotkey-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  .hotkey-table td {
    padding: 8px 4px;
    border-bottom: 1px solid var(--border);
  }
  kbd {
    background: var(--code-bg);
    border: 1px solid var(--border);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.8rem;
    color: var(--accent);
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--accent);"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
    talk-to-typed
    <span class="badge">v%STT_VERSION%</span>
  </div>
  <div class="header-actions">
    <button class="btn btn-outline btn-sm" id="btnTheme" title="Toggle Light/Dark Theme">◑ Theme</button>
    <button class="btn btn-outline btn-sm" id="btnHistory" title="Session History">⏱ History</button>
    <button class="btn btn-outline btn-sm" id="btnHotkeys" title="Keyboard Shortcuts (?)">
      <kbd>?</kbd> Hotkeys
    </button>
  </div>
</header>

<div class="container">
  <!-- Left Settings Sidebar -->
  <aside>
    <div class="section-title">Acoustic Strategy</div>
    <div class="control-group">
      <label for="strategy">Mode <span class="label-desc">(Alt+A / Alt+N)</span></label>
      <select id="strategy">
        <optgroup label="Adaptive Recovery">
          <option value="auto" selected>Auto — adaptive sub-noise profiling (recommended)</option>
          <option value="dynamic">Dynamic CN-ROVER — 4-model ensemble, best for faint speech</option>
          <option value="subnoise">Force Sub-Noise — aggressive low-energy scan (slowest)</option>
        </optgroup>
        <optgroup label="Fixed Denoising">
          <option value="standard">Standard VAD — clean audio, fastest</option>
          <option value="robust">Robust — moderate noise, Level 3 denoiser</option>
          <option value="distant">Distant — heavy noise, Level 5 denoiser</option>
        </optgroup>
        <optgroup label="Research">
          <option value="all">Matrix All-Models — compare all combinations (slow)</option>
        </optgroup>
      </select>
    </div>

    <div class="control-group">
      <label for="modelSelect">ASR Model</label>
      <select id="modelSelect">
        <option value="">Default (from config)</option>
      </select>
      <span class="label-desc">Installed models auto-loaded.</span>
    </div>

    <div class="section-title">Language & Translation</div>
    <div class="control-group">
      <label for="langSource">Source Language</label>
      <select id="langSource">
        <option value="auto" selected>Auto-detect</option>
        <option value="en">English</option>
        <option value="es">Spanish</option>
        <option value="pt">Portuguese</option>
        <option value="fr">French</option>
        <option value="de">German</option>
        <option value="it">Italian</option>
        <option value="nl">Dutch</option>
        <option value="pl">Polish</option>
        <option value="ru">Russian</option>
        <option value="zh">Chinese</option>
        <option value="ja">Japanese</option>
        <option value="ko">Korean</option>
        <option value="ar">Arabic</option>
        <option value="hi">Hindi</option>
      </select>
    </div>
    <div class="control-group">
      <label for="langTarget">Target Language <span class="label-desc">(for translation)</span></label>
      <select id="langTarget">
        <option value="en" selected>English</option>
        <option value="es">Spanish</option>
        <option value="pt">Portuguese</option>
        <option value="fr">French</option>
        <option value="de">German</option>
        <option value="it">Italian</option>
        <option value="nl">Dutch</option>
        <option value="pl">Polish</option>
        <option value="ru">Russian</option>
        <option value="zh">Chinese</option>
        <option value="ja">Japanese</option>
        <option value="ko">Korean</option>
        <option value="ar">Arabic</option>
        <option value="hi">Hindi</option>
      </select>
    </div>
    <div class="control-group">
      <label class="checkbox-row">
        <input type="checkbox" id="enableTranslation">
        <span>Enable Translation</span>
      </label>
    </div>
    <div class="control-group" id="langDetectInfo" style="display: none;">
      <label>Detected Language</label>
      <div id="detectedLangBadge" style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; font-size: 0.85rem;">
        <span class="label-desc" id="detectedLangText">Detecting...</span>
        <span class="val-badge" id="detectedLangConf"></span>
      </div>
    </div>

    <div class="section-title">Acoustic Equalization</div>
    <div class="control-group">
      <label for="rumbleCutoff">Low-Cut Rumble Filter <span class="val-badge" id="valRumble">120 Hz</span></label>
      <div class="slider-row">
        <input type="range" id="rumbleCutoff" min="40" max="250" step="10" value="120">
      </div>
      <span class="label-desc">Strips HVAC, wind, and mic handling rumble (<120Hz).</span>
    </div>

    <div class="control-group">
      <label for="formantBoost">Formant Boost <span class="val-badge" id="valBoost">+3.0 dB</span></label>
      <div class="slider-row">
        <input type="range" id="formantBoost" min="0.0" max="8.0" step="0.5" value="3.0">
      </div>
      <span class="label-desc">Lifts 300–3200 Hz speech harmonics above room noise.</span>
    </div>

    <div class="control-group">
      <label for="blankPenalty">Blank Penalty <span class="val-badge" id="valBlank">0.6</span></label>
      <div class="slider-row">
        <input type="range" id="blankPenalty" min="0.0" max="1.5" step="0.1" value="0.6">
      </div>
      <span class="label-desc">Forces phoneme emission on faint speech without token loops.</span>
    </div>

    <div class="section-title">Time Range <span class="label-desc">(optional)</span></div>
    <div class="control-group">
      <label for="rangeStart">Start (seconds)</label>
      <input type="number" id="rangeStart" value="" step="0.1" min="0" placeholder="0 = beginning">
    </div>
    <div class="control-group">
      <label for="rangeEnd">End (seconds)</label>
      <input type="number" id="rangeEnd" value="" step="0.1" min="0" placeholder="blank = end">
    </div>

    <div class="section-title">Window & Performance</div>
    <div class="control-group">
      <label for="windowSize">Window Length (s)</label>
      <input type="number" id="windowSize" value="5.0" step="0.5" min="2.0" max="30.0">
    </div>
    <div class="control-group">
      <label for="stepSize">Stride Step (s)</label>
      <input type="number" id="stepSize" value="1.5" step="0.25" min="0.5" max="15.0">
    </div>
    <div class="control-group">
      <label for="threads">CPU Threads</label>
      <input type="number" id="threads" value="4" min="1" max="16">
    </div>
    <div class="control-group">
      <label class="checkbox-row">
        <input type="checkbox" id="recursive" checked>
        <span>Recursive directory search (-r)</span>
      </label>
    </div>

    <div class="section-title">Buffer Overlap Preview</div>
    <div class="win-canvas-wrap">
      <canvas id="windowCanvas" width="310" height="90"></canvas>
      <div class="win-canvas-info" id="winInfo">window: 5.0s | step: 1.5s | overlap: 70%</div>
    </div>
  </aside>

  <!-- Main Work Area -->
  <main>
    <!-- Dropzone / File Picker -->
    <div class="dropzone" id="dropzone">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="color:var(--text-muted);"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
      <h3 style="margin-top:10px;">Drag & Drop Audio / Video Files Here</h3>
      <p>Supports .wav, .mp3, .m4a, .flac, .ogg, .opus, .aac, .mp4, .mkv, and more</p>
      
      <div class="file-buttons">
        <button class="btn btn-outline" id="btnBrowseFile">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Choose File <kbd>Ctrl+O</kbd>
        </button>
        <button class="btn btn-outline" id="btnBrowseFolder">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
          Choose Folder <kbd>Ctrl+D</kbd>
        </button>
        <button class="btn btn-outline" id="btnStartTranscribe" style="background:var(--accent); color:white; border:none;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Transcribe <kbd>Ctrl+Enter</kbd>
        </button>
      </div>

      <input type="file" id="fileInput" style="display:none;" accept="audio/*,video/*">
      <input type="file" id="folderInput" style="display:none;" webkitdirectory directory multiple>

      <div class="path-bar">
        <input type="text" id="serverPath" placeholder="Or enter local file/directory path on disk (e.g. /home/o/audio.wav or C:\recordings)...">
        <button class="btn btn-outline btn-sm" id="btnRunPath">Load Path</button>
      </div>
    </div>

    <!-- Error Banner -->
    <div class="error-banner" id="errorBanner"></div>

    <!-- Deterministic Reasoning Display -->
    <div class="reasoning-box" id="reasoningBox">
      <h4>Deterministic Configuration Reasoning:</h4>
      <ul id="reasoningList"></ul>
    </div>

    <!-- Live Progress Bar -->
    <div class="progress-wrap" id="progressWrap">
      <div class="progress-meta">
        <span id="progFile">Processing...</span>
        <span id="progPct">0%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" id="progFill"></div>
      </div>
      <div class="progress-meta">
        <span id="progTime">00:00 / 00:00</span>
        <span id="progSpeed">Speed: 0.00x | ETA: 00:00</span>
      </div>
    </div>

    <!-- SNR / Auto-Profile Badge -->
    <div id="snrBadge" style="display:none; padding:6px 12px; background:var(--code-bg); border:1px solid var(--border); border-radius:6px; font-size:0.82rem; color:var(--text-muted); margin-bottom:8px;">
      <strong style="color:var(--accent);">Profile:</strong> <span id="snrBadgeText"></span>
    </div>

    <!-- Output Textbox Card -->
    <div class="output-card">
      <div class="output-header">
        <div style="display:flex; align-items:center;">
          <h3 id="outputHeading" style="margin-right:12px;">Transcript</h3>
          <div class="view-switcher">
            <button class="view-btn active" id="btnViewText">Text</button>
            <button class="view-btn" id="btnViewSegments">Interactive Cues</button>
          </div>
        </div>
        <div class="output-actions">
          <button class="btn btn-outline btn-sm" id="btnCopy" title="Copy to Clipboard (Alt+C)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy
          </button>
          <button class="btn btn-outline btn-sm" id="btnExportTxt" title="Export as TXT (Ctrl+S)">
            TXT
          </button>
          <button class="btn btn-outline btn-sm" id="btnExportJsonl" title="Export JSONL">
            JSONL
          </button>
          <button class="btn btn-outline btn-sm" id="btnExportSrt" title="Export Subtitle SRT">
            SRT
          </button>
          <button class="btn btn-outline btn-sm" id="btnExportVtt" title="Export Subtitle VTT">
            VTT
          </button>
          <button class="btn btn-outline btn-sm btn-danger" id="btnClear" title="Clear Text (Alt+X)">
            Clear
          </button>
          <button class="btn btn-outline btn-sm btn-danger" id="btnCancel" title="Cancel Transcription (Alt+Z)">
            Cancel
          </button>
        </div>
      </div>

      <!-- Audio Player Bar (Audio Sync) -->
      <div class="audio-wrap" id="audioWrap" style="display:none;">
        <div class="audio-wrap-header">
          <span id="audioTrackName">Audio Player</span>
          <span id="audioCurrentTime">00:00 / 00:00</span>
        </div>
        <audio id="audioPlayer" class="audio-player" controls preload="metadata"></audio>
      </div>

      <textarea id="output" placeholder="Transcription text will stream here in real time..."></textarea>
      <div id="segmentsList" class="segments-list" style="display:none;"></div>
      <div class="output-footer">
        <span id="wordStats">0 words | 0 characters</span>
        <span id="procStats">Ready</span>
      </div>
    </div>
  </main>
</div>

<!-- Hotkey Guide Modal -->
<div class="modal-overlay" id="modalOverlay">
  <div class="modal">
    <h3>
      Keyboard Shortcuts
      <button class="btn btn-outline btn-sm" id="btnCloseModal">✕</button>
    </h3>
    <table class="hotkey-table">
      <tr><td><kbd>Space</kbd></td><td>Play / Pause Audio</td></tr>
      <tr><td><kbd>Alt+←</kbd> / <kbd>Alt+→</kbd></td><td>Seek Audio -5s / +5s</td></tr>
      <tr><td><kbd>Ctrl+O</kbd> / <kbd>Alt+O</kbd></td><td>Open File Picker</td></tr>
      <tr><td><kbd>Ctrl+D</kbd> / <kbd>Alt+D</kbd></td><td>Open Folder / Directory Picker</td></tr>
      <tr><td><kbd>Ctrl+Enter</kbd></td><td>Start Transcription</td></tr>
      <tr><td><kbd>Ctrl+S</kbd></td><td>Download Transcript (.txt)</td></tr>
      <tr><td><kbd>Alt+C</kbd></td><td>Copy Transcript to Clipboard</td></tr>
      <tr><td><kbd>Alt+A</kbd></td><td>Switch to Auto Mode</td></tr>
      <tr><td><kbd>Alt+N</kbd></td><td>Switch to Sub-Noise Mode</td></tr>
      <tr><td><kbd>Alt+X</kbd></td><td>Clear Output Text</td></tr>
      <tr><td><kbd>Alt+Z</kbd></td><td>Cancel Transcription</td></tr>
      <tr><td><kbd>?</kbd> / <kbd>F1</kbd></td><td>Toggle this Hotkey Guide</td></tr>
      <tr><td><kbd>Esc</kbd></td><td>Close Modal</td></tr>
    </table>
  </div>
</div>

<!-- History Modal -->
<div class="modal-overlay" id="historyOverlay" style="display:none;">
  <div class="modal" style="max-width:600px; width:90vw;">
    <h3>
      Session History
      <button class="btn btn-outline btn-sm" id="btnCloseHistory">✕</button>
    </h3>
    <div id="historyList" style="max-height:60vh; overflow-y:auto; display:flex; flex-direction:column; gap:8px; margin-top:12px;">
      <p style="color:var(--text-muted); font-size:0.85rem;">No previous transcriptions this session.</p>
    </div>
  </div>
</div>

<script>
  // Theme persistence
  if (localStorage.getItem('stt-theme') === 'light') document.body.classList.add('light-theme');
  document.getElementById('btnTheme').onclick = () => {
    document.body.classList.toggle('light-theme');
    localStorage.setItem('stt-theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
  };

  // Load installed models
  (async function loadModels() {
    try {
      const r = await fetch('/api/models');
      if (!r.ok) return;
      const models = await r.json();
      const sel = document.getElementById('modelSelect');
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.name} (${m.kind || 'asr'}, ${(m.size_mb || 0).toFixed(0)} MB)${m.active ? ' ✓' : ''}`;
        sel.appendChild(opt);
      }
    } catch {}
  })();

  let selectedFiles = [];
  let segmentsHistory = [];
  let transcriptHistory = [];

  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const folderInput = document.getElementById('folderInput');
  const serverPath = document.getElementById('serverPath');
  const output = document.getElementById('output');
  const segmentsList = document.getElementById('segmentsList');
  const btnViewText = document.getElementById('btnViewText');
  const btnViewSegments = document.getElementById('btnViewSegments');
  const audioWrap = document.getElementById('audioWrap');
  const audioPlayer = document.getElementById('audioPlayer');
  const audioTrackName = document.getElementById('audioTrackName');
  const audioCurrentTime = document.getElementById('audioCurrentTime');
  const wordStats = document.getElementById('wordStats');
  const procStats = document.getElementById('procStats');
  const reasoningBox = document.getElementById('reasoningBox');
  const reasoningList = document.getElementById('reasoningList');
  const errorBanner = document.getElementById('errorBanner');
  const progressWrap = document.getElementById('progressWrap');
  const progFill = document.getElementById('progFill');
  const progPct = document.getElementById('progPct');
  const progFile = document.getElementById('progFile');
  const progTime = document.getElementById('progTime');
  const progSpeed = document.getElementById('progSpeed');

  function formatSec(s) {
    if (isNaN(s) || s < 0) return "00:00.000";
    const totalMs = Math.round(s * 1000);
    const ms = totalMs % 1000;
    const totalSec = Math.floor(totalMs / 1000);
    const sec = totalSec % 60;
    const totalMin = Math.floor(totalSec / 60);
    if (totalMin >= 60) {
      const h = Math.floor(totalMin / 60);
      const m = totalMin % 60;
      return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
    }
    return `${totalMin.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // View Switcher
  btnViewText.onclick = () => {
    btnViewText.classList.add('active');
    btnViewSegments.classList.remove('active');
    output.style.display = 'block';
    segmentsList.style.display = 'none';
  };
  btnViewSegments.onclick = () => {
    btnViewSegments.classList.add('active');
    btnViewText.classList.remove('active');
    output.style.display = 'none';
    segmentsList.style.display = 'flex';
  };

  // Audio Player Sync
  audioPlayer.ontimeupdate = () => {
    const cur = audioPlayer.currentTime;
    const dur = audioPlayer.duration || 0;
    audioCurrentTime.textContent = `${formatSec(cur)} / ${formatSec(dur)}`;
    const items = segmentsList.querySelectorAll('.segment-item');
    items.forEach(el => {
      const s = parseFloat(el.dataset.start);
      const e = parseFloat(el.dataset.end);
      if (cur >= s && cur <= e) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    });
  };

  // Sliders & manual override tracking (so 'auto' strategy keeps auto-tuned DSP unless user adjusts a slider)
  const touchedControls = new Set();
  const rumbleCutoff = document.getElementById('rumbleCutoff');
  const valRumble = document.getElementById('valRumble');
  rumbleCutoff.addEventListener('input', () => {
    touchedControls.add('rumble_cutoff');
    valRumble.textContent = `${rumbleCutoff.value} Hz`;
  });

  const formantBoost = document.getElementById('formantBoost');
  const valBoost = document.getElementById('valBoost');
  formantBoost.addEventListener('input', () => {
    touchedControls.add('formant_boost');
    valBoost.textContent = `+${parseFloat(formantBoost.value).toFixed(1)} dB`;
  });

  const blankPenalty = document.getElementById('blankPenalty');
  const valBlank = document.getElementById('valBlank');
  blankPenalty.addEventListener('input', () => {
    touchedControls.add('blank_penalty');
    valBlank.textContent = parseFloat(blankPenalty.value).toFixed(1);
  });

  // ── Window Overlap Visualization ──────────────────────────────────────────
  function drawWindowViz() {
    const canvas = document.getElementById('windowCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const winSec  = parseFloat(document.getElementById('windowSize').value) || 5.0;
    const stepSec = parseFloat(document.getElementById('stepSize').value)   || 1.5;
    const overlapSec = Math.max(0, winSec - stepSec);
    const overlapPct = winSec > 0 ? Math.round(overlapSec / winSec * 100) : 0;

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#090b10';
    ctx.fillRect(0, 0, W, H);

    // Layout constants
    const PAD = 4, TOPPAD = 12, WBOX_H = 32, WIN_STACK = 6;
    // Show ~3 full windows: total display duration
    const showDur = stepSec * 2 + winSec;
    const pps = (W - 2 * PAD) / showDur;  // pixels per second
    const winPx = winSec * pps;

    const colors = [
      'rgba(59,130,246,0.50)',
      'rgba(99,160,250,0.42)',
      'rgba(59,130,246,0.35)',
    ];

    // Draw each window box (up to 3)
    const nWins = Math.min(3, Math.ceil(showDur / stepSec));
    for (let i = 0; i < nWins; i++) {
      const x = PAD + i * stepSec * pps;
      const y = TOPPAD + i * WIN_STACK;
      ctx.fillStyle = colors[i % colors.length];
      ctx.fillRect(x, y, winPx, WBOX_H);
      ctx.strokeStyle = 'rgba(96,165,250,0.75)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, winPx - 1, WBOX_H - 1);
    }

    // Overlap highlight (between window 0 and window 1)
    if (nWins >= 2 && overlapSec > 0) {
      const overlapStart = PAD + stepSec * pps;
      const overlapW = overlapSec * pps;
      ctx.fillStyle = 'rgba(245,158,11,0.18)';
      ctx.fillRect(overlapStart, TOPPAD, overlapW, WBOX_H);
      ctx.strokeStyle = 'rgba(245,158,11,0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.strokeRect(overlapStart + 0.5, TOPPAD + 0.5, overlapW - 1, WBOX_H - 1);
      ctx.setLineDash([]);
    }

    // Step-size brace (bottom)
    const arrowY = TOPPAD + WBOX_H + WIN_STACK * (nWins - 1) + 10;
    if (nWins >= 2) {
      const x0 = PAD, x1 = PAD + stepSec * pps;
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 2]);
      ctx.beginPath(); ctx.moveTo(x0, arrowY); ctx.lineTo(x1, arrowY); ctx.stroke();
      ctx.setLineDash([]);
      // arrowheads
      ctx.fillStyle = '#f59e0b';
      const ah = 3;
      ctx.beginPath(); ctx.moveTo(x0,    arrowY); ctx.lineTo(x0 + ah, arrowY - ah); ctx.lineTo(x0 + ah, arrowY + ah); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.moveTo(x1,    arrowY); ctx.lineTo(x1 - ah, arrowY - ah); ctx.lineTo(x1 - ah, arrowY + ah); ctx.closePath(); ctx.fill();
      // step label
      ctx.fillStyle = '#f59e0b';
      ctx.font = 'bold 9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('step', (x0 + x1) / 2, arrowY - 4);
      ctx.textAlign = 'left';
    }

    // Window-size brace (top)
    const bracY = TOPPAD - 3;
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(PAD, bracY); ctx.lineTo(PAD + winPx, bracY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    const winLabel = `win`;
    ctx.fillText(winLabel, PAD + winPx / 2, bracY - 2);
    ctx.textAlign = 'left';

    // Timeline bar
    ctx.fillStyle = '#28303f';
    ctx.fillRect(PAD, H - 10, W - 2 * PAD, 4);
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(PAD, H - 10, Math.min(winPx * nWins * 0.7, W - 2 * PAD), 4);

    // Update text info
    document.getElementById('winInfo').textContent =
      `window ${winSec.toFixed(1)}s  |  step ${stepSec.toFixed(1)}s  |  overlap ${overlapPct}%`;
  }

  document.getElementById('windowSize').addEventListener('input', () => {
    touchedControls.add('window_size');
    drawWindowViz();
  });
  document.getElementById('stepSize').addEventListener('input', () => {
    touchedControls.add('step_size');
    drawWindowViz();
  });
  drawWindowViz();  // initial draw

  // File pickers
  document.getElementById('btnBrowseFile').onclick = () => fileInput.click();
  document.getElementById('btnBrowseFolder').onclick = () => folderInput.click();

  fileInput.onchange = (e) => handleFiles(Array.from(e.target.files));
  folderInput.onchange = (e) => handleFiles(Array.from(e.target.files));

  function handleFiles(files) {
    if (!files.length) return;
    selectedFiles = files;
    serverPath.value = "";
    procStats.textContent = `${files.length} file(s) selected: ${files.map(f => f.name).slice(0, 3).join(', ')}${files.length > 3 ? '...' : ''}`;
    
    // Mount first file to audio player for instant playback
    audioPlayer.src = URL.createObjectURL(files[0]);
    audioTrackName.textContent = files[0].name;
    audioWrap.style.display = 'flex';
  }

  // Drag and drop
  dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('dragover'); };
  dropzone.ondragleave = () => dropzone.classList.remove('dragover');
  dropzone.ondrop = (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  // Run Transcription
  document.getElementById('btnStartTranscribe').onclick = () => startTranscription();
  document.getElementById('btnRunPath').onclick = () => startTranscription();

  function startTranscription() {
    const pathVal = serverPath.value.trim();
    if (!selectedFiles.length && !pathVal) {
      alert("Please select an audio file or enter a path on disk!");
      return;
    }

    if (pathVal) {
      const fileName = pathVal.replace(/^.*[\\\/]/, '');
      audioPlayer.src = `/api/audio?path=${encodeURIComponent(pathVal)}`;
      audioTrackName.textContent = fileName || pathVal;
      audioWrap.style.display = 'flex';
    }

    const strategyVal = document.getElementById('strategy').value;
    const useManual = (k) => strategyVal !== 'auto' || touchedControls.has(k);

    const rangeStartRaw = document.getElementById('rangeStart').value.trim();
    const rangeEndRaw = document.getElementById('rangeEnd').value.trim();
    const payload = {
      strategy: strategyVal,
      model: document.getElementById('modelSelect').value || null,
      rumble_cutoff: useManual('rumble_cutoff') ? parseFloat(rumbleCutoff.value) : null,
      formant_boost: useManual('formant_boost') ? parseFloat(formantBoost.value) : null,
      blank_penalty: useManual('blank_penalty') ? parseFloat(blankPenalty.value) : null,
      window_size: useManual('window_size') ? parseFloat(document.getElementById('windowSize').value) : null,
      step_size: useManual('step_size') ? parseFloat(document.getElementById('stepSize').value) : null,
      threads: parseInt(document.getElementById('threads').value),
      recursive: document.getElementById('recursive').checked,
      server_path: pathVal || null,
      start_sec: rangeStartRaw !== '' ? parseFloat(rangeStartRaw) : null,
      end_sec: rangeEndRaw !== '' ? parseFloat(rangeEndRaw) : null,
      // Multilingual & Translation
      source_lang: document.getElementById('langSource').value,
      target_lang: document.getElementById('langTarget').value,
      translate: document.getElementById('enableTranslation').checked,
    };

    // Show/hide language detection info
    const langDetectInfo = document.getElementById('langDetectInfo');
    if (payload.source_lang === 'auto') {
      langDetectInfo.style.display = 'block';
      document.getElementById('detectedLangText').textContent = 'Detecting...';
      document.getElementById('detectedLangConf').textContent = '';
    } else {
      langDetectInfo.style.display = 'none';
    }

    reasoningBox.style.display = 'none';
    errorBanner.style.display = 'none';
    document.getElementById('snrBadge').style.display = 'none';
    progressWrap.style.display = 'block';
    progFill.style.width = '0%';
    progPct.textContent = '0%';
    progFile.textContent = 'Starting…';
    progTime.textContent = '00:00 / 00:00';
    progSpeed.textContent = '';
    segmentsHistory = [];
    output.value = "";
    segmentsList.innerHTML = "";
    updateStats();

    function onFetchError(err) {
      progressWrap.style.display = 'none';
      errorBanner.textContent = 'Network error: ' + (err.message || err);
      errorBanner.style.display = 'block';
      procStats.textContent = 'Failed';
    }

    if (selectedFiles.length && !pathVal) {
      // Upload via FormData
      const formData = new FormData();
      for (const f of selectedFiles) formData.append('files', f);
      formData.append('config', JSON.stringify(payload));

      procStats.textContent = "Uploading & processing…";
      fetch('/api/transcribe_upload', { method: 'POST', body: formData })
        .then(res => {
          if (!res.ok) {
            res.text().then(t => {
              errorBanner.textContent = 'Server error ' + res.status + ': ' + t.substring(0, 200);
              errorBanner.style.display = 'block';
              progressWrap.style.display = 'none';
              procStats.textContent = 'Error';
            });
            return;
          }
          handleSSE(res);
        })
        .catch(onFetchError);
    } else {
      // Run on local path
      procStats.textContent = "Scanning & processing local path…";
      fetch('/api/transcribe_path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
        .then(res => {
          if (!res.ok) {
            res.text().then(t => {
              errorBanner.textContent = 'Server error ' + res.status + ': ' + t.substring(0, 200);
              errorBanner.style.display = 'block';
              progressWrap.style.display = 'none';
              procStats.textContent = 'Error';
            });
            return;
          }
          handleSSE(res);
        })
        .catch(onFetchError);
    }
  }

  function handleSSE(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    function read() {
      reader.read().then(({ done, value }) => {
        if (done) {
          // Only update if not already set by "done" event
          if (!procStats.textContent.startsWith('Completed') && !procStats.textContent.startsWith('Error') && !procStats.textContent.startsWith('Cancelled')) {
            procStats.textContent = "Finished!";
          }
          progressWrap.style.display = 'none';
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();

        for (const block of lines) {
          if (!block.trim()) continue;
          let eventType = "message";
          let dataStr = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim();
            if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
          }

          try {
            const data = JSON.parse(dataStr);
            if (eventType === "reasoning") {
              reasoningList.innerHTML = "";
              for (const r of data.reasons) {
                const li = document.createElement('li');
                li.textContent = `• ${r}`;
                reasoningList.appendChild(li);
              }
              reasoningBox.style.display = 'block';
              const snrBadge = document.getElementById('snrBadge');
              const snrText = document.getElementById('snrBadgeText');
              if (typeof data.snr_db === 'number') {
                snrText.textContent = `SNR ${data.snr_db >= 0 ? '+' : ''}${data.snr_db.toFixed(1)} dB`;
                snrBadge.style.display = 'block';
              } else {
                // Extract SNR from reasoning lines if not provided directly
                const snrLine = data.reasons.find(r => r.includes('dB') || r.includes('SNR'));
                if (snrLine) { snrText.textContent = snrLine; snrBadge.style.display = 'block'; }
              }
            } else if (eventType === "progress") {
              progFill.style.width = `${data.pct}%`;
              progPct.textContent = `${data.pct}%`;
              progFile.textContent = data.file || "Processing";
              progTime.textContent = `${data.cur_time} / ${data.tot_time}`;
              progSpeed.textContent = `Speed: ${data.rtf}x | ETA: ${data.eta}`;
            } else if (eventType === "language_detected") {
              // Update language detection display
              const langDetectInfo = document.getElementById('langDetectInfo');
              langDetectInfo.style.display = 'block';
              document.getElementById('detectedLangText').textContent = `Detected: ${data.lang.toUpperCase()}`;
              document.getElementById('detectedLangConf').textContent = `${(data.confidence * 100).toFixed(0)}%`;
            } else if (eventType === "segment") {
              segmentsHistory.push(data);
              const isBilingual = data.translated !== undefined && data.translated !== null;
              const tsInterval = `[${formatSec(data.start)} -> ${formatSec(data.end)}]`;
              
              // Build display text with preserved timeline interval (Rule 4)
              let displayText = `${tsInterval} ${data.text}`;
              if (isBilingual && data.translated) {
                displayText = `${tsInterval} ${data.text} / ${data.translated}`;
              }
              if (data.layer === "subnoise-buried" && typeof data.snr_db === "number") {
                const fgNote = typeof data.depth_vs_foreground_db === "number" ? `, ${data.depth_vs_foreground_db.toFixed(1)}dB vs fg` : "";
                displayText += `  [buried: ${data.snr_db.toFixed(1)}dB vs floor${fgNote}]`;
              }
              
              if (output.value && !output.value.endsWith("\n")) {
                output.value += "\n";
              }
              output.value += displayText;
              output.scrollTop = output.scrollHeight;

              // Append interactive cue item
              const segEl = document.createElement('div');
              segEl.className = 'segment-item';
              segEl.dataset.start = data.start;
              segEl.dataset.end = data.end;
              
              let segHtml = `<span class="seg-time">${formatSec(data.start)} → ${formatSec(data.end)}</span>`;
              if (isBilingual && data.translated) {
                segHtml += `<span class="seg-text">${escapeHtml(data.text)}</span><span class="seg-text" style="color: var(--text-muted); font-size: 0.85rem;"> → ${escapeHtml(data.translated)}</span>`;
              } else {
                segHtml += `<span class="seg-text">${escapeHtml(data.text)}</span>`;
              }
              if (data.layer === "subnoise-buried" && typeof data.snr_db === "number") {
                const fgDelta = typeof data.depth_vs_foreground_db === "number" ? ` · ${data.depth_vs_foreground_db.toFixed(1)}dB fg` : "";
                segHtml += `<span class="seg-time" style="color: var(--warning); font-size: 0.72rem; margin-left: 6px;">[BURIED ${data.snr_db >= 0 ? '+' : ''}${data.snr_db.toFixed(1)}dB floor${fgDelta}]</span>`;
              } else if (data.layer === "background-faint" && typeof data.snr_db === "number") {
                const fgDelta = typeof data.depth_vs_foreground_db === "number" ? ` · ${data.depth_vs_foreground_db.toFixed(1)}dB fg` : "";
                segHtml += `<span class="seg-time" style="color: var(--accent); font-size: 0.72rem; margin-left: 6px;">[FAINT ${data.snr_db >= 0 ? '+' : ''}${data.snr_db.toFixed(1)}dB floor${fgDelta}]</span>`;
              }
              if (data.lang) {
                segHtml += `<span class="seg-time" style="color: var(--accent); font-size: 0.7rem;">[${data.lang.toUpperCase()}]</span>`;
              }
              segEl.innerHTML = segHtml;
              segEl.onclick = () => {
                audioPlayer.currentTime = data.start;
                audioPlayer.play();
              };
              segmentsList.appendChild(segEl);

              updateStats();
            } else if (eventType === "done") {
              progressWrap.style.display = 'none';
              procStats.textContent = `Completed in ${data.elapsed.toFixed(1)}s (RTF: ${data.rtf.toFixed(2)}x)`;
              const histEntry = {
                ts: new Date().toLocaleTimeString(),
                text: output.value,
                segments: segmentsHistory.slice(),
                label: (selectedFiles.length ? selectedFiles[0].name : (serverPath.value || 'unknown')),
              };
              transcriptHistory.unshift(histEntry);
              renderHistory();
            } else if (eventType === "error") {
              progressWrap.style.display = 'none';
              errorBanner.textContent = 'Transcription error: ' + data.message;
              errorBanner.style.display = 'block';
              procStats.textContent = 'Error';
            } else if (eventType === "aborted") {
              progressWrap.style.display = 'none';
              procStats.textContent = 'Cancelled';
            }
          } catch (e) {
            console.error("SSE parse error", e, dataStr);
          }
        }
        read();
      }).catch(err => {
        progressWrap.style.display = 'none';
        procStats.textContent = 'Stream error';
        console.error('SSE read error:', err);
      });
    }
    read();
  }

  function updateStats() {
    const text = output.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const chars = text.length;
    wordStats.textContent = `${words} words | ${chars} characters`;
  }
  output.addEventListener('input', updateStats);

  // History modal
  function renderHistory() {
    const list = document.getElementById('historyList');
    if (!transcriptHistory.length) {
      list.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem;">No previous transcriptions this session.</p>';
      return;
    }
    list.innerHTML = '';
    transcriptHistory.forEach((entry, i) => {
      const div = document.createElement('div');
      div.style.cssText = 'background:var(--bg-card); border:1px solid var(--border); border-radius:6px; padding:10px 12px; cursor:pointer;';
      div.innerHTML = `<div style="display:flex; justify-content:space-between; margin-bottom:4px;"><strong style="font-size:0.85rem;">${escapeHtml(entry.label)}</strong><span style="color:var(--text-muted); font-size:0.78rem;">${entry.ts}</span></div><div style="font-size:0.8rem; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(entry.text.substring(0, 120))}${entry.text.length > 120 ? '...' : ''}</div>`;
      div.onclick = () => {
        output.value = entry.text;
        segmentsHistory = entry.segments;
        segmentsList.innerHTML = '';
        entry.segments.forEach(d => {
          const segEl = document.createElement('div');
          segEl.className = 'segment-item';
          segEl.dataset.start = d.start;
          segEl.dataset.end = d.end;
          segEl.innerHTML = `<span class="seg-time">${formatSec(d.start)} → ${formatSec(d.end)}</span><span class="seg-text">${escapeHtml(d.text)}</span>`;
          segEl.onclick = () => { audioPlayer.currentTime = d.start; audioPlayer.play(); };
          segmentsList.appendChild(segEl);
        });
        updateStats();
        document.getElementById('historyOverlay').style.display = 'none';
      };
      list.appendChild(div);
    });
  }
  document.getElementById('btnHistory').onclick = () => {
    renderHistory();
    document.getElementById('historyOverlay').style.display = 'flex';
  };
  document.getElementById('btnCloseHistory').onclick = () => {
    document.getElementById('historyOverlay').style.display = 'none';
  };

  // Copy
  document.getElementById('btnCopy').onclick = () => {
    navigator.clipboard.writeText(output.value);
    const orig = document.getElementById('btnCopy').innerHTML;
    document.getElementById('btnCopy').innerHTML = "Copied!";
    setTimeout(() => document.getElementById('btnCopy').innerHTML = orig, 1500);
  };

  // Clear
  document.getElementById('btnClear').onclick = () => {
    output.value = "";
    segmentsList.innerHTML = "";
    segmentsHistory = [];
    updateStats();
  };

  // Cancel Transcription
  document.getElementById('btnCancel').onclick = () => {
    fetch('/api/abort', { method: 'POST' })
      .then(() => {
        procStats.textContent = "Cancelled";
        const btn = document.getElementById('btnCancel');
        btn.textContent = "Cancelled";
        setTimeout(() => btn.textContent = "Cancel", 2000);
      })
      .catch(() => {});
  };

  // Export TXT
  document.getElementById('btnExportTxt').onclick = () => downloadFile("transcript.txt", output.value, "text/plain");

  // Export JSONL
  document.getElementById('btnExportJsonl').onclick = () => {
    const lines = segmentsHistory.map(s => JSON.stringify(s)).join("\n");
    downloadFile("transcript.jsonl", lines, "application/json");
  };

  function formatSubtitleClock(sec, sep) {
    const totalMs = Math.max(0, Math.round((sec || 0) * 1000));
    const ms = totalMs % 1000;
    const totalSec = Math.floor(totalMs / 1000);
    const s = totalSec % 60;
    const totalMin = Math.floor(totalSec / 60);
    const m = totalMin % 60;
    const h = Math.floor(totalMin / 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}${sep}${ms.toString().padStart(3, '0')}`;
  }

  // Export SRT
  document.getElementById('btnExportSrt').onclick = () => {
    let srt = "";
    segmentsHistory.forEach((seg, idx) => {
      const body = (seg.translated !== undefined && seg.translated !== null && seg.translated !== "")
        ? `${seg.text}\n${seg.translated}`
        : seg.text;
      srt += `${idx + 1}\n${formatSubtitleClock(seg.start, ',') } --> ${formatSubtitleClock(seg.end, ',')}\n${body}\n\n`;
    });
    downloadFile("transcript.srt", srt, "text/plain");
  };

  // Export VTT
  document.getElementById('btnExportVtt').onclick = () => {
    let vtt = "WEBVTT\n\n";
    segmentsHistory.forEach((seg, idx) => {
      const body = (seg.translated !== undefined && seg.translated !== null && seg.translated !== "")
        ? `${seg.text}\n${seg.translated}`
        : seg.text;
      vtt += `${idx + 1}\n${formatSubtitleClock(seg.start, '.')} --> ${formatSubtitleClock(seg.end, '.')}\n${body}\n\n`;
    });
    downloadFile("transcript.vtt", vtt, "text/vtt");
  };

  function downloadFile(filename, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Hotkey Modal
  const modal = document.getElementById('modalOverlay');
  document.getElementById('btnHotkeys').onclick = () => modal.style.display = 'flex';
  document.getElementById('btnCloseModal').onclick = () => modal.style.display = 'none';
  modal.onclick = (e) => { if (e.target === modal) modal.style.display = 'none'; };

  // Global Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const ctrl = isMac ? e.metaKey : e.ctrlKey;

    if (e.code === 'Space') {
      if (document.activeElement !== output && document.activeElement !== serverPath) {
        e.preventDefault();
        if (audioPlayer.paused) {
          audioPlayer.play();
        } else {
          audioPlayer.pause();
        }
      }
    } else if (e.altKey && e.key === 'ArrowLeft') {
      e.preventDefault();
      audioPlayer.currentTime = Math.max(0, audioPlayer.currentTime - 5);
    } else if (e.altKey && e.key === 'ArrowRight') {
      e.preventDefault();
      audioPlayer.currentTime = Math.min(audioPlayer.duration || 0, audioPlayer.currentTime + 5);
    } else if (e.key === '?' || e.key === 'F1') {
      if (document.activeElement !== output) {
        e.preventDefault();
        modal.style.display = modal.style.display === 'flex' ? 'none' : 'flex';
      }
    } else if (e.key === 'Escape') {
      modal.style.display = 'none';
    } else if (ctrl && (e.key === 'o' || e.key === 'O')) {
      e.preventDefault();
      fileInput.click();
    } else if (ctrl && (e.key === 'd' || e.key === 'D')) {
      e.preventDefault();
      folderInput.click();
    } else if (ctrl && e.key === 'Enter') {
      e.preventDefault();
      startTranscription();
    } else if (ctrl && (e.key === 's' || e.key === 'S')) {
      e.preventDefault();
      document.getElementById('btnExportTxt').click();
    } else if (e.altKey && (e.key === 'c' || e.key === 'C')) {
      e.preventDefault();
      document.getElementById('btnCopy').click();
    } else if (e.altKey && (e.key === 'x' || e.key === 'X')) {
      e.preventDefault();
      document.getElementById('btnClear').click();
    } else if (e.altKey && (e.key === 'z' || e.key === 'Z')) {
      e.preventDefault();
      document.getElementById('btnCancel').click();
    } else if (e.altKey && (e.key === 'a' || e.key === 'A')) {
      e.preventDefault();
      document.getElementById('strategy').value = 'auto';
    } else if (e.altKey && (e.key === 'n' || e.key === 'N')) {
      e.preventDefault();
      document.getElementById('strategy').value = 'subnoise';
    }
  });
</script>
</body>
</html>
"""


class _TranscriptionAborted(Exception):
    """Raised internally to unwind mid-file transcription promptly when the user clicks Cancel."""


_SENSITIVE_PATH_PARTS = {
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".git", ".hg", ".svn",
    ".env", ".netrc", ".npmrc", ".pypirc", ".azure", ".gcloud", ".pki",
}


def _is_same_origin_or_local(headers) -> bool:
    """Reject cross-origin browser requests (DNS-rebinding / CSRF defense) while allowing local requests."""
    origin = (headers.get("Origin") or "").strip()
    if not origin:
        return True
    try:
        parsed_origin = urllib.parse.urlparse(origin)
        host = (parsed_origin.hostname or "").lower()
    except Exception:
        return False
    if host in ("127.0.0.1", "localhost", "::1"):
        return True
    req_host = (headers.get("Host") or "").split(":")[0].strip("[]").lower()
    return bool(req_host and host == req_host)


def _validate_local_media_path(raw_path: str, allow_directory: bool = False) -> tuple[Path | None, int, str]:
    """Validate local path for versatile multi-drive media access while blocking non-media & credential paths."""
    from .cli import MEDIA_EXTENSIONS

    if not raw_path or not isinstance(raw_path, str):
        return None, 400 if allow_directory else 404, "Path not provided"

    p_raw = Path(raw_path)
    if not p_raw.exists():
        return None, 400 if allow_directory else 404, "Path not found"

    try:
        p = p_raw.resolve()
    except OSError:
        return None, 400, "Invalid path"

    # Block credential / secret dot-directories regardless of drive or extension
    lowered_parts = {part.lower() for part in p.parts}
    if lowered_parts & _SENSITIVE_PATH_PARTS:
        return None, 403, "Access to hidden or credential directory denied"

    if p.is_dir():
        if not allow_directory:
            return None, 404, "Audio file not found"
        return p, 200, "OK"

    if not p.is_file():
        return None, 404, "Audio file not found"

    if p.suffix.lower() not in MEDIA_EXTENSIONS:
        return None, 403, f"Unsupported media file extension: {p.suffix or '(none)'}"

    return p, 200, "OK"


class STTRequestHandler(http.server.BaseHTTPRequestHandler):
    """Zero-dependency HTTP server delivering the talk-to-typed frontend and streaming API."""

    def log_message(self, format, *args):
        # Keep stdout/stderr clean unless verbose
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            content = HTML_PAGE.replace("%STT_VERSION%", __version__).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif parsed.path == "/api/status":
            cfg = self.server.cfg
            md = Path(cfg["models_dir"])
            data = {
                "version": __version__,
                "models_dir": str(md),
                "threads": cfg.get("threads", 4),
            }
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/models":
            cfg = self.server.cfg
            rows = asr.catalog(cfg)
            body = json.dumps(rows).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/audio":
            if not _is_same_origin_or_local(self.headers):
                self.send_error(403, "Cross-origin request denied")
                return

            query = urllib.parse.parse_qs(parsed.query)
            target_path = query.get("path", [None])[0]
            p, status_code, err_msg = _validate_local_media_path(target_path, allow_directory=False)
            if p is None:
                self.send_error(status_code, err_msg)
                return

            content_type = "audio/wav"
            suf = p.suffix.lower()
            if suf in (".mp3",):
                content_type = "audio/mpeg"
            elif suf in (".m4a", ".aac"):
                content_type = "audio/mp4"
            elif suf in (".ogg", ".opus"):
                content_type = "audio/ogg"
            elif suf in (".flac",):
                content_type = "audio/flac"
            elif suf in (".mp4", ".mkv", ".webm", ".mov", ".m4v"):
                content_type = "video/mp4"

            try:
                size = p.stat().st_size
                range_header = self.headers.get("Range")

                if range_header:
                    range_match = range_header.strip().lower()
                    if not range_match.startswith("bytes="):
                        self.send_error(416, "Invalid Range header")
                        return
                    range_spec = range_match[6:]
                    parts = range_spec.split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if len(parts) > 1 and parts[1] else size - 1

                    if start >= size or end >= size or start > end:
                        self.send_response(416, "Range Not Satisfiable")
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return

                    content_length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(content_length))
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()

                    with open(p, "rb") as f:
                        f.seek(start)
                        remaining = content_length
                        chunk_size = 8192
                        while remaining > 0:
                            read_size = min(chunk_size, remaining)
                            data = f.read(read_size)
                            if not data:
                                break
                            self.wfile.write(data)
                            remaining -= len(data)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    with open(p, "rb") as f:
                        shutil.copyfileobj(f, self.wfile)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                self.send_error(500, f"Internal server error: {e}")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if not _is_same_origin_or_local(self.headers):
            self.send_error(403, "Cross-origin request denied")
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/transcribe_path":
            self.handle_transcribe_path()
        elif parsed.path == "/api/transcribe_upload":
            self.handle_transcribe_upload()
        elif parsed.path == "/api/abort":
            self.handle_abort()
        else:
            self.send_error(404, "Not Found")

    def handle_abort(self):
        _transcription_abort.set()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "aborted"}')

    def handle_transcribe_path(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            req = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON request body")
            return
        server_path = req.get("server_path")

        p, status_code, err_msg = _validate_local_media_path(server_path, allow_directory=True)
        if p is None:
            self.send_error(status_code, err_msg)
            return

        from .cli import collect_media_paths
        raw_paths = collect_media_paths([str(p)], recursive=req.get("recursive", True))
        paths = [
            mp for mp in raw_paths
            if not ({part.lower() for part in Path(mp).parts} & _SENSITIVE_PATH_PARTS)
        ]
        if not paths:
            self.send_error(400, "No supported audio/video files found in given path.")
            return

        self.stream_transcription_sse(paths, req)

    def handle_transcribe_upload(self):
        content_length = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")
        raw_body = self.rfile.read(content_length)

        parts = _parse_multipart(content_type, raw_body)
        if not parts:
            self.send_error(400, "Could not parse multipart upload (missing boundary?).")
            return

        req = {}
        temp_files = []
        try:
            for part in parts:
                if part["name"] == "config":
                    req = json.loads(part["data"].decode("utf-8", errors="replace"))
                elif part.get("filename"):
                    suffix = Path(part["filename"]).suffix or ".wav"
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    try:
                        tmp.write(part["data"])
                    finally:
                        tmp.close()
                    temp_files.append((tmp.name, part["filename"]))

            if not temp_files:
                self.send_error(400, "No audio files found in upload.")
                return

            self.stream_transcription_sse(
                [t[0] for t in temp_files], req,
                display_names={t[0]: t[1] for t in temp_files},
            )
        finally:
            for tmp_path, _ in temp_files:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def stream_transcription_sse(self, paths, req, display_names=None):
        from . import translation

        _transcription_abort.clear()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send_sse(event_type, payload):
            msg = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
            try:
                self.wfile.write(msg)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        try:
            cfg = self.server.cfg
            threads = req.get("threads") or cfg.get("threads", 4)
            blank_penalty = req.get("blank_penalty") if req.get("blank_penalty") is not None else cfg.get("blank_penalty")

            source_lang = req.get("source_lang", "auto")
            target_lang = req.get("target_lang", "en")
            do_translate = bool(req.get("translate", False))
            use_multilingual = do_translate or (source_lang not in ("auto", "en"))

            selected_model = req.get("model") or None
            if use_multilingual:
                rec = asr.get_multilingual_offline(cfg, model=selected_model, threads=threads, blank_penalty=blank_penalty)
            else:
                rec = asr.get_offline(cfg, model=selected_model, threads=threads, blank_penalty=blank_penalty)
        except Exception as e:
            send_sse("error", {"message": f"Failed to load ASR model: {e}"})
            return

        strategy = req.get("strategy", "auto")
        is_subnoise = strategy in ("auto", "subnoise")
        is_dynamic = strategy == "dynamic"
        if is_dynamic:
            is_subnoise = True  # dynamic uses same auto-tune path + CN-ROVER ensemble

        t0_total = time.time()
        total_audio_sec = 0.0

        for path in paths:
            if _transcription_abort.is_set():
                send_sse("aborted", {"message": "Transcription cancelled by user"})
                return

            disp_name = display_names.get(path, Path(path).name) if display_names else Path(path).name
            range_start = req.get("start_sec")
            range_end = req.get("end_sec")
            try:
                samples = audio.decode(
                    path,
                    start_sec=float(range_start) if range_start is not None else None,
                    end_sec=float(range_end) if range_end is not None else None,
                )
            except Exception as e:
                send_sse("segment", {"file": disp_name, "start": 0, "end": 0, "text": f"[Error decoding {disp_name}: {e}]"})
                continue
            if samples is None:
                send_sse("segment", {"file": disp_name, "start": 0, "end": 0, "text": f"[Error decoding {disp_name}: unreadable media file]"})
                continue

            dur = len(samples) / audio.SR
            total_audio_sec += dur

            t_start_file = time.time()

            def progress_cb(current_step=1, total_steps=1, current_sec=None, total_sec=None,
                            _t=t_start_file, _disp=disp_name, _dur=dur):
                if _transcription_abort.is_set():
                    raise _TranscriptionAborted()
                tot_s_val = float(total_sec if total_sec is not None else _dur)
                pct = int((current_step / max(total_steps, 1)) * 100)
                cur_s_val = float(current_sec if current_sec is not None else (tot_s_val * pct / 100.0))
                elapsed = time.time() - _t
                rtf = elapsed / max(cur_s_val, 0.001)
                eta_sec = max(0.0, tot_s_val - cur_s_val) * rtf
                cur_m, cur_s = divmod(int(cur_s_val), 60)
                tot_m, tot_s = divmod(int(tot_s_val), 60)
                eta_m, eta_s = divmod(int(eta_sec), 60)
                send_sse("progress", {
                    "file": _disp,
                    "pct": min(100, pct),
                    "cur_time": f"{cur_m:02d}:{cur_s:02d}",
                    "tot_time": f"{tot_m:02d}:{tot_s:02d}",
                    "rtf": round(rtf, 2),
                    "eta": f"{eta_m:02d}:{eta_s:02d}",
                })

            try:
                params = None
                if is_subnoise:
                    # Pass actual waveform samples so physical SNR, hum, and rumble drive auto-tuning!
                    params = robust.auto_tune_settings(dur, samples=samples)
                    if req.get("rumble_cutoff") is not None:
                        params["low_hz"] = float(req["rumble_cutoff"])
                    if req.get("formant_boost") is not None:
                        params["boost_db"] = float(req["formant_boost"])
                    if req.get("blank_penalty") is not None:
                        params["blank_penalty"] = float(req["blank_penalty"])
                    if req.get("window_size") is not None:
                        params["win_sec"] = float(req["window_size"])
                    if req.get("step_size") is not None:
                        params["step_sec"] = float(req["step_size"])

                    snr_val = params.get("snr_db")
                    send_sse("reasoning", {
                        "file": disp_name, "duration": dur, "reasons": params["reasoning"],
                        "snr_db": snr_val,
                    })

                # Matrix "all-models" mode
                if strategy == "all":
                    from . import matrix as _matrix
                    send_sse("progress", {"file": disp_name, "pct": 5, "cur_time": "00:00",
                                          "tot_time": f"{int(dur//60):02d}:{int(dur%60):02d}",
                                          "rtf": 0.0, "eta": "..."})
                    try:
                        matrix_results = _matrix.run_matrix(
                            samples, cfg, audio_path=path,
                            threads=threads, time_offset=float(range_start or 0),
                        )
                        for row in matrix_results:
                            send_sse("segment", {
                                "file": disp_name,
                                "start": round(row.get("start", 0), 2),
                                "end": round(row.get("end", 0), 2),
                                "text": f"[{row.get('combo', '?')}] {row.get('text', '')}",
                            })
                    except Exception as e:
                        send_sse("segment", {"file": disp_name, "start": 0, "end": 0,
                                             "text": f"[Matrix error: {e}]"})
                    continue

                level = 0 if is_subnoise else (3 if strategy == "robust" else (5 if strategy == "distant" else 0))

                if use_multilingual:
                    triples = robust.transcribe_multilingual(
                        rec, samples, cfg,
                        source_lang=source_lang,
                        target_lang=target_lang if do_translate else None,
                        translate=do_translate,
                        level=level,
                        progress_cb=progress_cb,
                        subnoise=is_subnoise,
                        params=params,
                    )
                else:
                    triples = robust.transcribe(
                        rec, samples, cfg, level=level, subnoise=is_subnoise,
                        params=params, progress_cb=progress_cb,
                        model=None, threads=threads,
                    )

                detected_lang = source_lang if source_lang != "auto" else None
                if source_lang == "auto" and triples:
                    joined_src = " ".join(item[2] for item in triples).strip()
                    detected_lang, conf = translation.detect_language_simple(joined_src)
                    send_sse("language_detected", {"lang": detected_lang, "confidence": conf})

            except _TranscriptionAborted:
                send_sse("aborted", {"message": "Transcription cancelled by user"})
                return
            except Exception as e:
                send_sse("segment", {"file": disp_name, "start": 0, "end": 0,
                                     "text": f"[Error transcribing {disp_name}: {e}]"})
                continue

            seg_diags = robust.get_last_segment_diagnostics()
            diag_by_idx = seg_diags if len(seg_diags) == len(triples) else None
            diag_by_span = {(round(d["start"], 2), round(d["end"], 2)): d for d in seg_diags}

            for idx, item in enumerate(triples):
                if _transcription_abort.is_set():
                    send_sse("aborted", {"message": "Transcription cancelled by user"})
                    return
                if len(item) == 4:
                    s, e, src_txt, tgt_txt = item
                    payload = {
                        "file": disp_name,
                        "start": round(s, 2),
                        "end": round(e, 2),
                        "text": src_txt,
                        "translated": tgt_txt,
                        "lang": detected_lang or source_lang,
                    }
                else:
                    s, e, text = item
                    payload = {"file": disp_name, "start": round(s, 2), "end": round(e, 2), "text": text}
                    if detected_lang:
                        payload["lang"] = detected_lang

                d = diag_by_idx[idx] if diag_by_idx is not None else diag_by_span.get((round(s, 2), round(e, 2)))
                if d:
                    payload["layer"] = d.get("layer")
                    payload["snr_db"] = d.get("snr_db")
                    payload["burial_below_floor_db"] = d.get("burial_below_floor_db")
                    if d.get("depth_vs_foreground_db") is not None:
                        payload["depth_vs_foreground_db"] = d.get("depth_vs_foreground_db")
                send_sse("segment", payload)

        dt = time.time() - t0_total
        send_sse("done", {
            "elapsed": dt,
            "total_audio": total_audio_sec,
            "rtf": dt / max(total_audio_sec, 1e-6),
        })


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def launch_ui(cfg: dict, host: str = "127.0.0.1", port: int = 8080, open_browser: bool = True):
    """Launch the Web front-end server."""
    server = ThreadedHTTPServer((host, port), STTRequestHandler)
    server.cfg = cfg
    url = f"http://{host}:{port}/"

    print("=" * 65, file=sys.stderr)
    print(f"  talk-to-typed Web UI running at: {url}", file=sys.stderr)
    print("  Press Ctrl+C to stop the server.", file=sys.stderr)
    print("=" * 65, file=sys.stderr)

    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stt ui] Server stopped.", file=sys.stderr)
    finally:
        server.server_close()
