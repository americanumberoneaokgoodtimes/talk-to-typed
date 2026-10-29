"""Web front-end for talk-to-typed / stt.

Features:
- File picker, folder picker, and drag-and-drop input.
- Real-time streaming transcription text box.
- Dynamic auto-tuning, sub-noise formant scanning, and denoising controls.
- Complete keyboard hotkeys for hands-free navigation.
- Zero external dependencies (pure Python standard library http.server).
"""
import email
import email.policy
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

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>talk-to-typed — Offline Speech-to-Text</title>
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
    <span class="badge">v0.2.0</span>
  </div>
  <div class="header-actions">
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
        <option value="auto" selected>Auto Dynamic Sub-Noise (Recommended)</option>
        <option value="subnoise">Force Aggressive Sub-Noise Scan</option>
        <option value="standard">Standard Silero VAD (Clean Audio)</option>
        <option value="robust">Robust Denoiser (Level 3)</option>
        <option value="distant">Distant Whisper (Level 5)</option>
      </select>
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
      <tr><td><kbd>?</kbd> / <kbd>F1</kbd></td><td>Toggle this Hotkey Guide</td></tr>
      <tr><td><kbd>Esc</kbd></td><td>Close Modal</td></tr>
    </table>
  </div>
</div>

<script>
  let selectedFiles = [];
  let segmentsHistory = [];

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
  const progressWrap = document.getElementById('progressWrap');
  const progFill = document.getElementById('progFill');
  const progPct = document.getElementById('progPct');
  const progFile = document.getElementById('progFile');
  const progTime = document.getElementById('progTime');
  const progSpeed = document.getElementById('progSpeed');

  function formatSec(s) {
    if (isNaN(s)) return "00:00";
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1);
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(4, '0')}`;
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

  // Sliders
  const rumbleCutoff = document.getElementById('rumbleCutoff');
  const valRumble = document.getElementById('valRumble');
  rumbleCutoff.addEventListener('input', () => valRumble.textContent = `${rumbleCutoff.value} Hz`);

  const formantBoost = document.getElementById('formantBoost');
  const valBoost = document.getElementById('valBoost');
  formantBoost.addEventListener('input', () => valBoost.textContent = `+${parseFloat(formantBoost.value).toFixed(1)} dB`);

  const blankPenalty = document.getElementById('blankPenalty');
  const valBlank = document.getElementById('valBlank');
  blankPenalty.addEventListener('input', () => valBlank.textContent = parseFloat(blankPenalty.value).toFixed(1));

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

    const payload = {
      strategy: document.getElementById('strategy').value,
      rumble_cutoff: parseFloat(rumbleCutoff.value),
      formant_boost: parseFloat(formantBoost.value),
      blank_penalty: parseFloat(blankPenalty.value),
      window_size: parseFloat(document.getElementById('windowSize').value),
      step_size: parseFloat(document.getElementById('stepSize').value),
      threads: parseInt(document.getElementById('threads').value),
      recursive: document.getElementById('recursive').checked,
      server_path: pathVal || null
    };

    reasoningBox.style.display = 'none';
    progressWrap.style.display = 'block';
    segmentsHistory = [];
    output.value = "";
    segmentsList.innerHTML = "";
    updateStats();

    if (selectedFiles.length && !pathVal) {
      // Upload via FormData
      const formData = new FormData();
      for (const f of selectedFiles) formData.append('files', f);
      formData.append('config', JSON.stringify(payload));

      procStats.textContent = "Uploading & processing...";
      fetch('/api/transcribe_upload', { method: 'POST', body: formData })
        .then(res => handleSSE(res));
    } else {
      // Run on local path
      procStats.textContent = "Scanning & processing local path...";
      fetch('/api/transcribe_path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).then(res => handleSSE(res));
    }
  }

  function handleSSE(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    function read() {
      reader.read().then(({ done, value }) => {
        if (done) {
          procStats.textContent = "Finished!";
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
                li.innerHTML = `• ${r}`;
                reasoningList.appendChild(li);
              }
              reasoningBox.style.display = 'block';
            } else if (eventType === "progress") {
              progFill.style.width = `${data.pct}%`;
              progPct.textContent = `${data.pct}%`;
              progFile.textContent = data.file || "Processing";
              progTime.textContent = `${data.cur_time} / ${data.tot_time}`;
              progSpeed.textContent = `Speed: ${data.rtf}x | ETA: ${data.eta}`;
            } else if (eventType === "segment") {
              segmentsHistory.push(data);
              if (output.value && !output.value.endsWith("\n") && !output.value.endsWith(" ")) {
                output.value += " ";
              }
              output.value += data.text;
              output.scrollTop = output.scrollHeight;

              // Append interactive cue item
              const segEl = document.createElement('div');
              segEl.className = 'segment-item';
              segEl.dataset.start = data.start;
              segEl.dataset.end = data.end;
              segEl.innerHTML = `<span class="seg-time">${formatSec(data.start)} → ${formatSec(data.end)}</span><span class="seg-text">${escapeHtml(data.text)}</span>`;
              segEl.onclick = () => {
                audioPlayer.currentTime = data.start;
                audioPlayer.play();
              };
              segmentsList.appendChild(segEl);

              updateStats();
            } else if (eventType === "done") {
              procStats.textContent = `Completed in ${data.elapsed.toFixed(1)}s (RTF: ${data.rtf.toFixed(2)}x)`;
            }
          } catch (e) {
            console.error("SSE parse error", e, dataStr);
          }
        }
        read();
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

  // Export TXT
  document.getElementById('btnExportTxt').onclick = () => downloadFile("transcript.txt", output.value, "text/plain");

  // Export JSONL
  document.getElementById('btnExportJsonl').onclick = () => {
    const lines = segmentsHistory.map(s => JSON.stringify(s)).join("\n");
    downloadFile("transcript.jsonl", lines, "application/json");
  };

  // Export SRT
  document.getElementById('btnExportSrt').onclick = () => {
    let srt = "";
    segmentsHistory.forEach((seg, idx) => {
      const formatTime = (sec) => {
        const d = new Date(sec * 1000);
        return d.toISOString().substr(11, 8) + "," + ("00" + d.getMilliseconds()).slice(-3);
      };
      srt += `${idx + 1}\n${formatTime(seg.start)} --> ${formatTime(seg.end)}\n${seg.text}\n\n`;
    });
    downloadFile("transcript.srt", srt, "text/plain");
  };

  // Export VTT
  document.getElementById('btnExportVtt').onclick = () => {
    let vtt = "WEBVTT\n\n";
    segmentsHistory.forEach((seg, idx) => {
      const formatTime = (sec) => {
        const d = new Date(sec * 1000);
        return d.toISOString().substr(11, 8) + "." + ("00" + d.getMilliseconds()).slice(-3);
      };
      vtt += `${idx + 1}\n${formatTime(seg.start)} --> ${formatTime(seg.end)}\n${seg.text}\n\n`;
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
    } else if (e.altKey && (e.key === 'a' || e.key === 'A')) {
      e.preventDefault();
      document.getElementById('strategy').value = 'auto';
    } else if (e.altKey && (e.key === 'n' || e.key === 'N')) {
      e.preventDefault();
      document.getElementById('strategy').value = 'subnoise';
    }
  });
</script>

    if (e.key === '?' || e.key === 'F1') {
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


class STTRequestHandler(http.server.BaseHTTPRequestHandler):
    """Zero-dependency HTTP server delivering the talk-to-typed frontend and streaming API."""

    def log_message(self, format, *args):
        # Keep stdout/stderr clean unless verbose
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            content = HTML_PAGE.encode("utf-8")
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
        elif parsed.path == "/api/audio":
            query = urllib.parse.parse_qs(parsed.query)
            target_path = query.get("path", [None])[0]
            if not target_path or not Path(target_path).is_file():
                self.send_error(404, "Audio file not found")
                return
            p = Path(target_path)
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
            elif suf in (".mp4", ".mkv", ".webm"):
                content_type = "video/mp4"

            try:
                size = p.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            except Exception:
                pass
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/transcribe_path":
            self.handle_transcribe_path()
        elif parsed.path == "/api/transcribe_upload":
            self.handle_transcribe_upload()
        else:
            self.send_error(404, "Not Found")

    def handle_transcribe_path(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        req = json.loads(body)
        server_path = req.get("server_path")

        if not server_path or not os.path.exists(server_path):
            self.send_error(400, f"Path not found: {server_path}")
            return

        from .cli import collect_media_paths
        paths = collect_media_paths([server_path], recursive=req.get("recursive", True))
        if not paths:
            self.send_error(400, "No supported audio/video files found in given path.")
            return

        self.stream_transcription_sse(paths, req)

    def handle_transcribe_upload(self):
        content_length = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")
        raw_body = self.rfile.read(content_length)

        full_msg = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + raw_body
        msg = email.message_from_bytes(full_msg, policy=email.policy.default)

        req = {}
        temp_files = []
        try:
            if msg.is_multipart():
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    filename = part.get_filename()
                    if name == "config":
                        req = json.loads(part.get_payload())
                    elif filename:
                        suffix = Path(filename).suffix or ".wav"
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        payload_data = part.get_payload(decode=True)
                        tmp.write(payload_data)
                        tmp.close()
                        temp_files.append((tmp.name, filename))

            if not temp_files:
                self.send_error(400, "No audio files uploaded.")
                return

            self.stream_transcription_sse([t[0] for t in temp_files], req, display_names={t[0]: t[1] for t in temp_files})
        finally:
            for tmp_path, _ in temp_files:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def stream_transcription_sse(self, paths, req, display_names=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def send_sse(event_type, payload):
            msg = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
            try:
                self.wfile.write(msg)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        cfg = self.server.cfg
        threads = req.get("threads") or cfg.get("threads", 4)
        rec = asr.offline(cfg, None, threads)

        strategy = req.get("strategy", "auto")
        is_auto = strategy == "auto"
        is_subnoise = strategy in ("auto", "subnoise")

        t0_total = time.time()
        total_audio_sec = 0.0

        for path in paths:
            disp_name = display_names.get(path, Path(path).name) if display_names else Path(path).name
            try:
                samples = audio.decode(path)
            except Exception as e:
                send_sse("segment", {"file": disp_name, "start": 0, "end": 0, "text": f"[Error decoding {disp_name}: {e}]"})
                continue

            dur = len(samples) / audio.SR
            total_audio_sec += dur

            if is_subnoise:
                params = robust.auto_tune_settings(dur)
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

                send_sse("reasoning", {"file": disp_name, "duration": dur, "reasons": params["reasoning"]})

                t_start_file = time.time()
                def progress_cb(current_step, total_steps, current_sec, total_sec):
                    now = time.time()
                    elapsed = now - t_start_file
                    pct = int((current_step / total_steps) * 100) if total_steps > 0 else 0
                    rtf = elapsed / max(current_sec, 0.001)
                    rem_sec = max(0.0, total_sec - current_sec)
                    eta_sec = rem_sec * rtf
                    cur_m, cur_s = divmod(int(current_sec), 60)
                    tot_m, tot_s = divmod(int(total_sec), 60)
                    eta_m, eta_s = divmod(int(eta_sec), 60)

                    send_sse("progress", {
                        "file": disp_name,
                        "pct": pct,
                        "cur_time": f"{cur_m:02d}:{cur_s:02d}",
                        "tot_time": f"{tot_m:02d}:{tot_s:02d}",
                        "rtf": round(rtf, 2),
                        "eta": f"{eta_m:02d}:{eta_s:02d}"
                    })

                triples = robust.transcribe(
                    rec, samples, cfg, level=0, subnoise=True,
                    params=params, progress_cb=progress_cb
                )
            else:
                level = 3 if strategy == "robust" else (5 if strategy == "distant" else 0)
                triples = robust.transcribe(rec, samples, cfg, level=level, subnoise=False)

            for s, e, text in triples:
                send_sse("segment", {"file": disp_name, "start": round(s, 2), "end": round(e, 2), "text": text})

        dt = time.time() - t0_total
        send_sse("done", {
            "elapsed": dt,
            "total_audio": total_audio_sec,
            "rtf": dt / max(total_audio_sec, 1e-6)
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
