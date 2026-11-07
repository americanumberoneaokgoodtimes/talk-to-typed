# Benchmarks & Exploratory Diagnostic Scripts

This directory contains standalone diagnostics, model release checks, and empirical speech recovery benchmarks used during development:

* **`comprehensive_test.py` / `test_stt_comprehensive.py`**: Full end-to-end integration tests across all models and VAD configurations.
* **`test_real_speech.py` / `test_real_speech2.py` / `test_better.py`**: Benchmarks evaluating transcription accuracy against challenging real-world recordings.
* **`check_releases.py` / `check_v201.py` / `check_v201_assets.py`**: Upstream `sherpa-onnx` release asset queries and version verification.
* **`find_models.py` / `find_models2.py` / `download_nllb.py`**: Model discovery and download helpers for offline ASR, denoisers, and CTranslate2 NLLB translation.
* **`generate_audio.py`**: Generates synthetic acoustic test patterns (rumble, hum, whispers, clicks).
