#!/bin/sh
# Download the models stt uses into ~/.local/share/stt/models (idempotent; deletes tarballs after extracting).
# Asset names come from the k2-fsa/sherpa-onnx GitHub releases (tags: asr-models, speech-enhancement-models).
set -e
D="${STT_MODELS:-$HOME/.local/share/stt/models}"
ASR=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
ENH=https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models
mkdir -p "$D" && cd "$D"
tarball() { [ -d "${1%.tar.bz2}" ] || { wget -q "$ASR/$1" && tar xjf "$1" && rm "$1"; echo "ok $1"; }; }
file()    { [ -f "$2" ] || { wget -q "$1/$2" && echo "ok $2"; }; }
tarball sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2                         # offline (stt file, --whisper, --distant)
tarball sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2 # streaming (stt mic)
file "$ASR" silero_vad.onnx                                                         # VAD
file "$ENH" dpdfnet8.onnx                                                           # default denoiser for -a 2..5
if [ "$1" = "--all" ]; then                                                         # optional extras
  tarball sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2               # fast streaming; check its license
  for f in dpdfnet2.onnx dpdfnet4.onnx gtcrn_simple.onnx; do file "$ENH" $f; done
fi
