import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

models_dir = Path(r'C:\Users\o\AppData\Local\stt\models')
models_dir.mkdir(parents=True, exist_ok=True)

# Download NLLB-200 distilled 600M CTranslate2 model
repo_id = "entai2965/nllb-200-distilled-600M-ctranslate2"
local_dir = models_dir / "nllb-200-distilled-600M"

print(f"Downloading {repo_id} to {local_dir}")
try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("Download complete!")
    
    # List files
    for f in sorted(local_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(local_dir)}: {f.stat().st_size} bytes")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()