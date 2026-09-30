"""Pin the official checkpoint, verify its license, download safetensors only."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
manifest_path = Path("configs/model.json")
if manifest_path.exists():
    revision = json.loads(manifest_path.read_text())["revision"]
else:
    revision = HfApi().model_info(model_id).sha
license_path = Path(hf_hub_download(model_id, "LICENSE", revision=revision))
license_text = license_path.read_text()
if "Apache License" not in license_text or "Version 2.0" not in license_text:
    raise RuntimeError("official checkpoint license not verified")
manifest = {
    "model_id": model_id,
    "revision": revision,
    "tokenizer_revision": revision,
    "license": "Apache-2.0",
    "license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
    "architecture": "Qwen2ForCausalLM",
    "parameter_count_model_card": 1540000000,
    "context_length": 32768,
    "intended_use": "fictional support message triage",
    "download_estimate_bytes": 3100000000,
    "source": f"https://huggingface.co/{model_id}/tree/{revision}",
}
manifest_path.parent.mkdir(exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest), flush=True)
local = snapshot_download(
    model_id,
    revision=revision,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "LICENSE", "README.md"],
)
print(f"Verified snapshot cached at {local}", flush=True)
