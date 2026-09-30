import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from adaptlm.contracts.schema import SCHEMA_VERSION, SPEC_VERSION


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def environment():
    packages = {}
    for name in ("adaptlm", "torch", "transformers", "peft", "trl", "datasets", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": result.stdout.strip()
        if result.returncode == 0
        else "uncommitted-new-repository",
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True
            ).stdout.strip()
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": packages,
        "lock_hash": digest(Path("uv.lock")) if Path("uv.lock").exists() else None,
        "schema_version": SCHEMA_VERSION,
        "annotation_spec_version": SPEC_VERSION,
        "source_hashes": {
            str(p.relative_to(Path(__file__).parents[1])): digest(p)
            for p in sorted(Path(__file__).parents[1].rglob("*.py"))
        },
        "monetary_cost": {"paid_services_used": False, "electricity_cost": "unmeasured"},
    }


def validate_bundle(path: Path, model_id: str, revision: str, template_hash: str):
    from adaptlm.inference.prompts import LEGACY_PROMPT_HASHES, PROMPT_VERSION, prompt_hash

    data = json.loads((path / "bundle.json").read_text())
    expected = {
        "base_model_id": model_id,
        "base_revision": revision,
        "tokenizer_revision": revision,
        "schema_version": SCHEMA_VERSION,
        "chat_template_hash": template_hash,
        "architecture": "Qwen2ForCausalLM",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"incompatible adapter bundle: {key}")
    known_prompts = LEGACY_PROMPT_HASHES | {PROMPT_VERSION: prompt_hash()}
    if data.get("prompt_version") not in known_prompts or known_prompts[
        data["prompt_version"]
    ] != data.get("prompt_hash"):
        raise ValueError("unknown or altered recorded training prompt")
    for name, sha in data["artifact_hashes"].items():
        file = path / name
        if file.parent != path or digest(file) != sha:
            raise ValueError("adapter artifact integrity check failed")
    if not (path / "adapter_model.safetensors").is_file():
        raise ValueError("missing safe adapter weights")
    if digest(path / "run.json") != data["run_manifest_hash"]:
        raise ValueError("training manifest integrity check failed")
    for package in ("torch", "transformers", "peft"):
        if version(package).split("+")[0] != data["runtime_packages"][package].split("+")[0]:
            raise ValueError(f"runtime version mismatch for {package}; revalidate before loading")
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != model_id or config.get("revision") != revision:
        raise ValueError("PEFT adapter base/revision mismatch")
    return data
