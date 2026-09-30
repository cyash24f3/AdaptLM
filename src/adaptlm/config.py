import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ADAPTLM_", env_file=".env", extra="ignore")
    profile: Literal["fixture", "local"] = "fixture"
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    attention_implementation: Literal["eager", "sdpa"] = "eager"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str = ""
    adapter_path: Path | None = None
    data_dir: Path = Path("datasets")
    report_dir: Path = Path("reports")
    admin_token: str = ""
    max_chars: int = Field(default=3000, ge=32, le=20000)
    max_input_tokens: int = Field(default=4096, ge=64, le=16384)
    max_new_tokens: int = Field(default=384, ge=16, le=1024)
    max_output_chars: int = Field(default=30000, ge=1000, le=100000)
    queue_capacity: int = Field(default=2, ge=0, le=8)
    queue_timeout_seconds: float = Field(default=60, gt=0, le=600)
    offline: bool = False
    sample_memory: bool = False

    @model_validator(mode="after")
    def pinned(self):
        if not self.revision and Path("configs/model.json").exists():
            manifest = json.loads(Path("configs/model.json").read_text())
            if manifest["model_id"] == self.model_id:
                self.revision = manifest["revision"]
        if self.profile == "local" and not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("local profile requires an immutable 40-character model revision")
        return self
