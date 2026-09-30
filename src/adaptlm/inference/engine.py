import contextlib
import json
import time

import psutil

from adaptlm.artifacts.manifest import fingerprint, validate_bundle
from adaptlm.config import Settings
from adaptlm.contracts.schema import SCHEMA_VERSION, TriageRecord
from adaptlm.data.pipeline import read_rows
from adaptlm.inference.prompts import PROMPT_VERSION, demonstrations, messages, prompt_hash


def parse_output(raw: str, message: str, max_chars: int = 30000):
    """One deterministic fence-only repair. No label/category/span correction."""
    if len(raw) > max_chars:
        return None, False, 0, "output exceeds byte/character budget"
    try:
        record = TriageRecord.model_validate_json(raw).validate_grounding(message)
        return record, True, 0, None
    except ValueError:
        pass
    repaired = raw.strip()
    attempts = 0
    if repaired.startswith("```json\n") and repaired.endswith("```"):
        repaired = repaired[8:-3].strip()
        attempts = 1
    elif repaired.startswith("```\n") and repaired.endswith("```"):
        repaired = repaired[4:-3].strip()
        attempts = 1
    if attempts:
        try:
            return (
                TriageRecord.model_validate_json(repaired).validate_grounding(message),
                False,
                1,
                None,
            )
        except ValueError:
            pass
    return None, False, attempts, "output failed schema or exact source-span validation"


class FixtureEngine:
    profile = "fixture"
    ready = True

    def __init__(self, settings: Settings):
        self.settings = settings
        rows = read_rows(settings.data_dir / "train.jsonl")
        self.samples = [
            next(
                r
                for r in rows
                if r["family_id"].startswith(c + "-") and r["provenance"]["variant"] == 0
            )
            for c in ("duplicate_charge", "delivery_problem", "account_access")
        ]
        self.by_message = {r["message"]: r["target"] for r in self.samples}

    def metadata(self):
        return {
            "profile": "fixture",
            "ready": True,
            "base_model_id": None,
            "base_revision": None,
            "adapter_bundle_id": None,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "warning": "Deterministic fixture responses; no model inference or quality claim.",
            "available_modes": ["fixture"],
            "demo_messages": list(self.by_message),
        }

    def generate(self, message, mode):
        if mode != "fixture":
            raise LookupError("model mode unavailable in fixture profile")
        if message not in self.by_message:
            raise LookupError("fixture recognizes only the documented demo messages")
        return {
            "raw": json.dumps(self.by_message[message], ensure_ascii=False),
            "usage": None,
            "timings": {"tokenization_seconds": 0, "generation_seconds": 0},
            "truncated": False,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "accelerator_allocated_bytes": None,
        }


class TransformersEngine:
    profile = "local"

    def __init__(self, settings: Settings, *, revalidate_torch_version: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.settings = settings
        self.torch = torch
        if settings.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            trust_remote_code=False,
            local_files_only=settings.offline,
        )
        self.template_hash = fingerprint(self.tokenizer.chat_template)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            dtype=getattr(torch, settings.dtype),
            trust_remote_code=False,
            use_safetensors=True,
            local_files_only=settings.offline,
            attn_implementation=settings.attention_implementation,
        ).to(settings.device)
        self.bundle = None
        if settings.adapter_path:
            from peft import PeftModel

            self.bundle = validate_bundle(
                settings.adapter_path,
                settings.model_id,
                settings.revision,
                self.template_hash,
                revalidate_torch_version=revalidate_torch_version,
            )
            self.model = PeftModel.from_pretrained(
                self.model, settings.adapter_path, is_trainable=False
            )
        self.model.eval()
        self.demos = demonstrations(settings.data_dir)
        self.load_seconds = time.perf_counter() - started
        self.ready = revalidate_torch_version is None
        self.runtime_validation = None

    def revalidate_runtime(self):
        """Explicit on-device smoke for a deployment runtime; no task-quality claim."""
        if self.ready:
            return
        from adaptlm.artifacts.manifest import environment, write_json

        row = read_rows(self.settings.data_dir / "train.jsonl")[0]
        ids = self.tokenizer.apply_chat_template(
            messages(row["message"]), tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.settings.device)
        with self.torch.inference_mode():
            adapted = self.model(ids).logits[:, -1].float()
            with self.model.disable_adapter():
                base = self.model(ids).logits[:, -1].float()
            if not self.torch.isfinite(adapted).all() or not self.torch.isfinite(base).all():
                raise ValueError("candidate runtime produced nonfinite logits")
            difference = (adapted - base).abs().max().item()
            if difference == 0:
                raise ValueError("adapter did not change candidate runtime logits")
            options = {
                "max_new_tokens": 24,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            a = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), **options)
            b = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), **options)
            if not self.torch.equal(a, b):
                raise ValueError("candidate runtime greedy prefix is not repeatable")
        self.runtime_validation = environment() | {
            "kind": "candidate-runtime finite-logit and repeated greedy prefix smoke; not task quality or cross-device parity",
            "original_bundle_id": self.bundle["bundle_id"],
            "training_runtime_packages": self.bundle["runtime_packages"],
            "candidate_device": self.settings.device,
            "candidate_dtype": self.settings.dtype,
            "example_id": row["id"],
            "finite_base_and_adapter_logits": True,
            "maximum_adapter_logit_difference": difference,
            "repeatable_generated_prefix": True,
            "generated_prefix_ids": a[0, ids.shape[1] :].tolist(),
            "original_training_manifest_unchanged": True,
        }
        write_json(
            self.settings.report_dir / "cloud-runtime-validation.json", self.runtime_validation
        )
        self.ready = True

    def metadata(self):
        return {
            "profile": "local",
            "ready": self.ready,
            "base_model_id": self.settings.model_id,
            "base_revision": self.settings.revision,
            "tokenizer_revision": self.settings.revision,
            "adapter_bundle_id": self.bundle["bundle_id"] if self.bundle else None,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash(),
            "training_prompt_version": self.bundle["prompt_version"] if self.bundle else None,
            "chat_template_hash": self.template_hash,
            "device": self.settings.device,
            "dtype": self.settings.dtype,
            "quantization": None,
            "attention_implementation": self.settings.attention_implementation,
            "load_seconds": self.load_seconds,
            "runtime_validation": self.runtime_validation,
            "available_modes": ["zero_shot", "few_shot"] + (["adapted"] if self.bundle else []),
            "demonstration_ids": [d["id"] for d in self.demos],
            "few_shot_selection": "coverage-v2: shortest train examples within filled-entity, missing, ambiguity, out-of-scope roles",
            "demo_messages": [],
        }

    def generate(self, message, mode):
        if not self.ready:
            raise ValueError("candidate deployment runtime requires on-device revalidation")
        if mode not in self.metadata()["available_modes"]:
            raise LookupError(f"{mode} unavailable: compatible trained adapter is required")
        started = time.perf_counter()
        chat = messages(message, self.demos if mode == "few_shot" else None)
        ids = self.tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.settings.device)
        if ids.shape[1] > self.settings.max_input_tokens:
            raise ValueError("tokenized prompt exceeds configured input budget; not truncated")
        tokenization = time.perf_counter() - started
        switch = (
            self.model.disable_adapter()
            if self.bundle and mode != "adapted"
            else contextlib.nullcontext()
        )
        started = time.perf_counter()
        with switch, self.torch.inference_mode():
            output = self.model.generate(
                input_ids=ids,
                attention_mask=self.torch.ones_like(ids),
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        if self.settings.device == "mps":
            self.torch.mps.synchronize()
        elif self.settings.device == "cuda":
            self.torch.cuda.synchronize()
        generation = time.perf_counter() - started
        new_ids = output[0, ids.shape[1] :].tolist()
        device_memory = None
        if self.settings.device == "mps":
            device_memory = self.torch.mps.current_allocated_memory()
        elif self.settings.device == "cuda":
            device_memory = self.torch.cuda.memory_allocated()
        return {
            "raw": self.tokenizer.decode(new_ids, skip_special_tokens=True),
            "usage": {"input_tokens": ids.shape[1], "output_tokens": len(new_ids)},
            "truncated": len(new_ids) >= self.settings.max_new_tokens
            and new_ids[-1] != self.tokenizer.eos_token_id,
            "timings": {"tokenization_seconds": tokenization, "generation_seconds": generation},
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "accelerator_allocated_bytes": device_memory,
        }
