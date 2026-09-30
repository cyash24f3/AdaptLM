import json
import os
import shutil
from pathlib import Path

import pytest

from adaptlm.artifacts.manifest import digest, fingerprint, validate_bundle
from adaptlm.data.pipeline import read_rows, validate
from adaptlm.evaluation.metrics import aggregate, paired_bootstrap, per_example
from adaptlm.inference.prompts import demonstrations, encode_training, messages
from adaptlm.training.runner import validate_resume_manifest


def test_frozen_hashes_and_grouping():
    report = validate(Path("datasets"))
    assert report["examples"] == 1280
    assert report["families"] == 160
    assert report["cross_split_near_duplicates"] == []
    assert report["human_reviewed"] == 0


def test_hash_tampering_fails(tmp_path):
    root = tmp_path / "data"
    shutil.copytree("datasets", root)
    with (root / "test.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="hash changed"):
        validate(root)


def test_exact_duplicate_rejected_even_with_updated_hash(tmp_path):
    root = tmp_path / "data"
    shutil.copytree("datasets", root)
    path = root / "train.jsonl"
    with path.open("a") as f:
        f.write(path.read_text().splitlines()[0] + "\n")
    m = json.loads((root / "manifest.json").read_text())
    m["split_hashes"]["train"] = digest(path)
    m["dataset_hash"] = fingerprint(m["split_hashes"])
    (root / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="duplicate"):
        validate(root)


def test_classifier_train_only_fit():
    report = json.loads(Path("reports/classifier.json").read_text())
    assert set(report["fit_ids"]) <= {r["id"] for r in read_rows(Path("datasets/train.jsonl"))}
    assert report["fit_split"].startswith("train only")


def test_few_shot_membership_and_format_stability():
    demos = demonstrations(Path("datasets"))
    assert all(r["split"] == "train" for r in demos)
    assert messages("hello", demos) == messages("hello", demos)
    bad = dict(demos[0], split="test")
    with pytest.raises(ValueError):
        messages("hi", [bad])


def test_invalid_outputs_count_in_every_applicable_denominator():
    row = read_rows(Path("datasets/train.jsonl"))[0]
    invalid = {
        "output": None,
        "final_valid": False,
        "raw_valid": False,
        "status": "invalid_output",
        "timings": {},
        "usage": None,
        "repair_count": 0,
    }
    result = aggregate([per_example(row, invalid)])
    assert result["attempted"] == 1
    assert result["category_accuracy_all_attempted"] == 0
    assert result["category_accuracy_valid_only"] is None
    assert result["complete_record_reference_success"] == 0
    assert result["summary_claims"]["semantic_supported"] is None


def test_bootstrap_is_paired_and_family_aware():
    a = [
        {"id": str(i), "family_id": str(i // 2), "complete_record_reference_success": False}
        for i in range(20)
    ]
    b = [dict(row, complete_record_reference_success=True) for row in a]
    result = paired_bootstrap(a, b)
    assert result["independent_families"] == 10
    assert result["95_percent_family_bootstrap_interval"] == [1.0, 1.0]
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[1:])


class FakeTokenizer:
    def apply_chat_template(self, chat, tokenize, add_generation_prompt):
        return [1, 2] if add_generation_prompt else [1, 2, 3, 4]


def test_no_silent_target_truncation_and_loss_mask():
    row = read_rows(Path("datasets/train.jsonl"))[0]
    result = encode_training(row, FakeTokenizer(), 8)
    assert result["completion_mask"] == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="no truncation"):
        encode_training(row, FakeTokenizer(), 3)


def test_bundle_base_mismatch(tmp_path):
    (tmp_path / "bundle.json").write_text(json.dumps({"base_model_id": "different"}))
    with pytest.raises(ValueError, match="base_model_id"):
        validate_bundle(tmp_path, "Qwen/Qwen2.5-1.5B-Instruct", "a" * 40, "template")


@pytest.mark.parametrize("field", ["config", "dataset_hash", "prompt_hash", "packages"])
def test_resume_rejects_changed_experiment(field):
    current = {
        "config": {"learning_rate": 0.0002},
        "model_id": "pinned-model",
        "base_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "dataset_hash": "frozen-data",
        "train_example_ids": ["train-1"],
        "validation_example_ids": ["validation-1"],
        "prompt_hash": "frozen-prompt",
        "chat_template_hash": "frozen-template",
        "packages": {"torch": "2.14.0"},
    }
    validate_resume_manifest(current, current)
    with pytest.raises(ValueError, match=field):
        validate_resume_manifest(current | {field: "changed"}, current)
    with pytest.raises(ValueError, match=field):
        validate_resume_manifest(
            {key: value for key, value in current.items() if key != field}, current
        )


@pytest.mark.parametrize(
    "field",
    ["base_revision", "tokenizer_revision", "schema_version", "chat_template_hash", "architecture"],
)
def test_bundle_contract_mismatch(tmp_path, field):
    data = {
        "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "base_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "schema_version": "triage-1.0",
        "chat_template_hash": "template",
        "architecture": "Qwen2ForCausalLM",
    }
    data[field] = "incompatible"
    (tmp_path / "bundle.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=field):
        validate_bundle(tmp_path, "Qwen/Qwen2.5-1.5B-Instruct", "a" * 40, "template")


@pytest.mark.integration
def test_actual_smoke_gradient_mask_and_reload_evidence():
    root = Path(os.environ.get("ADAPTLM_SMOKE_RUN", "runs/smoke-v2-214"))
    run = json.loads((root / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["diagnostics"]["export_reload_token_parity"]
    assert run["diagnostics"]["changed_adapter_parameters"]
    assert run["diagnostics"]["nonzero_gradient_modules"]
    assert run["diagnostics"]["frozen_sample_unchanged"]
    diagnostic = json.loads((root / "loss-mask-diagnostic.json").read_text())
    assert diagnostic["prompt_masked"] and diagnostic["stop_token_learned"]
