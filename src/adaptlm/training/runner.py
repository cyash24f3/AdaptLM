"""Offline PEFT/TRL LoRA. MPS and CUDA are explicit; no paid services."""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import psutil

from adaptlm.artifacts.manifest import digest, environment, fingerprint, write_json
from adaptlm.config import Settings
from adaptlm.contracts.schema import SCHEMA_VERSION
from adaptlm.data.pipeline import read_rows, validate
from adaptlm.inference.prompts import PROMPT_VERSION, encode_training, messages, prompt_hash


def validate_resume_manifest(previous, current):
    keys = (
        "config",
        "model_id",
        "base_revision",
        "tokenizer_revision",
        "dataset_hash",
        "train_example_ids",
        "validation_example_ids",
        "prompt_hash",
        "chat_template_hash",
        "packages",
    )
    mismatches = [key for key in keys if key not in previous or previous[key] != current[key]]
    if mismatches:
        raise ValueError("resume manifest mismatch: " + ", ".join(mismatches))


def train(config_path: Path, output: Path, resume: Path | None = None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    config = json.loads(config_path.read_text())
    settings = Settings(
        profile="local",
        device=config["device"],
        dtype=config["dtype"],
        attention_implementation=config.get("attention_implementation", "eager"),
    )
    validate(settings.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "bundle.json").exists():
        raise ValueError("immutable exported bundle already exists; use a new run path")
    if resume and (
        not resume.resolve().is_relative_to(output.resolve())
        or not (resume / "trainer_state.json").exists()
    ):
        raise ValueError("resume only from this run's locally produced Trainer checkpoint")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if settings.device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS unavailable; device fallback is forbidden")
        torch.mps.manual_seed(seed)
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; device fallback is forbidden")
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_id, revision=settings.revision, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    train_rows = read_rows(settings.data_dir / "train.jsonl")
    if config.get("limit_examples"):
        rng = random.Random(seed)
        rng.shuffle(train_rows)
        train_rows = train_rows[: config["limit_examples"]]
    validation_rows = read_rows(settings.data_dir / "validation.jsonl")
    # Fixed representative validation variants, selected without test access.
    validation_rows = [r for r in validation_rows if r["provenance"]["variant"] == 0]
    encoded = [encode_training(r, tokenizer, config["max_length"]) for r in train_rows]
    eval_encoded = [encode_training(r, tokenizer, config["max_length"]) for r in validation_rows]
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id, completion_only_loss=True, pad_to_multiple_of=64
    )
    batch = collator([encoded[0]])
    mask = encoded[0]["completion_mask"]
    labels = batch["labels"][0].tolist()
    assert all(label == -100 for label, m in zip(labels, mask) if not m)
    assert all(
        label == token for label, token, m in zip(labels, encoded[0]["input_ids"], mask) if m
    )
    learned = [label for label in labels if label != -100]
    assert learned and tokenizer.eos_token_id in learned
    write_json(
        output / "loss-mask-diagnostic.json",
        {
            "example_id": train_rows[0]["id"],
            "prompt_tokens": mask.count(0),
            "learned_tokens": mask.count(1),
            "prompt_masked": True,
            "stop_token_learned": True,
            "decoded_prompt": tokenizer.decode(encoded[0]["input_ids"][: mask.count(0)]),
            "decoded_learned_target": tokenizer.decode(learned),
            "lengths": {
                "min": min(len(r["input_ids"]) for r in encoded),
                "max": max(len(r["input_ids"]) for r in encoded),
            },
            "truncated_sequences": 0,
        },
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        dtype=getattr(torch, settings.dtype),
        use_safetensors=True,
        trust_remote_code=False,
        attn_implementation=settings.attention_implementation,
    ).to(settings.device)
    model.config.use_cache = False
    targets = config["target_modules"]
    found = {name.split(".")[-1] for name, _ in model.named_modules()}
    if set(targets) - found:
        raise ValueError("LoRA targets not found in architecture")
    lora = LoraConfig(
        r=config["rank"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=targets,
        task_type="CAUSAL_LM",
        base_model_name_or_path=settings.model_id,
        revision=settings.revision,
    )
    model = get_peft_model(model, lora)
    initial = {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad
    }
    frozen_name, frozen_parameter = next(
        (n, p) for n, p in model.named_parameters() if not p.requires_grad
    )
    frozen_sample = frozen_parameter.detach().flatten()[:128].cpu().clone()
    diagnostics = {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "nonzero_gradient_modules": set(),
        "frozen_parameter_gradients": False,
        "peak_device_allocated_bytes": 0,
    }

    class GradientAudit(TrainerCallback):
        def on_substep_end(self, args, state, control, **kwargs):
            if settings.device == "mps" and torch.mps.driver_allocated_memory() > 10 * 1024**3:
                torch.mps.synchronize()
                torch.mps.empty_cache()

        def on_step_end(self, args, state, control, **kwargs):
            if settings.device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            write_json(
                output / "progress.json",
                {
                    "optimizer_step": state.global_step,
                    "epoch": state.epoch,
                    "log_history": state.log_history,
                    "process_rss_bytes": psutil.Process().memory_info().rss,
                },
            )

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            for name, p in kwargs["model"].named_parameters():
                if state.global_step >= 2:
                    break
                if p.requires_grad and p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise ValueError("nonfinite adapter gradient")
                    if torch.count_nonzero(p.grad):
                        diagnostics["nonzero_gradient_modules"].add(name)
                elif not p.requires_grad and p.grad is not None:
                    diagnostics["frozen_parameter_gradients"] = True
            if settings.device == "mps":
                allocated = torch.mps.current_allocated_memory()
            elif settings.device == "cuda":
                allocated = torch.cuda.max_memory_allocated()
            else:
                allocated = 0
            diagnostics["peak_device_allocated_bytes"] = max(
                diagnostics["peak_device_allocated_bytes"], allocated
            )

    train_args = SFTConfig(
        output_dir=str(output),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"],
        max_steps=config.get("max_steps", -1),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        lr_scheduler_type="linear",
        seed=seed,
        data_seed=seed,
        optim="adamw_torch",
        report_to="none",
        max_length=config["max_length"],
        packing=False,
        completion_only_loss=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
        logging_steps=1,
        save_steps=config["save_steps"],
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=config["save_steps"],
        load_best_model_at_end=config["kind"] != "mechanics-smoke-not-quality",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        disable_tqdm=True,
        use_cpu=settings.device == "cpu",
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=Dataset.from_list(encoded),
        eval_dataset=Dataset.from_list(eval_encoded),
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[GradientAudit()],
    )
    if trainer.args.device.type != settings.device:
        raise ValueError(f"Trainer selected {trainer.args.device}; requested {settings.device}")
    metadata = environment() | {
        "config": config,
        "model_id": settings.model_id,
        "base_revision": settings.revision,
        "tokenizer_revision": settings.revision,
        "dataset_hash": json.loads((settings.data_dir / "manifest.json").read_text())[
            "dataset_hash"
        ],
        "train_example_ids": [r["id"] for r in train_rows],
        "validation_example_ids": [r["id"] for r in validation_rows],
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "chat_template_hash": fingerprint(tokenizer.chat_template),
        "checkpoint_selection": "last smoke checkpoint; no quality selection"
        if config["kind"] == "mechanics-smoke-not-quality"
        else "lowest validation completion cross entropy at saved checkpoints",
        "precision": "BF16 weights; FP32 LoRA parameters; no autocast",
        "attention_implementation": settings.attention_implementation,
        "nondeterminism": "MPS kernels may be nondeterministic; seeds do not imply bitwise reproduction",
        "resume_from": str(resume) if resume else None,
        "status": "running",
    }
    if resume:
        previous_path = output / "run.json"
        if not previous_path.is_file():
            raise ValueError("resume requires this run's original manifest")
        previous = json.loads(previous_path.read_text())
        validate_resume_manifest(previous, metadata)
        write_json(output / "pre-resume-run.json", previous)
    write_json(output / "run.json", metadata)
    started = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        evaluation = trainer.evaluate()
        for loss in (result.training_loss, evaluation["eval_loss"]):
            if not math.isfinite(loss):
                raise ValueError("nonfinite training/validation loss")
        changes = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and not torch.equal(initial[name], p.detach().cpu())
        ]
        diagnostics["nonzero_gradient_modules"] = sorted(diagnostics["nonzero_gradient_modules"])
        diagnostics["changed_adapter_parameters"] = changes
        diagnostics["frozen_sample_unchanged"] = torch.equal(
            frozen_sample, frozen_parameter.detach().flatten()[:128].cpu()
        )
        assert changes and diagnostics["nonzero_gradient_modules"]
        assert (
            not diagnostics["frozen_parameter_gradients"] and diagnostics["frozen_sample_unchanged"]
        )
        model.config.use_cache = True
        model.eval()
        check = tokenizer.apply_chat_template(
            messages(train_rows[0]["message"]),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(settings.device)
        with torch.inference_mode():
            before = (
                model.generate(
                    check,
                    attention_mask=torch.ones_like(check),
                    max_new_tokens=24,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )[0]
                .cpu()
                .tolist()
            )
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        # Reload the exported adapter on the same base weights; no second 3GB base allocation.
        base = model.unload()
        if hasattr(base, "peft_config"):
            delattr(base, "peft_config")
        reloaded = PeftModel.from_pretrained(base, output, is_trainable=False)
        reloaded.eval()
        with torch.inference_mode():
            after = (
                reloaded.generate(
                    check, max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.pad_token_id
                )[0]
                .cpu()
                .tolist()
            )
        diagnostics["export_reload_token_parity"] = before == after
        assert before == after
        metadata.update(
            status="completed",
            duration_seconds=time.perf_counter() - started,
            training_metrics=result.metrics,
            validation_metrics=evaluation,
            log_history=trainer.state.log_history,
            best_checkpoint=trainer.state.best_model_checkpoint,
            optimizer_steps=trainer.state.global_step,
            actual_epochs=trainer.state.epoch,
            unique_training_examples=len(train_rows),
            available_training_tokens=sum(len(r["input_ids"]) for r in encoded),
            process_rss_bytes=psutil.Process().memory_info().rss,
            diagnostics=diagnostics,
        )
        write_json(output / "run.json", metadata)
        write_json(output / "gradient-reload-diagnostic.json", diagnostics)
        artifacts = {
            p.name: digest(p)
            for p in output.iterdir()
            if p.is_file()
            and p.name
            in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer_config.json",
                "tokenizer.json",
            )
        }
        bundle = {
            "base_model_id": settings.model_id,
            "base_revision": settings.revision,
            "tokenizer_revision": settings.revision,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash(),
            "chat_template_hash": fingerprint(tokenizer.chat_template),
            "architecture": "Qwen2ForCausalLM",
            "dataset_hash": metadata["dataset_hash"],
            "artifact_hashes": artifacts,
            "run_manifest_hash": digest(output / "run.json"),
            "runtime_packages": metadata["packages"],
            "license": "Apache-2.0",
            "experiment_kind": config["kind"],
        }
        bundle["bundle_id"] = "adaptlm-" + fingerprint(bundle)[:16]
        write_json(output / "bundle.json", bundle)
        return metadata
    except BaseException as exc:
        metadata.update(
            status="failed",
            error_type=type(exc).__name__,
            duration_seconds=time.perf_counter() - started,
        )
        write_json(output / "run.json", metadata)
        raise
