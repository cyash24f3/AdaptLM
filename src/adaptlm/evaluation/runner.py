import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil

from adaptlm.artifacts.manifest import environment, fingerprint, write_json
from adaptlm.config import Settings
from adaptlm.data.pipeline import read_rows, validate
from adaptlm.evaluation.metrics import aggregate, paired_bootstrap, per_example
from adaptlm.inference.service import InferenceService


async def evaluate(
    settings: Settings,
    split: str,
    modes: list[str],
    output: Path,
    variant: int | None = None,
    limit: int | None = None,
):
    if settings.profile != "local":
        raise ValueError("fixture output cannot be used for model quality evaluation")
    validation = validate(settings.data_dir)
    rows = read_rows(settings.data_dir / f"{split}.jsonl")
    if variant is not None:
        rows = [r for r in rows if r["provenance"]["variant"] == variant]
    if limit:
        # Round robin categories: a prefix must not select only alphabetically early categories.
        groups = defaultdict(list)
        for row in rows:
            groups[row["slices"][0]].append(row)
        ordered = []
        while any(groups.values()):
            for key in sorted(groups):
                if groups[key]:
                    ordered.append(groups[key].pop(0))
        rows = ordered[:limit]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists() or (output / "predictions.jsonl").exists():
        raise ValueError("preserve prior experiment; choose a new output directory")
    frozen = {
        "split": split,
        "example_ids": [r["id"] for r in rows],
        "variant": variant,
        "dataset_hash": validation["dataset_hash"],
        "modes": modes,
        "decoding": {"do_sample": False, "max_new_tokens": settings.max_new_tokens},
        "selection": "deterministic category round robin; no output-informed sample selection",
    }
    frozen["configuration_hash"] = fingerprint(frozen)
    write_json(output / "frozen-config.json", frozen)
    if split == "test":
        exposure = settings.data_dir / "test-exposures.jsonl"
        with exposure.open("a") as file:
            file.write(json.dumps(environment() | frozen | {"output": str(output)}) + "\n")
    service = InferenceService(settings)
    write_json(output / "runtime.json", service.engine.metadata())
    results = {mode: [] for mode in modes}
    started = time.perf_counter()
    with (output / "predictions.jsonl").open("w") as file:
        for mode in modes:
            for row in rows:
                response = await service.triage(row["message"], mode, include_raw=True)
                result = per_example(row, response)
                results[mode].append(result)
                file.write(json.dumps({"mode": mode} | result, ensure_ascii=False) + "\n")
                file.flush()
                print(f"{mode}: {len(results[mode])}/{len(rows)} {response['status']}", flush=True)
    comparisons = {}
    if "zero_shot" in results:
        for mode in modes:
            if mode != "zero_shot":
                comparisons[f"{mode}_minus_zero_shot"] = {
                    key: paired_bootstrap(
                        [
                            r
                            for r in results["zero_shot"]
                            if key != "category_correct" or r["category_applicable"]
                        ],
                        [
                            r
                            for r in results[mode]
                            if key != "category_correct" or r["category_applicable"]
                        ],
                        key=key,
                    )
                    for key in (
                        "raw_valid",
                        "category_correct",
                        "status_correct",
                        "complete_record_reference_success",
                    )
                }
    report = (
        environment()
        | frozen
        | {
            "runtime": service.engine.metadata(),
            "duration_seconds": time.perf_counter() - started,
            "interpretation": "agreement with unreviewed implementation-AI-authored labels; semantic review pending",
            "complete_success_definition": "final_valid AND exact category/status AND entity set AND missing set AND reference summary spans/text",
            "metrics": {mode: aggregate(items) for mode, items in results.items()},
            "paired_comparisons": comparisons,
            "slices": {
                mode: {
                    slice_: aggregate([r for r in items if slice_ in r["slices"]])
                    for slice_ in sorted({s for r in items for s in r["slices"]})
                }
                for mode, items in results.items()
            },
            "failures": {
                mode: [r for r in items if not r["complete_record_reference_success"]][:20]
                for mode, items in results.items()
            },
            "recommendation": "Review labels and failure spans independently before expanding training or making real-world accuracy claims.",
        }
    )
    write_json(output / "report.json", report)
    return report


async def benchmark(settings, output, repeats=5, concurrency=1):
    if output.exists():
        raise ValueError("preserve prior serving benchmark; choose a new output")
    if repeats < 1 or repeats > 100:
        raise ValueError("benchmark repeats must be between one and 100")
    if concurrency < 1 or concurrency > 8:
        raise ValueError("benchmark concurrency must be between one and eight")
    settings = settings.model_copy(update={"sample_memory": True})
    host = {
        "total_ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes": psutil.virtual_memory().available,
        "swap_used_bytes": psutil.swap_memory().used,
        "logical_cpus": psutil.cpu_count(),
        "other_applications": "uncontrolled; host is not a dedicated benchmark machine",
    }
    service = InferenceService(settings)
    metadata = service.engine.metadata()
    texts = (
        metadata.get("demo_messages")
        or [
            r["message"]
            for r in read_rows(settings.data_dir / "validation.jsonl")
            if r["provenance"]["variant"] == 0
        ][:3]
    )
    results = {}
    for mode in metadata["available_modes"]:
        await service.triage(texts[0], mode)  # explicitly excluded warmup
        items = []
        start = time.perf_counter()
        for i in range(0, repeats, concurrency):
            batch = await asyncio.gather(
                *(
                    service.triage(texts[j % len(texts)], mode)
                    for j in range(i, min(i + concurrency, repeats))
                )
            )
            items.extend(batch)
        elapsed = time.perf_counter() - start
        timed = [r for r in items if r.get("timings", {}).get("total_seconds") is not None]
        latency = [r["timings"]["total_seconds"] for r in timed]
        tokens = sum((r["usage"] or {}).get("output_tokens", 0) for r in items)
        generation = sum(r["timings"].get("generation_seconds", 0) for r in items)
        results[mode] = {
            "requests": repeats,
            "timed_requests": len(timed),
            "transport_status_counts": dict(Counter(r["status"] for r in items)),
            "concurrency": concurrency,
            "warm": True,
            "warmup_requests_excluded": 1,
            "p50_seconds": float(np.quantile(latency, 0.5)) if latency else None,
            "p95_seconds": float(np.quantile(latency, 0.95)) if latency else None,
            "generated_tokens_per_generation_second": tokens / generation if generation else None,
            "completed_requests_per_wall_second": len(timed) / elapsed,
            "process_rss_bytes_max": max((r.get("process_rss_bytes", 0) for r in items), default=0),
            "accelerator_allocated_bytes_max": max(
                (r.get("accelerator_allocated_bytes") or 0 for r in items), default=0
            ),
            "memory_limitation": "sampled allocation at request completion, not peak total device memory",
            "memory_probes": [r["memory_probe"] for r in items if r.get("memory_probe")],
            "raw_validity": sum(r["raw_valid"] for r in items) / repeats,
            "final_validity": sum(r["final_valid"] for r in items) / repeats,
            "requests_detail": items,
        }
    report = environment() | {
        "runtime": metadata,
        "results": results,
        "semantic_quality": "not measured by a serving benchmark",
        "fixed_input_hash": fingerprint(texts),
        "fixed_inputs": texts,
        "decoding": {"do_sample": False, "max_new_tokens": settings.max_new_tokens},
        "admission": {
            "active_generations": 1,
            "waiting_capacity": settings.queue_capacity,
            "waiting_timeout_seconds": settings.queue_timeout_seconds,
        },
        "host_at_start": host,
        "memory_sampling": "20ms sampled RSS/Metal; CUDA allocator counter; includes probe overhead, excludes model-load peak",
    }
    write_json(output, report)
    return report
