# Native deployment and rollback

Bind native serving to 127.0.0.1 by default. Ports 8765 (fixture), 8766 (native model), and 8767 (temporary Docker QA) keep profiles explicit. The separate [public ZeroGPU app](https://huggingface.co/spaces/cyash1204/AdaptLM) is described in [cloud-deployment.md](cloud-deployment.md). Readiness reports fixture generation-disabled separately from a successfully loaded model. A model/tokenizer/adapter load failure leaves health alive and readiness false. Confirm `/api/v1/models` revision, device, backend, and bundle before a demo.

Use the native Apple MPS process on this Mac; Linux Docker does not provide Apple's GPU. The fixture container is limited to 2 GB in Compose. The CPU inference service is limited to 16 GB and may be slow with a 1.5B checkpoint; do not claim GPU latency from that target. Model load/generation measurements determine practical hosting. Cache downloads are approximately 3.1 GB; LoRA checkpoints are much smaller. Keep at least several GB available for cache, adapter/checkpoint exports, and reports. Avoid running other model-heavy jobs concurrently with training.

Model acquisition: `uv run --extra model python scripts/acquire_model.py`. Native cache defaults to `~/.cache/huggingface`; container cache is `/app/models/hf`. Compose mounts `./models:/app/models` and `./reports:/app/reports`, so acquisition and saved reports survive restart. The container runs as UID 10001; grant this user write access to the report/cache mount on Linux. Set credentials through environment or a local ignored `.env`, never image layers or committed files. `docker compose config --quiet` validates syntax; it does not establish a running service.

For rollback, stop serving, remove `ADAPTLM_ADAPTER_PATH` to restore base-only operation, or set it to an earlier **verified** immutable bundle. Restart and inspect readiness/metadata. Adapter file/hash, base revision, tokenizer revision/template, prompt/schema, architecture, training-manifest hash, and runtime-package compatibility checks run before activation. Keep the previous bundle on disk. Exported bundles cannot be overwritten by another training run. No hot-replacement API is exposed, so model selection cannot change during a request.

Interactive requests have zero retention. Local evaluation files use fictional inputs only. The protected deletion route removes the selected report and associated prediction/frozen/runtime files. Operator-owned backups are outside this API's scope. Frozen dataset files are intentionally retained and must remain fictional; real customer data is unsupported in this published demo.

Container build/startup verification, native model startup, resource measurements, and remote status are listed in [acceptance.md](acceptance.md). Public cloud inference has been verified with the genuine exported adapter. [GitHub Actions](https://github.com/cyash24f3/AdaptLM/actions) has passed the deterministic fixture checks and container build. Those checks do not run GPU training or establish semantic quality.

The serving benchmark enables a 20 ms memory sampler. RSS, Metal tensor allocations and Metal driver allocations are distinct measurements; their sampled maxima can miss a brief peak. CUDA uses PyTorch's peak allocator counter, which excludes allocations outside that allocator. Model-load peak and total system memory are not measured by this probe. The original training report sampled only optimizer boundaries, so its memory figure remains a lower bound. Benchmarks record host free RAM and swap because this Mac also runs other applications.

## Tested Docker named-volume path

This Mac's Docker daemon could not bind-mount the Documents directory. The successful CPU verification used named volumes, without changing macOS privacy settings. After building both targets and acquiring the pinned model/adapter, seed only the selected model cache and project bundle:

```sh
docker build --target fixture -t adaptlm-fixture:verified .
docker build --target inference -t adaptlm-inference:verified .
docker run --name adaptlm-cache-seed --user 0 \
  --entrypoint /app/.venv/bin/python \
  -v adaptlm-base-cache:/app/models/hf \
  -v adaptlm-adapter-cache:/app/models/main-adapter-214 \
  adaptlm-fixture:verified \
  -c 'from pathlib import Path; Path("/app/models/hf/hub").mkdir(parents=True, exist_ok=True)'
docker cp "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct" \
  adaptlm-cache-seed:/app/models/hf/hub/
docker cp models/main-adapter-214/. adaptlm-cache-seed:/app/models/main-adapter-214/
docker run --rm -d --name adaptlm-inference-qa --memory 7g \
  -p 127.0.0.1:8768:8000 \
  -e ADAPTLM_OFFLINE=true -e ADAPTLM_ATTENTION_IMPLEMENTATION=sdpa \
  -e ADAPTLM_ADAPTER_PATH=/app/models/main-adapter-214 \
  -e ADAPTLM_SAMPLE_MEMORY=true -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v adaptlm-base-cache:/app/models/hf:ro \
  -v adaptlm-adapter-cache:/app/models/main-adapter-214:ro \
  -v adaptlm-qa-reports:/app/reports adaptlm-inference:verified
```

Use fresh helper/container names when reproducing; an existing container name cannot be reused. Cache and adapter volumes are read-only during serving. The separate report volume was written and reread after a completed container restart, and offline readiness returned true again. Named volumes survive removal of the temporary container. No Hugging Face token or entire user cache was copied.

One selected validation message generated a structurally valid adapted record: 416 input / 81 output tokens, 35.70 seconds total, sampled CPU-process RSS 7,354,621,952 bytes (6.85 GiB), PyTorch 2.14.0+cpu, float32, four CPU threads. The 7 GiB cap had little headroom; it is a bounded smoke check, not a safe production sizing recommendation. Larger inputs, parallel processes and model-load peaks were not tested here. Docker daemon memory was 8,321,515,520 bytes, and unrelated containers were left running. See container-inference-smoke.json for actual metadata. The public deployment remains the verified free ZeroGPU profile.
