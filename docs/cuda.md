# Optional CUDA training — unverified

This Mac has no CUDA device. The provided configuration is a separate planned accelerator experiment, not a tested universal installation. Native Mac training uses the committed uv lock. Linux's standard uv model profile deliberately uses official CPU wheels so CI/containers do not install gigabytes of unused CUDA libraries.

For an actual compatible NVIDIA machine, make an isolated environment explicitly and inspect its device, driver, and installed wheel before training. The following recipe is **unverified** on CUDA hardware and must not be reported as passed:

```sh
uv venv --python 3.12 .venv-cuda
uv pip install --python .venv-cuda/bin/python torch==2.14.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-cuda/bin/python -e '.[model,train]'
.venv-cuda/bin/python scripts/inspect_environment.py
.venv-cuda/bin/adaptlm train --config configs/train-cuda.json --output models/cuda-run
```

Use the official wheel index matching the actual supported driver/runtime. The example cu130 choice is explicit, not a driver compatibility promise. Save `uv pip freeze --python .venv-cuda/bin/python` into the experiment's environment record when actually installed; no tested CUDA lock is claimed here. Do not run the CPU-profile `uv sync` against this isolated CUDA environment.

The supplied config requests BF16. Check native BF16 support on the actual GPU; a free T4 usually needs a different validated precision/configuration. QLoRA is not implemented because the actual M5 run does not require it. Start with the real smoke diagnostics on the CUDA machine, measure memory, and adjust a new configuration on validation before the full run. Do not launch a paid machine; free notebook runtime availability is not guaranteed. Checkpoints and safe adapter exports are portable only after the contract/runtime checks pass.
