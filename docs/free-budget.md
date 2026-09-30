# Free-only operating plan — 30 September 2026

| Need | Chosen path | Paid service required |
|---|---|---|
| Data authoring and labeling | Committed fictional scenarios + local deterministic tooling | No |
| Baseline/classifier evaluation | Local CPU | No |
| Base weights | Public official pinned Hugging Face snapshot | No |
| Training and inference | Native Apple M5 / PyTorch MPS | No |
| Experiment tracking | JSON/JSONL manifests and reports | No |
| Web application and database | FastAPI/Jinja2, local files; no database needed | No |
| Deployment | Local native server; Docker for fixture/CPU | No |
| Semantic judging | Review queue; no paid LLM judge | No |

No paid subscriptions, card entry, API credit consumption, or cloud GPU provisioning occurred. This budget refers to service spend; electricity, bandwidth, and existing hardware/subscription costs are not measured. The project uses the supplied Mac rather than signing up for unnecessary services. No hidden fallback invokes a hosted API.

## Optional free resources and their limits

Google Colab can provide free notebooks, but GPUs and runtime duration are variable and not guaranteed; the project does not depend on them. Check the [official FAQ](https://research.google.com/colaboratory/faq.html) before planning a free GPU session. Export locally generated Trainer checkpoints to preserve resume state if a session is interrupted. CUDA setup remains unverified until run on the assigned device.

The [current Hugging Face Spaces documentation](https://huggingface.co/docs/hub/spaces-overview) says static Spaces are free for everyone, while creating Docker/Gradio compute Spaces requires a paid plan, with separate limited ZeroGPU eligibility. Consequently **do not describe a new Docker Space as an unconditional free deployment**. Static hosting cannot run this Python/model API. No remote hosting provider was selected or provisioned; the chosen free serving path is localhost. Use account-specific eligibility checks before changing this plan, and do not enable paid upgrades.
