# Free-only operating plan — 1 October 2026

| Need | Chosen path | Paid service required |
|---|---|---|
| Data authoring and labeling | Committed fictional scenarios + local deterministic tooling | No |
| Baseline/classifier evaluation | Local CPU | No |
| Base weights | Public official pinned Hugging Face snapshot | No |
| Training and inference | Native Apple M5 / PyTorch MPS | No |
| Experiment tracking | JSON/JSONL manifests and reports | No |
| Web application and database | FastAPI/Jinja2, local files; no database needed | No |
| Public deployment | Hugging Face ZeroGPU Space, explicitly free hardware | No |
| Source, CI and adapter download | Public GitHub repository, Actions and release | No |
| Native development | Local FastAPI; Docker for fixture/CPU | No |
| Semantic judging | Review queue; no paid LLM judge | No |

No paid subscriptions, card entry, paid API credits, or paid cloud GPU provisioning occurred. The existing free Hugging Face account was eligible for ZeroGPU; the app is deployed and actual generation was verified. This budget refers to service spend; electricity, bandwidth, and existing hardware/subscription costs are not measured. Training uses the supplied Mac. Cloud generation uses the free Space with visible quota failures and no paid fallback.

## Optional free resources and their limits

Google Colab can provide free notebooks, but GPUs and runtime duration are variable and not guaranteed; the project does not depend on them. Check the [official FAQ](https://research.google.com/colaboratory/faq.html) before planning a free GPU session. Export locally generated Trainer checkpoints to preserve resume state if a session is interrupted. CUDA setup remains unverified until run on the assigned device.

The [official ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) permits two ZeroGPU Spaces for qualifying free personal accounts. Anonymous visitors currently receive two GPU minutes per day, and free logged-in visitors five minutes. These quotas are shared with other ZeroGPU use and may change. Queue delays and exhausted quotas can prevent a request; the application does not promise continuous availability.

The [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) distinguishes free static hosting from compute Spaces. New Docker/Gradio CPU Spaces require a paid plan; this project uses the explicitly free ZeroGPU exception. Do not select a dedicated GPU or paid plan to reproduce it. [Cloud deployment](cloud-deployment.md) records eligibility, hardware selection and the separate cloud runtime.
