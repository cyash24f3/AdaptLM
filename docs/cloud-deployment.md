# Public free cloud deployment

The public target is [cyash1204/AdaptLM on Hugging Face](https://huggingface.co/spaces/cyash1204/AdaptLM). GitHub source is [cyash24f3/AdaptLM](https://github.com/cyash24f3/AdaptLM). The deployment manifest distinguishes an uploaded build from verified generation; see acceptance.md for current evidence.

This deployment uses **ZeroGPU**, not a paid dedicated GPU or inference API. Free personal accounts with a verified email and age over 30 days can host two ZeroGPU Spaces. Visitor quotas currently provide two GPU minutes per day without login and five minutes for free logged-in users. Queue delays, cold starts, quota exhaustion and sleep are expected. No PRO subscription, payment method, prepaid credits or hardware upgrade was added. Read the [official ZeroGPU rules](https://huggingface.co/docs/hub/spaces-zerogpu) for current eligibility and quotas. Gradio/Docker CPU Space creation currently requires a paid plan; this deployment explicitly requests the free ZeroGPU exception.

## Profiles and compatibility

The native polished FastAPI interface supports fixture/local MPS profiles and protected data/report administration. The public cloud interface uses four read-only Gradio tabs (Triage, Compare, Dataset, Experiments), reusing the exact source contract, prompts, model engine, inference service and published reports. Public Dataset/Experiments are intentionally published fictional snapshots; no administrative write, upload, training or secret-entry route is exposed. Gradio's API exposes `triage` and `compare`; the native `/api/v1` endpoints belong to the FastAPI deployment profile.

Local training and the controlled held-out experiment pin PyTorch 2.14.0 on MPS. ZeroGPU currently supports 2.13.0, so deployment pins that version separately, with the same Transformers/PEFT, base/tokenizer revision, task prompt, safe adapter files and original training manifest. Default bundle loading still rejects version mismatches. Only the trusted cloud entry point explicitly selects 2.13.0 for revalidation. Its first GPU request checks finite base/adapted logits, a nonzero adapter effect, and repeated 24-token greedy-prefix agreement before enabling generation. The resulting audit accompanies response metadata and is written to `reports/cloud-runtime-validation.json` in the Space. This verifies mechanics in the candidate runtime, not cross-device output equality or semantic quality. No retraining or relabeling of the M5 adapter occurs. Failed revalidation prevents generation. CUDA quality/latency must not be presented as the controlled MPS scores.

## Reproduce publication

Authenticate with the Hugging Face CLI using your own write access, acquire or train the genuine adapter, and run:

```sh
uv run --extra model python scripts/deploy_space.py --space YOUR_ACCOUNT/AdaptLM
```

The script requests only `zero-a10g` (the SDK's free ZeroGPU hardware identifier), checks the authenticated personal owner, and refuses to change a different hardware configuration. It copies an allowlist of source/configs/public fictional data/reports plus the exported small adapter; it excludes `.env`, local credentials, checkpoints, base weights and caches. Review reports before publishing your own additional experiments. Separate pinned requirements are in deployment/huggingface. Their package/runtime details are captured by the on-device audit. No secret token is copied into the public Space. Public base weights download without a token from the pinned official snapshot.

Interactive text is sent to Hugging Face infrastructure. The application does not intentionally store it, enable analytics, or log raw messages. Provider-level processing and logs are governed by Hugging Face, so visitors should use fictional inputs. Any intentionally saved evaluation outputs use authored fictional data. Queue size is three with one shared model execution across both cloud buttons, 3,000 input characters, tokenized input limits and 384 generated-token maximum. Quota/queue failures remain visible; they do not trigger a paid provider fallback.

For rollback, retain the Space commit and original bundle ID in cloud-deployment.json, redeploy a previous verified source/bundle combination, and repeat a real GPU request. Base-only comparison disables the adapter inside the same serialized model. Never overwrite unrelated Spaces or select paid hardware as a workaround.
