# Acceptance evidence

Statuses refer to observed runs, not just implemented code. The main adapter has been trained and deployed; the complete held-out comparison and serving measurements are recorded separately. Independent human semantic review, an external public-source track, and CUDA training remain unfinished research tracks.

| Item | Current status | Evidence |
|---|---|---|
| Credential-free fixture startup | Verified | Native HTTP walkthrough, clean checkout, fixture container, screenshots/fixture-triage.jpg |
| Dataset/span/frozen-group validation | Verified | datasets/validation-report.json; 1,280 examples / 160 families; no lexical cross-split matches >=0.85 |
| Category baseline | Verified | reports/classifier.json; train-only fit, validation C selection; test 0.8802 category agreement / 0.8830 macro-F1 |
| Actual base-model baselines | Verified validation; full test running | reports/validation-final; same pinned base, tokenizer and controlled decoding |
| Main adapter trained/exported/reloaded | Verified | reports/training-main-214-run.json; 110 steps / 880 examples / one epoch; downloadable GitHub release |
| Loss masks, gradients, export/reload | Verified smoke and main | reports/training-main-214-gradient-reload-diagnostic.json; adapter tensors changed, sampled frozen base unchanged, 24-token reload parity |
| Checkpoint resume | Verified on 2.8; current runtime check pending | reports/training-smoke-resume-28.json; current code rejects changed experiment manifests |
| Actual model inference / invalid-output handling | Verified MPS and public CUDA | reports/validation-final; reports/cloud-comparison-smoke.json; failed source spans return no record |
| Four interface views | Verified | Native protected Dataset/Experiments; public fictional read-only Dataset/Experiments and actual three-mode comparison |
| Frozen complete-output held-out comparison | Running | reports/test-final/frozen-config.json and append-only predictions; no completed-report claim until report.json exists |
| Serving latency/memory | Fixture verified; real measurement pending | reports/fixture-serving.json; separate sampled RSS/Metal probe implemented |
| Fixture container | Verified build/startup | adaptlm-fixture:verified; real HTTP demo at localhost:8767 |
| Real inference container | Verified CPU build; startup pending | runs/docker-inference-final.log; native Metal is not available inside Linux Docker |
| Clean setup and deterministic checks | Verified | Clean cloned fixture checkout; 46 deterministic tests; Ruff and mypy |
| GitHub public source / CI / artifact | Verified | Public cyash24f3/AdaptLM; Actions passed; v0.1.0 adapter release downloaded and hashes checked |
| Public free hosting | Verified actual generation | cyash1204/AdaptLM; zero-a10g; original M5 adapter revalidated on PyTorch 2.13 CUDA |
| Free-only service budget | Verified | No paid provider, dedicated GPU, PRO upgrade, card or credits; electricity unknown |
| Rollback compatibility | Implemented and tested | Base/tokenizer/schema/prompt/artifact/runtime hash checks; restart rollback documented |
| CUDA training / QLoRA | Unverified | No CUDA training performed; BF16 LoRA fits the M5, so QLoRA is unnecessary here |
| Human semantic review / external track | Missing | review-queue.csv has blank human columns; data-card.md documents limits |

Historical PyTorch 2.8 failures remain available. The original main training prompt and manifests are immutable; corrected v2 inference prompts apply equally to base and adapted comparison. The selected successful public demo does not replace all-attempted quality measurement. Historical initial classifier manifests contain a `HEAD` placeholder from an empty-repository reporting bug; later manifests record an honest uncommitted state and source hashes instead of assigning a nonexistent originating commit.
