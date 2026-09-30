# Acceptance evidence

Statuses refer to observed runs, not just implemented code. The main adapter has been trained and deployed; the complete held-out comparison and serving measurements are recorded separately. Independent human semantic review, an external public-source track, and CUDA training remain unfinished research tracks.

| Item | Current status | Evidence |
|---|---|---|
| Credential-free fixture startup | Verified | Native HTTP walkthrough, clean checkout, fixture container, screenshots/fixture-triage.jpg |
| Dataset/span/frozen-group validation | Verified | datasets/validation-report.json; 1,280 examples / 160 families; no lexical cross-split matches >=0.85 |
| Category baseline | Verified | reports/classifier.json; train-only fit, validation C selection; test 0.8802 category agreement / 0.8830 macro-F1 |
| Actual base-model baselines | Verified complete validation/test | reports/validation-final and reports/test-final; same pinned base, tokenizer and controlled decoding |
| Main adapter trained/exported/reloaded | Verified | reports/training-main-214-run.json; 110 steps / 880 examples / one epoch; downloadable GitHub release |
| Loss masks, gradients, export/reload | Verified smoke and main | reports/training-main-214-gradient-reload-diagnostic.json; adapter tensors changed, sampled frozen base unchanged, 24-token reload parity |
| Checkpoint resume | Verified current 2.14/v2 and historical 2.8 | reports/training-resume-v2-214-run.json; copied own step-one checkpoint recovered optimizer/RNG state and reached step two; original run retained |
| Actual model inference / invalid-output handling | Verified MPS and public CUDA | reports/validation-final; reports/cloud-comparison-smoke.json; failed source spans return no record |
| Four interface views | Verified | Native protected Dataset/Experiments; public fictional read-only Dataset/Experiments and actual three-mode comparison |
| Frozen complete-output held-out comparison | Verified; quality targets failed | reports/test-final/report.json; 720 real outputs, paired family bootstrap, slices, structural audit; adapter accepted 0/240 |
| Serving latency/memory | Verified actual MPS single and bounded load | reports/serving-m5-single.json and serving-m5-load.json; separate sampled RSS/Metal; overloads retained |
| Fixture container | Verified build/startup | adaptlm-fixture:verified; real HTTP demo at localhost:8767 |
| Real inference container | Verified build, offline startup, actual generation and restart | reports/container-inference-smoke.json and container-verification.json; CPU float32; persistent named reports verified |
| Clean setup and deterministic checks | Verified | reports/clean-checkout-verification.json; 46 deterministic tests; Ruff, mypy and separately run real-smoke evidence check |
| GitHub public source / CI / artifact | Verified | Public cyash24f3/AdaptLM; Actions passed; v0.1.0 adapter release downloaded and hashes checked |
| Public free hosting | Verified actual generation | cyash1204/AdaptLM; zero-a10g; original M5 adapter revalidated on PyTorch 2.13 CUDA |
| Free-only service budget | Verified | No paid provider, dedicated GPU, PRO upgrade, card or credits; electricity unknown |
| Rollback compatibility | Implemented and tested | Base/tokenizer/schema/prompt/artifact/runtime hash checks; restart rollback documented |
| CUDA training / QLoRA | Unverified | No CUDA training performed; BF16 LoRA fits the M5, so QLoRA is unnecessary here |
| Human semantic review / external track | Missing | review-queue.csv has blank human columns; data-card.md documents limits |

Historical PyTorch 2.8 failures remain available. The original main training prompt and manifests are immutable; corrected v2 inference prompts apply equally to base and adapted comparison. The selected successful public demo does not replace all-attempted quality measurement. Historical initial classifier manifests contain a `HEAD` placeholder from an empty-repository reporting bug; later manifests record an honest uncommitted state and source hashes instead of assigning a nonexistent originating commit.
