# Acceptance evidence

This file distinguishes actual verification from implemented/unverified behavior. A real main adapter and adapted held-out evaluation are required before calling the adaptation milestone complete. Remote hosting, CUDA, external semantic review, and independent public-source validation are separate unfinished tracks.

| Item | Current status | Evidence |
|---|---|---|
| Credential-free fixture startup | Verified | Native HTTP demo, fixture Docker startup, screenshots/fixture-triage.jpg |
| Dataset/span/frozen-group validation | Verified | datasets/validation-report.json; 1,280 examples, 160 families, no lexical cross-split matches >=0.85 |
| Category baseline | Verified | reports/classifier.json; train-only fit, validation C selection; test 0.8802 category agreement / 0.8830 macro-F1 |
| Actual base-model baselines | Verified pilot, final comparison pending | reports/validation-base/report.json (old runtime investigation), base-214-validation-smoke.json; actual raw outputs retained |
| Main-model adapter trained/exported/reloaded | Verified | models/main-adapter-214; reports/training-main-214-run.json; 110 steps / 880 examples / 1 epoch |
| Loss masks, gradients, export/reload | Verified smoke and main | reports/training-main-214-gradient-reload-diagnostic.json; adapters changed, frozen base sample unchanged, reloaded token prefix parity |
| Checkpoint resume | Verified on 2.8; final runtime check pending | reports/training-smoke-resume-28.json; local optimizer/RNG checkpoint |
| Real Transformers base inference / invalid output handling | Verified | actual base-214 validation response; source validation rejects malformed/unsupported records |
| Four interface views and unavailable modes | Verified | Actual browser Triage, Compare, protected Dataset, Experiments screenshots |
| Frozen complete-output held-out comparison | Implemented, pending execution | evaluate CLI, metrics tests; semantic support remains independently unreviewed |
| Serving latency/memory | Fixture verified, real measurement pending | reports/fixture-serving.json; real benchmark CLI records tokenizer/timing/allocation details |
| Fixture container | Verified | adaptlm-fixture:local built and HTTP demo passed at localhost:8767 |
| Real inference container | Built on old lock; final build/startup pending | docker-inference log; CPU-only Linux profile |
| Clean source / lock / CI checks | Local checks passed; clean checkout pending | 36 deterministic tests, Ruff, mypy; remote CI not run |
| Free-only operating budget | Verified for service spend | No paid provider/API/GPU provisioned; local public checkpoint; electricity unknown |
| Rollback compatibility | Implemented / unit tested | Bundle checks for base/tokenizer/schema/prompt/hash; restart rollback documented |
| CUDA/QLoRA/remote deployment | Unverified / not configured | Mac has MPS, no CUDA; QLoRA not needed; no remote service provisioned |
| Human semantic review / external track | Missing | review-queue.csv has blank human columns; data-card.md explains limitations |

The original PyTorch 2.8 investigations are retained with failures. Validation-informed runtime/prompt changes are separate from any final adapted comparison. Never treat those backend changes as an adapter improvement. Further measured results will update this checklist when execution finishes.
