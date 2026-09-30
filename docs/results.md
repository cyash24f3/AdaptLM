# Controlled results and limitations

The main adapter was actually trained, exported, reloaded and publicly deployed. It **did not meet the frozen task-quality targets**. On the complete held-out track, adapted generation produced zero structurally accepted records out of 240 attempts. This negative result is retained as the central experimental finding.

## Frozen held-out comparison

[Full report](../reports/test-final/report.json), [raw per-example predictions](../reports/test-final/predictions.jsonl), [frozen configuration](../reports/test-final/frozen-config.json), and [post-test structural audit](../reports/test-final/structural-audit.json).

All three generative modes used the same pinned Qwen2.5-1.5B-Instruct checkpoint, tokenizer, v2 task specification, greedy decoding, 384-token output maximum, and MPS BF16/SDPA runtime. Base and adapted modes differ only by adapter activation. Few-shot adds four fixed training demonstrations. Training used the original v1 prompt; its manifests remain unchanged. Checkpoint 110 was selected on validation completion loss before generative test exposure. No test-informed changes to prompts, adapter, decoding or metrics were made.

The test contains 240 rows from 30 independent scenario families. Category agreement excludes 48 null-category rows, leaving 192 rows / 24 families, with 24 examples per issue category. All attempts remain in applicable denominators, including failures. Labels are implementation-AI-authored and independently unreviewed; agreement is not established real-world accuracy.

| Mode | Raw schema + source validity | After fence-only repair | Category agreement, all applicable attempts | Status agreement, all attempts | Complete exact reference agreement |
|---|---:|---:|---:|---:|---:|
| Zero-shot base | 30/240 (12.5%) | 32/240 (13.3%) | 0/192 (0%) | 7/240 (2.9%) | 0/240 (0%) |
| Few-shot base | 0/240 (0%) | 0/240 (0%) | 0/192 (0%) | 0/240 (0%) | 0/240 (0%) |
| Main adapter | 0/240 (0%) | 0/240 (0%) | 0/192 (0%) | 0/240 (0%) | 0/240 (0%) |

The zero-shot valid-output-only category score is also 0%; few-shot/adapted valid-only scores are undefined because no outputs were accepted. Their zero all-attempted category scores measure usable validated predictions, not a separate category extracted from rejected JSON. Zero-shot's 32 accepted outputs all declared out_of_scope; only seven matched the reference status. Structural validity therefore did not establish correct triage.

Strict accepted-entity recall/F1 and missing-field recall/F1 were zero for every mode: 264 reference entity mentions and 174 required-field instances were missed. Precision is undefined because no accepted entity/missing-field predictions existed. No accepted summary claims were available for semantic assessment. Summary semantic support and complete semantic success remain unmeasured, not zero measured human accuracy.

The category-only TF-IDF/logistic baseline achieved 169/192 (88.0%) category agreement and macro-F1 0.883 on the same applicable test rows. It produces no complete triage record. The result supports retaining this simple category baseline for further evaluation; it does not establish entity, summary or operational-workflow correctness.

## Uncertainty and predeclared targets

Family-level paired bootstrap uses 1,000 resamples, seed 42, and keeps variants together. Adapted-minus-zero-shot raw validity was **−12.5 percentage points**, with a 95% interval **[−17.1, −7.9] points**, over 30 families. Category and complete-reference deltas were zero with intervals [0, 0]. Status delta was −2.9 points, interval [−6.7, 0]. These intervals characterize this narrow authored track and do not correct correlated labeling errors or establish real-world generalization.

[Targets were frozen before test](../configs/final-experiment.json): adapted raw validity at least the observed zero-shot validation rate of 60% — **failed**; positive category paired interval lower bound — **failed**. Exact grounding remains an enforced API contract, but zero accepted adapter records make any accepted-entity success claim vacuous. No entity-recall or unnecessary-clarification threshold was invented because canonical validation lacked the relevant reference cases. Test unnecessary clarification was 0/69 reference-triaged examples, and wrong confident labels on ambiguous/out-of-scope inputs were 0/48; widespread rejection prevents interpreting those zeros as useful behavior.

The provisional warm p95 latency budget is 10.825 seconds. Serving is measured separately; it cannot compensate for failed quality targets.

## What failed

The post-test audit checks the same published contract without repairing, regenerating or changing predictions:

| Primary structural outcome | Zero-shot | Few-shot | Adapted |
|---|---:|---:|---:|
| Accepted by structural checks | 32 | 0 | 0 |
| Exact source span / quote violation | 64 | 79 | 213 |
| Schema / workflow violation | 142 | 126 | 26 |
| Invalid JSON | 1 | 17 | 1 |
| Output limit without stop | 1 | 18 | 0 |

For adapted test example `authored-account_access-02-0`, the 48-character message was “My account is locked and I need access restored.” The generated summary quoted it correctly but declared span [0, 52), outside the input. The API rejected the record. In an emoji-prefixed few-shot example, the quote began after the emoji while the predicted start remained zero. Repeated-email base outputs declared offsets that did not select the claimed address. The audit contains actual first examples and raw outputs for each reason.

Slices by issue, input length, source, entity type, missing field and injection are in the full report. The adapter's zero acceptance applies across all test slices; this is not evidence of successful injection resistance. Messages are only 27–275 characters, and the 1,280 rows comprise eight template-related variants per 160 authored families. Long real conversations, broad typos, independent sources and human-reviewed meanings remain missing.

The highest-value next experiment is reliable source alignment, with independently reviewed examples and an explicitly evaluated quote-to-offset or constrained-decoding method. Any such change must receive a new experiment identity and be marked informed by this exposed holdout; it must not overwrite or masquerade as the first frozen run. More epochs cannot be assumed to solve numeric alignment. The original adapter stays available as a research artifact, with visible rejection and a selected successful validation demo.

## Provenance

The original long-running evaluator logged its source/environment before generation in [execution-start.json](../reports/test-final/execution-start.json). Its original report took another environment snapshot at completion, after unrelated source/documentation updates. [provenance.json](../reports/test-final/provenance.json) identifies that distinction and hashes the unchanged predictions/report. Loaded evaluation code is identified by the startup manifest. Future evaluations now freeze environment-at-start.json automatically. The reported evaluation-process duration was 5,485.6 seconds (91.4 minutes); the shared host and long-lived session were not a dedicated timing experiment.

The public CUDA smoke and selected monitor demo verify deployment mechanics. They do not replace these MPS scores. Human semantic review and a licensed independent public-source test track remain explicit next research steps.

A separate v2/PyTorch 2.14 mechanics smoke trained two steps on eight training examples in 21.14 seconds. Recovery from an exact copy of its own step-one checkpoint and original manifest reached step two in 14.71 seconds. Both runs verified nonzero adapter gradients, a sampled unchanged base tensor, completion masks and export/reload prefix parity. Their reports and checkpoint-file provenance are published; this is a recovery check, not a second main-quality run or a test-informed adapter replacement.

## Serving measurements

[Single-request report](../reports/serving-m5-single.json) and [bounded-load report](../reports/serving-m5-load.json) use the actual main bundle on Apple M5, 24 GB unified RAM, MPS BF16/SDPA, with no quantization. The three fixed short validation inputs and their hash are recorded. Each mode excludes one warm-up; the single-request run measures ten requests per mode. Cold process/model setup with cached weights took 2.536 seconds; model-load peak memory was not sampled. The separate load process took 2.201 seconds to load. Other applications were uncontrolled, around 10 GB RAM was available, and 2.09 GB swap was in use at start.

| Mode | Warm p50 | Warm p95 | Generated tokens / generation second | Input tokens | Output tokens |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 1.559 s | 1.619 s | 21.05 | 417–421 | 33 |
| Few-shot | 4.222 s | 4.930 s | 19.87 | 972–976 | 81–97 |
| Adapted | 5.061 s | 5.306 s | 19.48 | 417–421 | 85–101 |

The measured single-request p95 met the provisional 10.825-second budget. All ten adapted/few-shot serving outputs failed validation; all ten zero-shot outputs passed structural validation but declared out_of_scope. These timings therefore describe generated responses and failures, not useful-triage throughput. Output lengths differ, so total latency alone cannot isolate adapter overhead. Ten observations do not establish a stable production p95 or SLA.

The modest load run submits eight requests per mode in batches of four. One generation plus two waiting requests are admitted; the fourth is rejected. Every mode recorded six completed generations and two overloaded responses. Completed-request p95 including queue wait was 4.228 / 11.235 / 12.293 seconds for zero-shot / few-shot / adapted. Throughput of completed generations, including invalid outputs, was 0.711 / 0.267 / 0.244 requests per wall second. Rejections are retained in request counts and validity denominators and excluded from generation-latency quantiles. There was no measured speedup from asynchronous admission: one worker owns adapter routing.

The nominal 20 ms probe's largest sampled single-request maxima across modes were: process RSS **624,476,160 bytes (0.58 GiB)**, Metal live tensor allocation **3,149,640,448 bytes (2.93 GiB)**, and Metal driver allocation **3,382,493,184 bytes (3.15 GiB)**. Values are distinct, can overlap, and must not be summed as total unified-memory use. Scheduling can delay samples and miss brief peaks. Probe overhead is included; model-load peak and total host/system peak are unmeasured. The original training-boundary allocation lower bound remains separate from these inference measurements.

The free public ZeroGPU route has been verified with actual CUDA generation; its selected smoke timing is separate from this controlled MPS benchmark. Free quotas and queues make it suitable for a portfolio demo, with no availability promise. Native MPS remains the measured local development profile. No paid hosting recommendation or tier upgrade is implied.

The offline Linux ARM64 inference container also loaded the original bundle on PyTorch 2.14.0+cpu / float32 and generated the selected monitor record in 35.70 seconds (81 output tokens). Sampled process RSS reached 6.85 GiB under a 7 GiB cap. This single selected smoke has little memory headroom and is not a CPU quality/load comparison. A separate report-volume write survived restart, and offline readiness returned true. [Container evidence](../reports/container-inference-smoke.json) records packages, image identity and actual response; the [deployment guide](deployment.md) includes the tested named-volume workaround for macOS Documents mount restrictions.
