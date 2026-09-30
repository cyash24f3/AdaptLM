# AdaptLM main adapter model card

This is an actual PEFT LoRA adapter for Qwen/Qwen2.5-1.5B-Instruct, pinned base/tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Local bundle: `models/main-adapter-214`. Bundle ID: `adaptlm-26632e778935a9dd`. Apache-2.0 license; base license and notices are included in the repository.

## Training evidence

The main run completed on Apple M5 / MPS with BF16 base weights, FP32 adapter parameters, PyTorch 2.14.0 / Transformers 4.57.1 / PEFT 0.17.1 / TRL 0.24.0. It processed 880 training examples (110 scenario families), 475,113 available formatted tokens, one epoch / 110 optimizer steps. Wall duration including final diagnostics: 25.38 minutes. Rank 8, alpha 16, dropout .05, q/v projection targets, microbatch one / accumulation eight, learning rate 2e-4, linear schedule with 5% warm-up, seed 42, completion-only loss, no packing, no quantization. Selected checkpoint 110 by lowest validation completion loss over 20 fixed validation examples: 0.077011. This teacher-forced loss is not generated-output accuracy.

1,089,536 trainable parameters out of 1,544,803,840 total. Diagnostics observed nonzero adapter gradients, all 112 adapter tensors changed, intended base parameters had no gradients, and a sampled base tensor stayed identical. Export/reload matched a 24-token generated prefix. Prefix parity is a mechanical check, not full-output equivalence. The loss-mask diagnostic verifies masked prompt and learned completion/stop tokens.

The historical `peak_device_allocated_bytes` field in the training diagnostic is 3,130,538,240 bytes sampled at optimizer boundaries. It is a lower bound on observed live allocation, **not a true peak or total unified-memory footprint**. Process RSS was sampled separately. The main run did not include continuous device sampling; true peak memory remains unmeasured. No allocator-limit bypass was enabled. Earlier failed/interrupted PyTorch 2.8 runs remain published as historical investigations, not successful main adapters.

## Provenance and task limits

All labels and messages are implementation-AI-authored fictional data, mechanically checked but independently semantically unreviewed. The same implementation produced inputs, labels, and annotation rules. Dataset: 1,280 rows / 160 scenario families, with frozen 880/160/240 family-grouped train/validation/test splits. Eight variants per family share substantial template structure. Messages are short (27–275 characters); realistic long conversations and broad typo diversity are underrepresented. No independently sourced public ticket track or human review is claimed. Generalization to real customer messages is unknown.

The training prompt is v1, while the matched final inference comparison uses corrected prompt v2 for both base and adapted weights. The exact correction and few-shot selection are documented in experiment-decisions.md. Run-time/source hashes and the original training lock are preserved. Training began before the repository's first commit, so the manifest records an uncommitted repository and source hashes, rather than inventing an originating commit.

The model returns an issue category, raw Unicode entity spans, operational missing fields, and summary claims. Exact source spans are structurally checked; this does not establish semantic correctness. No refund decisions, policy advice, customer actions, or external system writes are performed. Invalid generation returns a failure. Real-world use requires independent annotation review and operational testing. This is a portfolio research artifact.

## Quality and use

Generated validation and held-out results are in the experiment reports and acceptance checklist. Do not infer task quality from completion loss or the classifier's category-only result. Semantic summary support is unmeasured. Fixture outputs are explicitly excluded from quality evaluation. See README for local acquisition, training, inference, container, and rollback commands. Base weights are separately acquired from the pinned official snapshot; the adapter alone cannot run without them.
