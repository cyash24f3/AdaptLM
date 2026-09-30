# Authored dataset v1

1,280 examples, 160 scenario families, eight expansions per family. 880 train / 160 validation / 240 test. Families are allocated before expansion with seed 42 and stratification over eight issue categories, ambiguity, and out-of-scope. IDs, entities, greetings, repeated mentions, Unicode context, informal text, and injection tails are variants, **not independent tickets**.

Inputs and underlying scenario meanings are implementation-AI-generated. Labels are rule-generated with the same scenario assumptions. Structural checks are automatic; no human semantic review has occurred. This correlated source is a major benchmark limitation. Template-derived categories can be easier than real customer intent. Prefix/suffix variety must not be advertised as thousands of independently varied scenarios.

Frozen file hashes, family-plan hash, generator seed, creation method, source distribution, category distributions, and character lengths are in `manifest.json` and `validation-report.json`. The prompt tokenizer diagnostic records actual training token lengths. Full benchmark records retain tokenizer counts. Exact original-message duplicates and cross-split family membership are rejected. Cross-split character 3–5 gram TF-IDF cosine similarities are checked over scenario cores at 0.85; IDs and repeated variant boilerplate are excluded. Lexical screening is not a semantic contamination guarantee. No embedding check has been run.

`scenarios.json` contains AI-authored scenario sentences. `review-queue.csv` contains blank reviewer columns. Do not change frozen membership after evaluation; produce a new dataset version instead. `test-exposures.jsonl` logs final test evaluations. Reading reference summaries for split/structural validation is dataset construction, not prompt or training tuning; the classifier has already had its first final test exposure. Preserve first results and record any later test-informed change.

## External validation

No public support corpus was imported. Licensing, schema mapping, independent full-output annotations, and privacy review remain unfinished for the external track. A future category-only corpus may evaluate the classifier's category field separately; it cannot verify full triage records. The current repository contains fictional data only, and no scraped customer conversations.
