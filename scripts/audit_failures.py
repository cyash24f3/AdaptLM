"""Post-test structural audit; never repairs predictions or changes quality metrics."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from adaptlm.artifacts.manifest import digest, write_json
from adaptlm.contracts.schema import TriageRecord


def diagnose(item):
    response = item["response"]
    if response["final_valid"]:
        return "accepted_structurally; semantic support unreviewed"
    if response["status"] != "invalid_output":
        return "transport:" + response["status"]
    if "token limit" in (response.get("error") or ""):
        return "output_token_limit_without_stop"
    raw = response.get("raw", "").strip()
    if raw.startswith("```json\n") and raw.endswith("```"):
        raw = raw[8:-3].strip()
    elif raw.startswith("```\n") and raw.endswith("```"):
        raw = raw[4:-3].strip()
    try:
        value = json.loads(raw)
    except ValueError:
        return "invalid_json"
    try:
        record = TriageRecord.model_validate(value)
    except ValueError:
        return "schema_or_workflow_violation"
    try:
        record.validate_grounding(item["message"])
    except ValueError:
        return "exact_source_span_or_quote_violation"
    return "other_validation_failure"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    output = args.experiment / "structural-audit.json"
    if output.exists():
        raise ValueError("preserve prior audit; choose another experiment copy")
    prediction_path = args.experiment / "predictions.jsonl"
    report = json.loads((args.experiment / "report.json").read_text())
    counts, examples = defaultdict(Counter), defaultdict(dict)
    for line in prediction_path.read_text().splitlines():
        item = json.loads(line)
        reason = diagnose(item)
        counts[item["mode"]][reason] += 1
        if reason not in examples[item["mode"]]:
            examples[item["mode"]][reason] = {
                key: item[key] for key in ("id", "message", "response", "gold")
            }
    for mode, values in counts.items():
        if sum(values.values()) != report["metrics"][mode]["attempted"]:
            raise ValueError("audit counts do not match the complete experiment")
    write_json(
        output,
        {
            "kind": "post-test structural failure audit; no prediction changes",
            "predictions_sha256": digest(prediction_path),
            "report_sha256": digest(args.experiment / "report.json"),
            "script_sha256": digest(Path(__file__)),
            "counts": {mode: dict(values) for mode, values in counts.items()},
            "first_example_by_reason": dict(examples),
            "limitations": "Schema/source checks do not establish semantic correctness. Reasons use the published contract; this audit does not tune, regenerate, repair, or replace the frozen run.",
        },
    )
    print(json.dumps({mode: dict(values) for mode, values in counts.items()}, indent=2))


if __name__ == "__main__":
    main()
