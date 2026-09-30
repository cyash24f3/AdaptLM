"""All-attempted denominators; reference agreement is not independently verified accuracy."""

import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import f1_score


def entity_set(record):
    return {
        (e["type"], e["span"]["start"], e["span"]["end"], e["value"])
        for e in record.get("entities", [])
    }


def missing_set(record):
    return {m["field"] for m in record.get("missing_information", [])}


def prf(tp, fp, fn):
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def per_example(row, response):
    gold = row["target"]
    prediction = response.get("output") or {}
    valid = response["final_valid"]
    entities, expected = entity_set(prediction), entity_set(gold)
    missing, required = missing_set(prediction), missing_set(gold)
    category_correct = valid and prediction.get("primary_issue") == gold["primary_issue"]
    status_correct = valid and prediction.get("triage_status") == gold["triage_status"]
    summaries = prediction.get("summary_claims", [])
    gold_summaries = gold["summary_claims"]
    # This is strict reference agreement, never a semantic judge.
    summary_reference_match = summaries == gold_summaries
    complete = (
        valid
        and category_correct
        and status_correct
        and entities == expected
        and (missing == required)
        and summary_reference_match
    )
    raw_mentions, unsupported_spans, malformed_mentions = 0, 0, 0
    try:
        raw = response.get("raw", "").strip()
        if raw.startswith("```json\n") and raw.endswith("```"):
            raw = raw[8:-3].strip()
        raw_record = json.loads(raw)
        for entity in raw_record.get("entities", []):
            raw_mentions += 1
            try:
                span = entity["span"]
                start, end, quote = span["start"], span["end"], span["text"]
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not isinstance(quote, str)
                ):
                    raise TypeError
                if (
                    start < 0
                    or end <= start
                    or end > len(row["message"])
                    or row["message"][start:end] != quote
                ):
                    unsupported_spans += 1
            except (TypeError, KeyError):
                malformed_mentions += 1
    except (ValueError, TypeError, AttributeError):
        pass
    return {
        "id": row["id"],
        "family_id": row["family_id"],
        "source": row["source"],
        "slices": row["slices"]
        + [
            "source:" + row["source"],
            "length:<=80"
            if len(row["message"]) <= 80
            else "length:81-160"
            if len(row["message"]) <= 160
            else "length:>160",
        ]
        + sorted({"entity:" + e["type"] for e in gold["entities"]})
        + sorted({"missing:" + m["field"] for m in gold["missing_information"]}),
        "length_chars": len(row["message"]),
        "gold_category": gold["primary_issue"],
        "predicted_category": prediction.get("primary_issue"),
        "gold_status": gold["triage_status"],
        "predicted_status": prediction.get("triage_status", "invalid"),
        "category_applicable": gold["primary_issue"] is not None,
        "category_correct": bool(category_correct),
        "status_correct": bool(status_correct),
        "entity_tp": len(entities & expected),
        "entity_fp": len(entities - expected),
        "entity_fn": len(expected - entities),
        "raw_entity_mentions": raw_mentions,
        "raw_entity_unsupported_spans": unsupported_spans,
        "raw_entity_malformed_mentions": malformed_mentions,
        "missing_tp": len(missing & required),
        "missing_fp": len(missing - required),
        "missing_fn": len(required - missing),
        "summary_reference_match": summary_reference_match,
        "summary_claims_attempted": len(summaries),
        "summary_support_semantic_review": "pending independent human review",
        "extractive_claims": sum(
            c["text"] == " ".join(s["text"] for s in c["spans"]) for c in summaries
        ),
        "complete_record_reference_success": bool(complete),
        "wrong_confident_ambiguous_or_oos": gold["primary_issue"] is None
        and valid
        and prediction.get("primary_issue") is not None,
        "unnecessary_clarification": gold["triage_status"] == "triaged"
        and valid
        and prediction.get("triage_status") == "needs_clarification",
        "raw_valid": response["raw_valid"],
        "final_valid": valid,
        "status": response["status"],
        "timings": response["timings"],
        "usage": response["usage"],
        "repair_count": response["repair_count"],
        "response": response,
        "gold": gold,
        "message": row["message"],
    }


def aggregate(results):
    n = len(results)
    if not n:
        return {"attempted": 0}
    categories = [r for r in results if r["category_applicable"]]
    valid_categories = [r for r in categories if r["final_valid"]]
    all_gold = [r["gold_category"] for r in categories]
    labels = sorted(set(all_gold))

    def acc(rs, key):
        return sum(r[key] for r in rs) / len(rs) if rs else None

    def total(key):
        return sum(r[key] for r in results)

    uncertain = [r for r in results if r["gold_category"] is None]
    clear = [r for r in results if r["gold_status"] == "triaged"]
    claims = total("summary_claims_attempted")
    entities = total("entity_tp") + total("entity_fp")
    return {
        "attempted": n,
        "valid_outputs": total("final_valid"),
        "category_applicable": len(categories),
        "category_exclusions": n - len(categories),
        "category_accuracy_all_attempted": acc(categories, "category_correct"),
        "category_accuracy_valid_only": acc(valid_categories, "category_correct"),
        "category_macro_f1_all_attempted": float(
            f1_score(
                all_gold,
                [r["predicted_category"] or "invalid" for r in categories],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        if categories
        else None,
        "category_support": dict(Counter(all_gold)),
        "status_accuracy_all_attempted": acc(results, "status_correct"),
        "status_macro_f1": float(
            f1_score(
                [r["gold_status"] for r in results],
                [r["predicted_status"] for r in results],
                labels=["triaged", "needs_clarification", "out_of_scope"],
                average="macro",
                zero_division=0,
            )
        ),
        "strict_entities": prf(total("entity_tp"), total("entity_fp"), total("entity_fn")),
        "missing_information": prf(total("missing_tp"), total("missing_fp"), total("missing_fn")),
        "raw_schema_validity": total("raw_valid") / n,
        "post_repair_validity": total("final_valid") / n,
        "entity_nonreference_rate": total("entity_fp") / entities if entities else None,
        "entity_nonreference_example_incidence": sum(r["entity_fp"] > 0 for r in results) / n,
        "unsupported_entity_semantic_rate": None,
        "raw_entity_span_unsupported_rate": total("raw_entity_unsupported_spans")
        / total("raw_entity_mentions")
        if total("raw_entity_mentions")
        else None,
        "raw_entity_mentions_denominator": total("raw_entity_mentions"),
        "raw_entity_malformed_mentions": total("raw_entity_malformed_mentions"),
        "raw_unsupported_entity_example_incidence": sum(
            r["raw_entity_unsupported_spans"] > 0 for r in results
        )
        / n,
        "unsupported_entity_limitation": "Nonreference spans are not proof of unsupported meaning; semantic review pending.",
        "summary_claims": {
            "attempted_valid_outputs": claims,
            "verbatim_extracts": total("extractive_claims"),
            "semantic_supported": None,
            "semantic_unsupported": None,
            "semantic_ambiguous": None,
            "unreviewed": claims,
            "rubric": "docs/annotation-guide.md; exact matching cannot establish semantic correctness",
        },
        "summary_reference_agreement": acc(results, "summary_reference_match"),
        "complete_record_reference_success": acc(results, "complete_record_reference_success"),
        "complete_record_semantic_success": None,
        "wrong_confident_on_ambiguous_or_oos": acc(uncertain, "wrong_confident_ambiguous_or_oos"),
        "ambiguous_or_oos_denominator": len(uncertain),
        "unnecessary_clarification_on_clear": acc(clear, "unnecessary_clarification"),
        "clear_denominator": len(clear),
        "transport_status_counts": dict(Counter(r["status"] for r in results)),
    }


def paired_bootstrap(a, b, key="complete_record_reference_success", seed=42, samples=1000):
    """Resample independent scenario families, preserving paired inputs and variants."""
    left, right = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if set(left) != set(right):
        raise ValueError("paired comparison requires identical example IDs")
    grouped = defaultdict(list)
    for id_, row in left.items():
        grouped[row["family_id"]].append(float(right[id_][key]) - float(row[key]))
    families = sorted(grouped)
    if len(families) < 10:
        return {
            "independent_families": len(families),
            "interval": None,
            "reason": "fewer than ten independent families",
        }
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        sampled = rng.choice(families, size=len(families), replace=True)
        deltas.append(float(np.mean([d for family in sampled for d in grouped[family]])))
    return {
        "independent_families": len(families),
        "paired_delta_b_minus_a": float(
            np.mean([d for values in grouped.values() for d in values])
        ),
        "95_percent_family_bootstrap_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "seed": seed,
        "resamples": samples,
        "metric": key,
    }
