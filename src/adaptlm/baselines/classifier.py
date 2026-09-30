"""Strong word + character TF-IDF; hyperparameters use validation only."""

import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from adaptlm.artifacts.manifest import environment, write_json
from adaptlm.data.pipeline import read_rows, validate


def run(root: Path, output: Path):
    if output.exists():
        raise ValueError("preserve prior classifier experiment; choose a new output")
    validate(root)
    train = [r for r in read_rows(root / "train.jsonl") if r["target"]["primary_issue"]]
    val = [r for r in read_rows(root / "validation.jsonl") if r["target"]["primary_issue"]]
    all_test = read_rows(root / "test.jsonl")
    test = [r for r in all_test if r["target"]["primary_issue"]]
    x = [r["message"] for r in train]
    y = [r["target"]["primary_issue"] for r in train]
    candidates = []
    for c in (0.5, 2.0, 8.0):
        pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c, max_iter=2000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        )
        pipeline.fit(x, y)
        predictions = pipeline.predict([r["message"] for r in val])
        score = f1_score([r["target"]["primary_issue"] for r in val], predictions, average="macro")
        candidates.append((float(score), c, pipeline))
    best = max(candidates, key=lambda t: (t[0], -t[1]))
    with (root / "test-exposures.jsonl").open("a") as file:
        file.write(
            json.dumps(
                environment()
                | {
                    "method": "word-char-tfidf-logistic-regression",
                    "split": "test",
                    "output": str(output),
                    "selected_C": best[1],
                    "example_ids": [r["id"] for r in test],
                }
            )
            + "\n"
        )
    predictions = best[2].predict([r["message"] for r in test])
    gold = [r["target"]["primary_issue"] for r in test]
    errors = [
        {
            "id": row["id"],
            "family_id": row["family_id"],
            "gold": target,
            "prediction": str(pred),
            "message": row["message"],
        }
        for row, target, pred in zip(test, gold, predictions)
        if pred != target
    ]
    report = environment() | {
        "method": "word-char-tfidf-logistic-regression",
        "split": "test",
        "dataset_hash": json.loads((root / "manifest.json").read_text())["dataset_hash"],
        "interpretation": "category-only agreement with unreviewed authored labels",
        "fit_split": "train only; vectorizers not refit on validation/test",
        "fit_ids": [r["id"] for r in train],
        "selected_C": best[1],
        "validation_macro_f1": best[0],
        "validation_candidates": [{"C": c, "macro_f1": score} for score, c, _ in candidates],
        "attempted": len(test),
        "excluded_null_category": len(all_test) - len(test),
        "category_accuracy": float(accuracy_score(gold, predictions)),
        "category_macro_f1": float(f1_score(gold, predictions, average="macro")),
        "per_category": classification_report(gold, predictions, output_dict=True, zero_division=0),
        "confusion_pairs": [
            {"gold": a, "prediction": b, "count": n}
            for (a, b), n in Counter((e["gold"], e["prediction"]) for e in errors).most_common()
        ],
        "errors": errors,
        "complete_record_metrics": "not applicable; classifier only predicts category",
    }
    write_json(output, report)
    predictions_path = output.with_name("classifier-predictions.jsonl")
    predictions_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": r["id"],
                    "family_id": r["family_id"],
                    "prediction": str(p),
                    "gold": r["target"]["primary_issue"],
                }
            )
            + "\n"
            for r, p in zip(test, predictions)
        )
    )
    return report
