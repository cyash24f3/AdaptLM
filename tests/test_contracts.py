import pytest
from pydantic import ValidationError

from adaptlm.contracts.schema import Entity, Span, TriageRecord


def record(message="Charged twice."):
    return {
        "triage_status": "needs_clarification",
        "primary_issue": "duplicate_charge",
        "entities": [],
        "missing_information": [{"field": "email", "question": "Which email?"}],
        "summary_claims": [
            {"text": message, "spans": [{"start": 0, "end": len(message), "text": message}]}
        ],
    }


@pytest.mark.parametrize("issue", ["urgent", "refund", "", 4])
def test_unknown_category(issue):
    value = record()
    value["primary_issue"] = issue
    with pytest.raises(ValidationError):
        TriageRecord.model_validate(value)


def test_operational_missing_retains_category():
    value = TriageRecord.model_validate(record())
    assert value.primary_issue == "duplicate_charge"
    assert value.triage_status == "needs_clarification"


def test_ambiguous_can_remain_unresolved():
    value = record()
    value["primary_issue"] = None
    assert TriageRecord.model_validate(value).primary_issue is None


def test_triaged_requires_issue_and_no_missing():
    value = record()
    value["triage_status"] = "triaged"
    with pytest.raises(ValueError):
        TriageRecord.model_validate(value)
    value["missing_information"] = []
    value["primary_issue"] = None
    with pytest.raises(ValueError):
        TriageRecord.model_validate(value)


@pytest.mark.parametrize(
    "message", ["🙂 café café ORD-１２３", "अनुरोध नमस्ते", "é é é", "🧑‍💻 logged out"]
)
def test_unicode_codepoints(message):
    TriageRecord.model_validate(record(message)).validate_grounding(message)
    with pytest.raises(ValueError):
        TriageRecord.model_validate(record(message)).validate_grounding("x" + message)


def test_repeated_mentions_remain_distinct():
    message = "ORD-123 then ORD-123"
    value = record(message)
    value["entities"] = [
        {
            "type": "order_id",
            "value": "ORD-123",
            "span": {"start": start, "end": start + 7, "text": "ORD-123"},
        }
        for start in (0, 13)
    ]
    assert len(TriageRecord.model_validate(value).validate_grounding(message).entities) == 2


def test_relative_date_preserved_and_normalization_rejected():
    entity = Entity(
        type="date_expression", value="yesterday", span=Span(start=0, end=9, text="yesterday")
    )
    assert entity.value == "yesterday"
    data = entity.model_dump()
    data["value"] = "2026-09-29"
    with pytest.raises(ValueError):
        Entity.model_validate(data)


def test_bad_quotes_and_duplicate_fields():
    data = record()
    data["summary_claims"][0]["spans"][0]["text"] = "refund approved"
    with pytest.raises(ValueError):
        TriageRecord.model_validate(data).validate_grounding("Charged twice.")
    data = record()
    data["missing_information"] *= 2
    with pytest.raises(ValueError):
        TriageRecord.model_validate(data)


def test_schema_valid_can_still_be_semantically_wrong():
    data = record()
    data["primary_issue"] = "product_help"
    data["summary_claims"][0]["text"] = "Refund approved."
    # These deliberate semantic errors pass structural checks; evaluator must not call this correct.
    assert TriageRecord.model_validate(data).validate_grounding("Charged twice.")


@pytest.mark.parametrize("start,end", [(0, 0), (-1, 3), (4, 3), (True, 3)])
def test_span_bounds(start, end):
    with pytest.raises(ValueError):
        Span(start=start, end=end, text="abc")
