"""Versioned wire contract. Span existence is not semantic correctness."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "triage-1.0"
SPEC_VERSION = "annotation-1.0"


class Issue(StrEnum):
    duplicate_charge = "duplicate_charge"
    refund_request = "refund_request"
    cancellation_request = "cancellation_request"
    delivery_problem = "delivery_problem"
    damaged_item = "damaged_item"
    account_access = "account_access"
    product_help = "product_help"
    other_support = "other_support"


class EntityType(StrEnum):
    order_id = "order_id"
    transaction_id = "transaction_id"
    email = "email"
    product_name = "product_name"
    plan_name = "plan_name"
    amount = "amount"
    currency = "currency"
    date_expression = "date_expression"


class MissingField(StrEnum):
    order_id = "order_id"
    transaction_id = "transaction_id"
    email = "email"
    product_name = "product_name"
    plan_name = "plan_name"
    issue_details = "issue_details"


REQUIRED: dict[Issue, set[MissingField]] = {
    Issue.duplicate_charge: {MissingField.email, MissingField.transaction_id},
    Issue.refund_request: {MissingField.email, MissingField.transaction_id},
    Issue.cancellation_request: {MissingField.email, MissingField.plan_name},
    Issue.delivery_problem: {MissingField.order_id},
    Issue.damaged_item: {MissingField.order_id},
    Issue.account_access: {MissingField.email},
    Issue.product_help: {MissingField.product_name},
    Issue.other_support: {MissingField.issue_details},
}
QUESTIONS = {
    MissingField.order_id: "What is the order ID?",
    MissingField.transaction_id: "What is the transaction ID?",
    MissingField.email: "Which email is associated with the account?",
    MissingField.product_name: "Which product do you need help with?",
    MissingField.plan_name: "Which subscription plan should be cancelled?",
    MissingField.issue_details: "Which support issue should we address first?",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Span(StrictModel):
    start: int = Field(ge=0, strict=True)
    end: int = Field(ge=1, strict=True)
    text: str = Field(min_length=1, max_length=3000)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("span end must exceed start")
        return self


class Entity(StrictModel):
    type: EntityType
    value: str = Field(min_length=1, max_length=512)
    span: Span

    @model_validator(mode="after")
    def raw_value(self):
        if self.value != self.span.text:
            raise ValueError("v1 preserves raw values: value must equal span.text")
        return self


class MissingInformation(StrictModel):
    field: MissingField
    question: str = Field(min_length=1, max_length=240)


class SummaryClaim(StrictModel):
    text: str = Field(min_length=1, max_length=600)
    spans: list[Span] = Field(min_length=1, max_length=8)


class TriageRecord(StrictModel):
    triage_status: Literal["triaged", "needs_clarification", "out_of_scope"]
    primary_issue: Issue | None
    entities: list[Entity] = Field(max_length=32)
    missing_information: list[MissingInformation] = Field(max_length=6)
    summary_claims: list[SummaryClaim] = Field(max_length=8)

    @model_validator(mode="after")
    def coherent(self):
        if self.triage_status == "triaged" and self.primary_issue is None:
            raise ValueError("triaged requires a primary issue")
        if self.triage_status == "out_of_scope" and (
            self.primary_issue is not None or self.missing_information
        ):
            raise ValueError("out_of_scope cannot prescribe operational questions")
        if self.triage_status == "triaged" and self.missing_information:
            raise ValueError("missing operational fields require needs_clarification")
        fields = [m.field for m in self.missing_information]
        if len(fields) != len(set(fields)):
            raise ValueError("duplicate missing fields")
        entities = [(e.type, e.span.start, e.span.end) for e in self.entities]
        if len(entities) != len(set(entities)):
            raise ValueError("duplicate entity mentions")
        return self

    def validate_grounding(self, message: str) -> "TriageRecord":
        spans = [e.span for e in self.entities] + [
            s for claim in self.summary_claims for s in claim.spans
        ]
        for span in spans:
            if span.end > len(message) or message[span.start : span.end] != span.text:
                raise ValueError("source span does not match original Unicode message")
        return self
