"""Structured outputs the LLM must return (validated by Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


def _no_default(schema: dict) -> None:
    """OpenAI strict structured outputs reject `default` in JSON schema; keep the
    Python default but hide it from the schema sent to the model."""
    schema.pop("default", None)

Intent = Literal[
    "product_question",   # asks about products / coverage / premiums / conditions
    "catalog",            # asks which products InsureX offers in general ("มีประกันอะไรบ้าง")
    "purchase_interest",  # wants to buy, apply, or be contacted by an agent
    "provide_info",       # gives personal details (name, job, income, phone) during lead collection
    "confirm",            # yes / confirm / agree (to a pending confirmation)
    "correct",            # wants to change something previously given
    "cancel",             # no longer wants to be contacted / stop lead collection
    "smalltalk",          # greetings, thanks, chit-chat
    "out_of_scope",       # clearly unrelated to insurance (weather, coding, ...)
]


class RouteDecision(BaseModel):
    """Classify the customer's latest message and rewrite it as a standalone search query."""

    intent: Intent = Field(description="Primary intent of the latest customer message.")
    wants_contact: bool = Field(description="True if the customer expresses interest in buying/applying or asks to be contacted.")
    has_question: bool = Field(description="True if the message contains a question about insurance products that should be answered from the knowledge base.")
    products: list[str] = Field(default_factory=list, description="Product ids (from the catalogue) the question is about, resolved using the conversation history. Empty if none/unclear.")
    standalone_query: str = Field(default="", json_schema_extra=_no_default, description="The product question rewritten as a self-contained Thai search query (resolve 'แผนนี้', 'อันนั้น', 'it' from history; include product names). Empty if no question.")
    language: Literal["th", "en"] = Field(description="Language the customer is writing in.")


class RetryQuery(BaseModel):
    """A reformulated search query for a second retrieval attempt."""

    query: str = Field(description="New Thai search query using different wording/synonyms (e.g. ทุนประกัน ↔ จำนวนเงินเอาประกันภัย, ตาย ↔ เสียชีวิต, ค่าเบี้ย ↔ เบี้ยประกันภัย).")


class GradeResult(BaseModel):
    """Which context blocks are useful for answering."""

    relevant_blocks: list[int] = Field(description="Numbers of the context blocks that contain information needed to answer. Empty if none.")
    answerable: bool = Field(description="True if the relevant blocks are sufficient to answer at least the core of the question.")


class GeneratedAnswer(BaseModel):
    answer: str = Field(description="The reply to the customer, in their language, with [n] citations to context blocks.")
    cited_blocks: list[int] = Field(description="Context block numbers actually used.")
    fully_answered: bool = Field(description="False if part of the question could not be answered from the context.")


class GroundingCheck(BaseModel):
    grounded: bool = Field(description="True if every factual claim and number in the answer is supported by the context.")
    unsupported_claims: list[str] = Field(default_factory=list, description="Claims/numbers not supported by the context.")
