"""Generate LeaseGPT answers and score them with a separate LLM judge."""

import json
import re
from typing import Any

from langchain.schema import HumanMessage, SystemMessage

from eval.catalog import EvalListing
from eval.retrievers import BASELINE, retrieve
from leasegpt.generator import generate_response, setup_leasing_agent
from leasegpt.groq_chat import GROQ_MODEL, ChatGroq


JUDGE_RUBRIC = """Faithfulness (support from retrieved listings):
5 = Every factual claim is directly supported; no unsupported details.
4 = Grounded overall, with one minor unsupported or imprecise detail.
3 = Mostly grounded, but contains a meaningful unsupported claim or omission.
2 = Several claims are unsupported or conflict with the retrieved listings.
1 = Mostly fabricated, contradicted, or not based on the retrieved listings.

Relevance (answering the user's request):
5 = Directly answers the request and addresses all material constraints.
4 = Answers the request but misses one minor constraint or useful detail.
3 = Partially answers; one important constraint is ignored.
2 = Only loosely related to the request.
1 = Off-topic or fails to answer the request."""

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a rental-listing RAG system.
Use only the supplied retrieved listings as evidence. Do not reward plausibility.
Return JSON only with this exact schema:
{"faithfulness": 1, "relevance": 1, "rationale": "one concise explanation"}
Both scores must be integers from 1 through 5."""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Judge response did not contain a JSON object")
        cleaned = match.group(0)
    result = json.loads(cleaned)
    for field in ("faithfulness", "relevance"):
        score = result.get(field)
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"Judge field {field!r} must be an integer from 1 to 5")
    rationale = result.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Judge rationale must be a non-empty string")
    return {
        "faithfulness": result["faithfulness"],
        "relevance": result["relevance"],
        "rationale": rationale.strip(),
    }


def _judge_prompt(query: str, answer: str, contexts: list[str]) -> str:
    context_text = "\n\n--- RETRIEVED LISTING ---\n".join(contexts)
    return f"""RUBRIC
{JUDGE_RUBRIC}

USER QUERY
{query}

RETRIEVED LISTINGS
{context_text}

ANSWER TO SCORE
{answer}
"""


def score_answer(
    judge: ChatGroq, query: str, answer: str, contexts: list[str]
) -> dict[str, Any]:
    """Score one answer, retrying once when the judge returns invalid JSON."""
    prompt = _judge_prompt(query, answer, contexts)
    last_error = None
    for attempt in range(2):
        retry_instruction = (
            "\nYour prior response was invalid. Return only the requested JSON object."
            if attempt
            else ""
        )
        try:
            response = judge(
                [
                    SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=prompt + retry_instruction),
                ]
            )
            return _extract_json(response.content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    return {
        "faithfulness": None,
        "relevance": None,
        "rationale": "",
        "error": last_error or "Unable to parse judge response",
    }


def evaluate_generation(
    query: str,
    store,
    catalog: list[EvalListing],
    api_key: str,
) -> dict[str, Any]:
    """Run the production generation path, then independently judge it."""
    contexts = retrieve(store, catalog, query, BASELINE, k=4)
    agent = setup_leasing_agent(store, api_key)
    answer = generate_response(agent, query)
    judge = ChatGroq(groq_api_key=api_key, temperature=0, model=GROQ_MODEL)
    scores = score_answer(
        judge, query, answer, [result.text for result in contexts]
    )
    return {
        "answer": answer,
        "context_ids": [result.id for result in contexts],
        **scores,
    }
