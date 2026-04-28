"""Shared LLM-call helpers used by Vanilla RAG and MADAM-RAG.

Exposes `execute_llm_call` (a tenacity-retried wrapper around the
OpenAI Responses API supporting structured outputs and reasoning
controls) and `_LLM_CALL_TIMEOUT_SECONDS`, the per-call HTTP timeout
shared across architectures.

Notes:
    Agentic RAG and RLM construct their own OpenAI clients through
    different paths (PydanticAI for Agentic, the `rlm` package for
    RLM) but both import `_LLM_CALL_TIMEOUT_SECONDS` so all four
    architectures use the same per-call ceiling. The OpenAI client's
    own retries are disabled (`max_retries=0`) so tenacity owns
    retry/backoff exclusively.
"""

from typing import Optional, Type, Union

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv; load_dotenv()
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from src.architectures.qa_system import STANDARD_PROMPT


# Per-call HTTP timeout applied at the OpenAI client layer. Bounds a single API
# call, not the surrounding Python callable — the outer @timeout in
# experiment.py still bounds the full per-question budget. Shared across
# architectures: imported by agentic_rag.py and recursive_lm.py so all four
# architectures use the same per-call ceiling.
_LLM_CALL_TIMEOUT_SECONDS: float = 180.0


@retry(
    wait=wait_exponential_jitter(initial=1, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def execute_llm_call(
    model_id: str,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    system_prompt: str = STANDARD_PROMPT,
    user_prompt: str = ' ',
    temperature: float = 1.0,
    response_format: Optional[Type[BaseModel]] = None,
    truncation: Optional[str] = None,
) -> Union[str, BaseModel]:
    """Call the OpenAI Responses API with shared timeout, retry, and reasoning controls.

    Builds a fresh `OpenAI` client per call (max_retries=0 so tenacity
    owns all retries) and routes the request through either
    `responses.create` or `responses.parse` depending on whether a
    `response_format` was supplied.

    Args:
        model_id: OpenAI model identifier.
        reasoning_effort: Optional reasoning-effort level
            (`low`, `medium`, `high`). Passed through to the
            `reasoning.effort` field on the request.
        reasoning_summary: Optional reasoning-summary level passed
            through to the `reasoning.summary` field.
        system_prompt: System-level instructions. Defaults to the
            shared `STANDARD_PROMPT`.
        user_prompt: User message body.
        temperature: Sampling temperature. Defaults to 1.0.
        response_format: Optional Pydantic model. When provided,
            `responses.parse` is used and the parsed model instance
            is returned; otherwise `responses.create` is used and
            the raw output text is returned.
        truncation: Optional truncation strategy passed through to
            the API (`auto`, `disabled`).

    Returns:
        The raw output text (`str`) when no `response_format` is
        provided, or a parsed instance of `response_format` when one
        is.

    Raises:
        ValueError: When `response_format` is provided and the
            model refuses to emit structured output.

    Notes:
        Wrapped by `tenacity` for exponential-jitter backoff (3
        attempts max). The retry wrapper re-raises the underlying
        exception once attempts are exhausted, so callers see the
        same exception type they would without retries.
    """
    # max_retries=0: let tenacity own retries (avoid SDK retries stacking
    # underneath the @retry wrapper).
    openai_client = OpenAI(timeout=_LLM_CALL_TIMEOUT_SECONDS, max_retries=0)
    params = dict(
        model=model_id,
        instructions=system_prompt,
        input=user_prompt,
        temperature=temperature,
    )
    if truncation is not None:
        params['truncation'] = truncation
    if reasoning_effort or reasoning_summary:
        params['reasoning'] = {
            k: v for k, v in [
                ('effort', reasoning_effort),
                ('summary', reasoning_summary),
            ] if v is not None
        }
    if response_format is not None:
        # structured output parsing
        params['text_format'] = response_format
        response = openai_client.responses.parse(**params)
        if response.output_parsed is None:
            refusal = response.refusal
            raise ValueError(f"Model refused structured output: {refusal}")
        return response.output_parsed
    else:
        # regular response
        response = openai_client.responses.create(**params)
        return response.output_text
