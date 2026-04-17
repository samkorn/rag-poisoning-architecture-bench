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
