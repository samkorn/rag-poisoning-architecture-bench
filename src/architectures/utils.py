import threading
from typing import Optional, Type, Union

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv; load_dotenv()
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from timeout_decorator import timeout

from src.architectures.qa_system import STANDARD_PROMPT


def _execute_llm_call(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    reasoning_effort: Optional[str],
    reasoning_summary: Optional[str],
    response_format: Optional[Type[BaseModel]],
    truncation: Optional[str],
) -> Union[str, BaseModel]:
    openai_client = OpenAI()
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


# Pre-build both timeout flavors at module load so retry-wraps-timeout ordering
# is preserved and we avoid per-call wrapping overhead. Auto-dispatch at call
# time based on thread context.
#   use_signals=True:  SIGALRM-based, main-thread only. Used locally.
#   use_signals=False: multiprocessing-based, works in non-main threads (Modal
#                      workers) but requires picklable args.
_execute_llm_call_signal_timed = timeout(60*3, use_signals=True)(_execute_llm_call)
_execute_llm_call_thread_timed = timeout(60*3, use_signals=False)(_execute_llm_call)


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
    _execute_llm_call_timed = (
        _execute_llm_call_signal_timed
        if threading.current_thread() is threading.main_thread()
        else _execute_llm_call_thread_timed
    )
    return _execute_llm_call_timed(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        response_format=response_format,
        truncation=truncation,
    )
