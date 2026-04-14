from typing import Optional, Type, Union

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv; load_dotenv()
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from timeout_decorator import timeout

from qa_system import STANDARD_PROMPT


@retry(
    wait=wait_exponential_jitter(initial=1, max=60), # 1 second initial delay, 60 second max delay
    stop=stop_after_attempt(3), # 3 attempts
    reraise=True, # raise the last exception if all attempts fail
)
def execute_llm_call(
    model_id: str,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    system_prompt: str = STANDARD_PROMPT,
    user_prompt: str = ' ',
    temperature: float = 1.0,
    response_format: Optional[Type[BaseModel]] = None,
    use_signals: bool = False,
    truncation: Optional[str] = None,
) -> Union[str, BaseModel]:
    # use_signals=False (default): multiprocessing-based timeout, works in
    #   non-main threads (Modal workers) but requires pickling — breaks with
    #   sys.path-based imports.
    # use_signals=True: SIGALRM-based timeout, no pickling, but must run on
    #   the main thread (fine for local scripts).
    @timeout(60*3, use_signals=use_signals)
    def _call():
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

    return _call()
