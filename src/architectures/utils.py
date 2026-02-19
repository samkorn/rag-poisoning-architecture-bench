from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv; load_dotenv()

from qa_system import STANDARD_PROMPT


def execute_llm_call(
    model_id: str,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    system_prompt: str = STANDARD_PROMPT,
    user_prompt: str = ' ',
    temperature: float = 1.0,
) -> str:
    openai_client = OpenAI()
    params = dict(
        model=model_id,
        instructions=system_prompt,
        input=user_prompt,
        temperature=temperature,
    )
    if reasoning_effort or reasoning_summary:
        params['reasoning'] = {
            k: v for k, v in [
                ('effort', reasoning_effort),
                ('summary', reasoning_summary),
            ] if v is not None
        }
    response = openai_client.responses.create(**params)
    return response.output_text
