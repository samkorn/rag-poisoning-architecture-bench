"""
Structural smoke tests for client-level HTTP timeouts.

These tests don't exercise actual timeout behavior (that would require
simulating a hanging API call, which is slow and flaky). Instead, they verify
that each architecture constructs its underlying OpenAI client with the
expected ``timeout`` kwarg, catching regressions where a future edit silently
drops the timeout.

Run from repo root:
    venv/bin/python tests/test_timeouts.py
"""

import os
from unittest.mock import patch

import openai


EXPECTED_TIMEOUT = 180.0


def test_vanilla_client_has_timeout():
    """execute_llm_call constructs OpenAI() with timeout set."""
    print("\n=== test_vanilla_client_has_timeout ===")
    captured = {}

    def capturing_init(self, *args, **kwargs):
        captured['timeout'] = kwargs.get('timeout')
        captured['max_retries'] = kwargs.get('max_retries')
        # Short-circuit so we don't actually make a request.
        raise RuntimeError("short-circuit: client captured")

    with patch.object(openai.OpenAI, '__init__', capturing_init):
        from src.architectures.utils import execute_llm_call
        try:
            execute_llm_call(model_id='gpt-5-mini', user_prompt='hi')
        except RuntimeError:
            pass

    assert captured.get('timeout') == EXPECTED_TIMEOUT, (
        f"Expected timeout={EXPECTED_TIMEOUT}, got {captured.get('timeout')}"
    )
    assert captured.get('max_retries') == 0, (
        f"Expected max_retries=0 (tenacity owns retries), "
        f"got {captured.get('max_retries')}"
    )
    print(f"  timeout={captured['timeout']}, max_retries={captured['max_retries']}")
    print("  PASSED")


def test_agentic_client_has_timeout():
    """AgenticRAG constructs AsyncOpenAI client with timeout set."""
    print("\n=== test_agentic_client_has_timeout ===")
    captured = {}

    original_init = openai.AsyncOpenAI.__init__

    def capturing_init(self, *args, **kwargs):
        captured['timeout'] = kwargs.get('timeout')
        return original_init(self, *args, **kwargs)

    with patch.object(openai.AsyncOpenAI, '__init__', capturing_init):
        from src.architectures.agentic_rag import AgenticRAG
        AgenticRAG(corpus_type='original')

    assert captured.get('timeout') == EXPECTED_TIMEOUT, (
        f"Expected timeout={EXPECTED_TIMEOUT}, got {captured.get('timeout')}"
    )
    print(f"  timeout={captured['timeout']}")
    print("  PASSED")


def test_rlm_client_has_timeout():
    """RLM passes timeout via backend_kwargs to the underlying rlm package.

    The rlm package forwards ``backend_kwargs['timeout']`` to
    ``BaseLM.__init__(timeout=...)`` and then to ``openai.OpenAI(timeout=...)``
    — see ``rlm/clients/base_lm.py`` and ``rlm/clients/openai.py``. That
    client is constructed lazily on ``.completion()``, so this test asserts
    the contract at our boundary: ``backend_kwargs`` contains the timeout.
    """
    print("\n=== test_rlm_client_has_timeout ===")
    from src.architectures.recursive_lm import RLM
    rlm = RLM(corpus_type='original', model_id='gpt-5-mini')
    timeout = rlm.rlm.backend_kwargs.get('timeout')
    assert timeout == EXPECTED_TIMEOUT, (
        f"Expected backend_kwargs['timeout']={EXPECTED_TIMEOUT}, got {timeout}"
    )
    print(f"  backend_kwargs['timeout']={timeout}")
    print("  PASSED")


if __name__ == '__main__':
    test_vanilla_client_has_timeout()
    test_agentic_client_has_timeout()
    test_rlm_client_has_timeout()
    print("\n=== ALL TESTS PASSED ===")
    os._exit(0)
