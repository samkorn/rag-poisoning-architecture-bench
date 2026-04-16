"""Unit tests asserting client-level HTTP timeouts on every architecture.

These tests don't exercise actual timeout behavior (that would require
simulating a hanging API call, which is slow and flaky). Instead, they
verify that each architecture constructs its underlying OpenAI client with
the expected ``timeout`` kwarg, catching regressions where a future edit
silently drops the timeout.

See ``.claude/plans/CLIENT_TIMEOUT_REFACTOR.md`` for the design rationale.
"""

import unittest
from unittest.mock import patch

import openai


EXPECTED_TIMEOUT = 180.0


class ClientTimeoutUnitTests(unittest.TestCase):
    """Each architecture's OpenAI client construction includes timeout=180."""

    def test_vanilla_client_has_timeout(self):
        """execute_llm_call constructs OpenAI() with timeout set."""
        captured: dict = {}

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

        self.assertEqual(captured.get('timeout'), EXPECTED_TIMEOUT)
        self.assertEqual(
            captured.get('max_retries'), 0,
            "Expected max_retries=0 (tenacity owns retries)",
        )

    def test_agentic_client_has_timeout(self):
        """AgenticRAG constructs AsyncOpenAI client with timeout set."""
        captured: dict = {}

        original_init = openai.AsyncOpenAI.__init__

        def capturing_init(self, *args, **kwargs):
            captured['timeout'] = kwargs.get('timeout')
            return original_init(self, *args, **kwargs)

        with patch.object(openai.AsyncOpenAI, '__init__', capturing_init):
            from src.architectures.agentic_rag import AgenticRAG
            AgenticRAG(corpus_type='original')

        self.assertEqual(captured.get('timeout'), EXPECTED_TIMEOUT)

    def test_rlm_client_has_timeout(self):
        """RLM passes timeout via backend_kwargs to the underlying rlm package.

        The rlm package forwards ``backend_kwargs['timeout']`` to
        ``BaseLM.__init__(timeout=...)`` and then to ``openai.OpenAI(timeout=...)``
        — see ``rlm/clients/base_lm.py`` and ``rlm/clients/openai.py``. That
        client is constructed lazily on ``.completion()``, so this test asserts
        the contract at our boundary: ``backend_kwargs`` contains the timeout.
        """
        from src.architectures.recursive_lm import RLM
        rlm = RLM(corpus_type='original', model_id='gpt-5-mini')
        timeout = rlm.rlm.backend_kwargs.get('timeout')
        self.assertEqual(timeout, EXPECTED_TIMEOUT)
