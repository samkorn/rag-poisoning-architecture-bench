"""Recursive Language Model (RLM) architecture: full topic-context QA.

Wraps the upstream `rlm` package to give the model the entire
Wikipedia article (all passages sharing a `title`) for the gold doc,
rather than the top-K nearest neighbors. The intent is to exercise
the model's ability to handle long contexts without retrieval acting
as a filter.

Notes:
    `top_k` is fixed at 100 (a deliberately oversized window) and
    cannot be overridden by callers — passing `top_k=...` raises.
    Title-grouped passages are sorted by `_doc_id_sort_key` so the
    model sees them in original-corpus order. Per-call timeout flows
    through `rlm` -> `OpenAIClient` -> `BaseLM` -> `openai.OpenAI`.
"""

import re
import os
import time

from rlm import RLM as RLMClass

from src.architectures.qa_system import QASystem, STANDARD_PROMPT
from src.architectures.utils import _LLM_CALL_TIMEOUT_SECONDS
from src.embeddings.vector_store import VectorStore
from src.data.utils import (
    get_query_id_from_question,
    load_title_to_doc_ids_map,
)


TOP_K = 100 # huge context window to show RLM's ability to see the full context


class RLM(QASystem):
    """Recursive Language Model architecture: full topic-context QA.

    Wraps the upstream `rlm` package. Instead of retrieving the
    top-K nearest passages, RLM identifies which Wikipedia articles
    a top-K retrieval surfaces and then loads *every* passage from
    those articles, concatenated in original-corpus order. The
    intent is to test the model's ability to handle long contexts
    without retrieval acting as a filter.

    Attributes:
        vector_store: Shared `VectorStore` instance used for the
            initial top-K retrieval before article expansion.
        rlm: Upstream `RLM` client (`rlm.RLM`) configured with the
            OpenAI backend, the chosen model ID, and the shared
            per-call HTTP timeout.
    """

    def __init__(self, corpus_type: str, verbose: bool = False, **kwargs):
        """Initialize an RLM instance and its underlying `rlm` client.

        Preloads the title→doc-id map into the module-level cache in
        `src.data.utils` so subsequent `_run` calls return the
        cached dict instantly.

        Args:
            corpus_type: Which corpus to retrieve from
                (`original`, `naive_poisoned`, `corruptrag_ak_poisoned`).
            verbose: Forwarded to the upstream `RLMClass`. When
                `True`, the `rlm` package logs each recursive call.
            **kwargs: Forwarded to `QASystem.__init__`. Must NOT
                include `top_k` — RLM hard-fixes it (see Raises).

        Raises:
            ValueError: If `top_k` is passed via `**kwargs`. RLM
                fixes the retrieval window at `TOP_K = 100` to give
                title grouping a wide enough net; smaller K can
                miss the gold article entirely.

        Notes:
            Per-call timeout flows through `rlm` -> `OpenAIClient`
            -> `BaseLM` -> `openai.OpenAI`. No `tenacity` wrapper
            surrounds `rlm.completion`, so the OpenAI SDK's default
            `max_retries=2` provides transient 429/5xx protection.
        """
        if 'top_k' in kwargs:
            raise ValueError("top_k is not allowed to be set for RLM")
        super().__init__(
            architecture='rlm',
            corpus_type=corpus_type,
            top_k=TOP_K,
            **kwargs
        )
        # Preload into the module-level cache in data.utils;
        # subsequent calls return the cached dict instantly
        load_title_to_doc_ids_map(self.corpus_type)
        self.vector_store = VectorStore(self.corpus_type)
        # timeout flows through rlm -> OpenAIClient -> BaseLM -> openai.OpenAI.
        # No tenacity wrapper surrounds rlm.completion(), so SDK-default
        # max_retries (2) is fine — leaves transient 429/5xx protection.
        self.rlm = RLMClass(
            backend='openai',
            backend_kwargs={
                'model_name': self.model_id,
                'timeout': _LLM_CALL_TIMEOUT_SECONDS,
            },
            verbose=verbose
        )

    @staticmethod
    def _doc_id_sort_key(doc_id: str) -> tuple:
        """Build a sort key for article passages with poisoned docs last.

        Original passages (`doc0`, `doc1`, ...) sort first by their
        trailing integer, so the article reads in natural order.
        Poisoned documents (e.g. `poisoned-naive-q:test3`) sort
        after all originals, by trailing query number — this matters
        only if multiple poisoned docs target the same article
        (N > 1), which the bench doesn't currently use.

        Args:
            doc_id: Corpus document identifier.

        Returns:
            A `(group, ordinal)` tuple. `group=0` for original
            passages, `group=1` for poisoned passages.

        Raises:
            ValueError: If `doc_id` matches neither the original
                nor poisoned naming pattern.
        """
        m = re.match(r'^doc(\d+)$', doc_id)
        if m:
            return (0, int(m.group(1)))
        elif m := re.search(r'(\d+)$', doc_id):
            return (1, int(m.group(1)))
        else:
            raise ValueError(f"Unexpected doc_id format: '{doc_id}'")

    def _get_all_relevant_doc_ids_for_retrieved_docs(
        self,
        retrieved_docs: list[dict[str, str]]
    ) -> list[str]:
        """Collect every passage for each retrieved article title.

        Groups by title so each Wikipedia article's passages are
        contiguous, then sorts within each group via
        `_doc_id_sort_key`. Used to reconstruct full article context
        for the recursive language model.

        Args:
            retrieved_docs: Documents returned by
                `VectorStore.retrieve`. Only the `title` field is
                consulted.

        Returns:
            Flat list of doc IDs covering every passage of every
            article surfaced by retrieval, in reading order.
        """
        title_to_doc_ids = load_title_to_doc_ids_map(self.corpus_type)
        retrieved_doc_titles = set(doc['title'] for doc in retrieved_docs)
        result = []
        for title in sorted(retrieved_doc_titles):
            doc_ids_for_title = sorted(
                title_to_doc_ids[title], key=self._doc_id_sort_key
            )
            result.extend(doc_ids_for_title)
        return result

    def _run(self, question: str, query_id: str) -> str:
        """Retrieve top-K passages, expand to full articles, run RLM.

        Args:
            question: Natural-language question.
            query_id: NQ test query ID, or `None` for ad-hoc
                questions. Forwarded to `VectorStore.retrieve` so
                the precomputed query-embedding fast path can be
                used when available.

        Returns:
            The RLM's free-form answer.
        """
        retrieved_document_results = self.vector_store.retrieve(
            question=question,
            top_k=self.top_k,
            query_id=query_id
        )
        all_relevant_doc_ids = self._get_all_relevant_doc_ids_for_retrieved_docs(
            retrieved_docs=retrieved_document_results
        )
        all_relevant_docs = [
            self.vector_store.get_document_from_doc_id(doc_id)
            for doc_id in all_relevant_doc_ids
        ]
        context = '\n'.join(doc['text'] for doc in all_relevant_docs)
        prompt = f"Instructions: {STANDARD_PROMPT.strip()}\n\nContext: {context}"
        answer = self.rlm.completion(prompt, root_prompt=question).response
        return answer


if __name__ == "__main__":
    print("=== Sanity check: RLM ===\n")
    qa_system = RLM(
        corpus_type='original',
        model_id='gpt-5-mini',
        verbose=True
    )
    
    # test a real question that's in the corpus
    question1 = "What is the capital of France?"
    print(f"QUESTION 1: {question1}")
    t0 = time.time()
    answer1 = qa_system.run(question=question1)
    print(f"ANSWER 1: {answer1}")
    print(f"(Time taken: {time.time() - t0:.2f} seconds)\n")
    
    # test a real question that's not in the corpus
    question2 = "Who did Luigi Mangione assassinate?"
    print(f"QUESTION 2: {question2}")
    t0 = time.time()
    answer2 = qa_system.run(question=question2)
    print(f"ANSWER 2: {answer2}")
    print(f"(Time taken: {time.time() - t0:.2f} seconds)\n")
    
    # test a nonsensical question that's not in the corpus
    question3 = "Who was the Grand Vizier of Shmorgasborgistan in 1742?"
    print(f"QUESTION 3: {question3}")
    t0 = time.time()
    answer3 = qa_system.run(question=question3)
    print(f"ANSWER 3: {answer3}")
    print(f"(Time taken: {time.time() - t0:.2f} seconds)\n")

    # exit without waiting for large variables to be garbage collected
    os._exit(0)
