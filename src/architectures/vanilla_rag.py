"""Vanilla RAG: the retrieve-then-generate baseline.

Single retrieval call (`top_k` documents from `VectorStore`), then a
single LLM call with the concatenated context. No tool use, no
debate, no iteration.

Usage:
    python src/architectures/vanilla_rag.py

Output:
    Sanity-check answers printed to stdout for two hand-picked
    questions (one in-corpus, one out-of-corpus). No files written.
"""

import os
import time

from src.architectures.qa_system import QASystem
from src.architectures.utils import execute_llm_call
from src.embeddings.vector_store import VectorStore


class VanillaRAG(QASystem):
    """Single-shot retrieve-then-generate RAG baseline.

    Retrieves `top_k` passages from a `VectorStore` and concatenates
    them as context for one LLM call. No tool use, no debate, no
    iteration — the simplest possible RAG and the experimental
    control against which the other three architectures are compared.
    """

    def __init__(self, corpus_type: str, top_k: int = 5, **kwargs):
        """Initialize a Vanilla RAG instance and load its `VectorStore`.

        Args:
            corpus_type: Which corpus to retrieve from
                (`original`, `naive_poisoned`, `corruptrag_ak_poisoned`).
            top_k: Number of passages to retrieve per question.
                Defaults to 5; the bench runs at `top_k=10`.
            **kwargs: Forwarded to `QASystem.__init__` (model,
                reasoning controls, system prompt, etc.).
        """
        super().__init__(
            architecture='vanilla_rag',
            corpus_type=corpus_type,
            top_k=top_k,
            **kwargs
        )
        self.vector_store = VectorStore(self.corpus_type)

    def _run(self, question: str, query_id: str) -> str:
        """Retrieve top-K passages and answer the question in one LLM call.

        Args:
            question: Natural-language question.
            query_id: NQ test query ID, or `None` for ad-hoc
                questions. Forwarded to `VectorStore.retrieve` so
                the precomputed query-embedding fast path can be
                used when available.

        Returns:
            The model's free-form answer.
        """
        retrieved_document_results = self.vector_store.retrieve(
            question=question,
            top_k=self.top_k,
            query_id=query_id
        )
        retrieved_documents = [result['text'] for result in retrieved_document_results]
        context = '\n\n'.join(retrieved_documents)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        answer = execute_llm_call(
            model_id=self.model_id,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )
        return answer


if __name__ == "__main__":
    print("=== Sanity check: Vanilla RAG ===\n")
    qa_system = VanillaRAG(corpus_type='original')
    
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
