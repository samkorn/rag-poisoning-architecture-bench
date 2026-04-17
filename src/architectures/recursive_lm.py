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

    def __init__(self, corpus_type: str, verbose: bool = False, **kwargs):
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
        """Sort key for ordering passages within a single Wikipedia article's title group.

        Original corpus passages ('doc0', 'doc1', ...) are sorted numerically and
        placed first, so the article reads in its natural order.  Poisoned documents
        (e.g. 'poisoned-naive-q:test3') sort after all original passages, numerically
        by trailing query number (relevant if N>1 poisoned docs per article).

        This is used by RLM to reconstruct full article context; other architectures
        retrieve fixed-K passages and don't need ordered reconstruction.
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
        """Collect all passages for each retrieved article title, ordered for reading.

        Groups by title so each Wikipedia article's passages are contiguous,
        then sorts within each group via _doc_id_sort_key (original passages
        in numeric order, poisoned docs at the end).
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
