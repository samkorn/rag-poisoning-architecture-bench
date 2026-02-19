import sys
import os
import time

from rlm import RLM as RLMClass

from qa_system import QASystem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embeddings.vector_store import VectorStore
from data.utils import (
    get_query_id_from_question,
    preload_title_to_doc_ids_map,
    get_all_relevant_doc_ids_for_retrieved_docs,
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
        preload_title_to_doc_ids_map(self.corpus_type)
        self.vector_store = VectorStore(self.corpus_type)
        self.rlm = RLMClass(
            backend='openai',
            backend_kwargs={'model_name': self.model_id},
            verbose=verbose
        )

    def _run(self, question: str, query_id: str) -> str:
        retrieved_document_results = self.vector_store.retrieve(
            question=question,
            top_k=self.top_k,
            query_id=query_id
        )
        all_relevant_doc_ids = get_all_relevant_doc_ids_for_retrieved_docs(
            corpus_type=self.corpus_type,
            retrieved_docs=retrieved_document_results
        )
        all_relevant_docs = [
            self.vector_store.get_document_from_doc_id(doc_id)
            for doc_id in all_relevant_doc_ids
        ]
        context = "\n\n".join(doc['text'] for doc in all_relevant_docs)
        user_prompt = f"Question: {question}\n\nContext: {context}"
        answer = self.rlm.completion(user_prompt).response
        return answer


if __name__ == "__main__":
    print("=== Sanity check: RLM ===\n")
    qa_system = RLM(corpus_type='original', model_id='gpt-5-nano', verbose=True)
    
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
