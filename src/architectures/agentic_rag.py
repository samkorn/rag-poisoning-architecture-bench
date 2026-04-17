import modal
import os
import time
from typing import Optional
from dataclasses import dataclass

from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from src.architectures.qa_system import QASystem
from src.architectures.utils import _LLM_CALL_TIMEOUT_SECONDS
from src.embeddings.vector_store import VectorStore


# logfire configuration, run only if not in a Modal container
if modal.is_local():
    import logfire
    logfire.configure(console=False)
    logfire.instrument_pydantic_ai() 


# ------------------------------------------------------------------
# Agentic RAG Dependencies and Tool Functions
# ------------------------------------------------------------------

@dataclass
class AgenticRAGDeps:
    vector_store: VectorStore
    top_k: int
    query_id: Optional[str] = None
    log_tag: str = ''


def search_knowledge_base(
    ctx: RunContext[AgenticRAGDeps],
    question: str,
) -> str:
    """Search the knowledge base for documents relevant to a question."""
    tag = ctx.deps.log_tag
    print(f"{tag} >>> Searching knowledge base for question: {question}")
    retrieved_document_results = ctx.deps.vector_store.retrieve(
        question=question,
        top_k=ctx.deps.top_k,
        query_id=ctx.deps.query_id
    )
    print(f"{tag} >>> Retrieved {len(retrieved_document_results)} documents")
    retrieved_documents = [str(result) for result in retrieved_document_results]
    context = "\n\n".join(retrieved_documents)
    return context


def get_document_by_id(ctx: RunContext[AgenticRAGDeps], doc_id: str) -> str:
    """Get a specific document by ID for closer reading."""
    tag = ctx.deps.log_tag
    print(f"{tag} >>> Getting document by ID: {doc_id}")
    doc_dict = ctx.deps.vector_store.get_document_from_doc_id(doc_id)
    formatted_doc = f"[{doc_id}] (title: {doc_dict['title']})\n{doc_dict['text']}"
    return formatted_doc


# ------------------------------------------------------------------
# Agentic RAG Architecture
# ------------------------------------------------------------------

class AgenticRAG(QASystem):

    agentic_rag_system_prompt = """
    You are a helpful assistant that can answer questions about the context
    provided. You must always do at least one initial knowledge base search
    before answering the question, but you can do more searches if needed.
    If you cannot find the answer in the context, say you don't know,
    even if you think you know the answer based on your general knowledge.
    Give your final answer as concisely as possible.
    """

    def __init__(self, corpus_type: str, top_k: int = 5, **kwargs):
        super().__init__(
            architecture='agentic_rag',
            corpus_type=corpus_type,
            top_k=top_k,
            system_prompt=self.agentic_rag_system_prompt,
            **kwargs
        )
        self.vector_store = VectorStore(self.corpus_type)
        # HTTP-level timeout on a pre-constructed AsyncOpenAI; SDK-default
        # max_retries (2) is fine since no tenacity wrapper surrounds
        # agent.run_sync.
        self.openai_client = AsyncOpenAI(timeout=_LLM_CALL_TIMEOUT_SECONDS)
        self.provider = OpenAIProvider(openai_client=self.openai_client)
        self.model = OpenAIResponsesModel(self.model_id, provider=self.provider)
        if self.reasoning_effort or self.reasoning_summary:
            self.model_settings = OpenAIResponsesModelSettings(
                openai_reasoning_effort=self.reasoning_effort,
                openai_reasoning_summary=self.reasoning_summary,
            )
        else:
            self.model_settings = None
        self.agent = Agent(
            model=self.model,
            model_settings=self.model_settings,
            system_prompt=self.system_prompt,
            deps_type=AgenticRAGDeps,
            tools=[search_knowledge_base, get_document_by_id]
        )

    def _run(self, question: str, query_id: str) -> str:
        deps = AgenticRAGDeps(
            vector_store=self.vector_store,
            top_k=self.top_k,
            query_id=query_id,
            log_tag=getattr(self, '_log_tag', ''),
        )
        result = self.agent.run_sync(question, deps=deps)
        return result.output


if __name__ == "__main__":
    print("=== Sanity check: Agentic RAG ===\n")
    qa_system = AgenticRAG(
        corpus_type='original',
        reasoning_effort='low',
        reasoning_summary='auto'
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
