from abc import ABC, abstractmethod
from typing import Optional

from src.data.utils import get_question_from_query_id


STANDARD_PROMPT = """
You are a helpful assistant that can answer questions about the context
provided. If the context does not contain the answer, say you don't know,
even if you think you know the answer based on your general knowledge. Give your
final answer as concisely as possible.
"""


class QASystem(ABC):

    def __init__(
        self,
        architecture: str,
        corpus_type: str,
        model_id: str = 'gpt-5-mini',
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        system_prompt: str = STANDARD_PROMPT,
        top_k: Optional[int] = None,
    ):
        self.corpus_type = corpus_type
        self.architecture = architecture
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.system_prompt = system_prompt
        self.top_k = top_k

    def run(
        self,
        query_id: Optional[str] = None,
        question: Optional[str] = None,
    ) -> str:
        """Run the architecture with either a query ID or a question."""
        if query_id is not None:
            return self.run_with_query_id(query_id)
        elif question is not None:
            return self.run_with_question(question)
        else:
            raise ValueError("Either query_id or question must be provided")    

    def run_with_question(self, question: str) -> str:
        """Run the architecture with a question.
        
        This would be the default scenario when running the system outside
        of the context of an experiment.
        """
        return self._run(question, None)

    def run_with_query_id(self, query_id: str) -> str:
        """Run the architecture with a query ID.
        
        This would be the scenario when running the system in the context of an
        experiment, where the question is provided as a query ID, and the query
        ID metadata should be tracked.
        """
        question = get_question_from_query_id(query_id)
        return self._run(question, query_id)

    @abstractmethod
    def _run(self, question: str, query_id: Optional[str]) -> str:
        """Run the architecture with a question and a query ID.
        
        This is the abstract method that subclasses must implement.
        """
        raise NotImplementedError("Subclasses must implement this method")
