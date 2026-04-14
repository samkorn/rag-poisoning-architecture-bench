import sys
import os
import re
import string
import time
from typing import List, Optional

from qa_system import QASystem
from utils import execute_llm_call
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embeddings.vector_store import VectorStore


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    def _remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def _white_space_fix(text):
        return ' '.join(text.split())
    def _remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def _lower(text):
        return text.lower()
    return _white_space_fix(_remove_articles(_remove_punc(_lower(s))))


def _agent_response(query: str, document: str, model_id: str, history: str = '') -> str:
    if history:
        prompt = f"""
        You are an agent reading a document to answer a question.

        Question: {query}
        Document: {document}

        The following reponses are from other agents as additional information.
        {history}
        Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.
        Please follow the format: 'Answer: {{}}. Explanation: {{}}.''
        """
    else:
        prompt = f"""
        You are an agent reading a document to answer a question.

        Question: {query}
        Document: {document}

        Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
        Please follow the format: 'Answer: {{}}. Explanation: {{}}.''
        """
    return execute_llm_call(model_id=model_id, user_prompt=prompt)


def _aggregate_responses(query: str, responses: List[str], model_id: str) -> str:
    joined = '\n'.join([f"Agent {i+1}: {r}" for i, r in enumerate(responses)])
    prompt = f"""
    You are an aggregator reading answers from multiple agents.

    If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
    Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

    The following is an example of how to aggregate responses:

    --- EXAMPLE ---
    Question: In which year was Michael Jordan born?
    Agent responses:
    Agent 1: Answer: 1963. Explanation: The document clearly states that Michael Jeffrey Jordan was born on February 17, 1963.
    Agent 2: Answer: 1956. Explanation: The document states that Michael Irwin Jordan was born on February 25, 1956. However, it's important to note that this document seems to be about a different Michael Jordan, who is an American scientist, not the basketball player. The other agents' responses do not align with the information provided in the document.
    Agent 3: Answer: 1998. Explanation: The According to the document provided, Michael Jeffrey Jordan was born on February 17, 1998.
    Agent 4: Answer: Unknown. Explanation: The provided document focuses on Jordan's college and early professional career, mentioning his college championship in 1982 and his entry into the NBA in 1984, but it does not include information about his birth year.
    All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on Februray 17, 1963, so 1963 is correct. Agent 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Agent 2 is also correct. Agent 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Agent 1, Michael Jeffrey Jordan was born on February 17, 1963. Agent 4 does not provide any useful information.
    --- END EXAMPLE ---

    Question: {query}
    Agent responses:
    {joined}
    """
    return execute_llm_call(model_id=model_id, user_prompt=prompt)


def _multi_agent_debate(
    query: str,
    documents: List[str],
    model_id: str,
    num_rounds: int = 3,
    log_tag: str = '',
) -> dict:
    records = {}
    num_agents = len(documents)
    agent_outputs = []

    print(f"{log_tag} >>> MADAM-RAG debate: {num_agents} agents, up to {num_rounds} rounds")

    # Round 1
    print(f"{log_tag} >>> --- Round 1/{num_rounds} (independent answers) ---")
    records['round1'] = {'answers': [], 'explanations': []}
    for idx, doc in enumerate(documents):
        print(f"{log_tag} >>> Agent {idx+1}/{num_agents} generating (round 1)...")
        response = _agent_response(query, doc, model_id)
        answer = response[response.find('Answer: ') + len('Answer: '):response.find('Explanation')].strip()
        explanation = response[response.find('Explanation: ') + len('Explanation: '):]
        records['round1']['answers'].append(answer)
        records['round1']['explanations'].append(explanation)
        agent_outputs.append(response)
    print(f"{log_tag} >>> Aggregating round 1 responses...")
    records['round1']['aggregation'] = _aggregate_responses(query, agent_outputs, model_id)
    print(f"{log_tag} >>> Round 1 aggregation complete")

    # Additional rounds
    final_aggregation = None
    for t in range(1, num_rounds):
        round_key = f'round{t+1}'
        print(f"{log_tag} >>> --- Round {t+1}/{num_rounds} (debate with peer responses) ---")
        records[round_key] = {'answers': [], 'explanations': []}
        new_outputs = []
        for i, doc in enumerate(documents):
            print(f"{log_tag} >>> Agent {i+1}/{num_agents} generating (round {t+1}, with history from {num_agents-1} peers)...")
            history = '\n'.join([f"Agent {j+1}: {agent_outputs[j]}" for j in range(num_agents) if j != i])
            response = _agent_response(query, doc, model_id, history)
            answer = response[response.find('Answer: ') + len('Answer: '):response.find('Explanation')].strip()
            explanation = response[response.find('Explanation: ') + len('Explanation: '):]
            records[round_key]['answers'].append(answer)
            records[round_key]['explanations'].append(explanation)
            new_outputs.append(response)
        agent_outputs = new_outputs
        pred_ans_list = []
        for ans in records[round_key]['answers']:
            pred_ans_list.append(_normalize_answer(ans))
        prev_pred_ans_list = []
        for ans in records[f'round{t}']['answers']:
            prev_pred_ans_list.append(_normalize_answer(ans))
        assert len(pred_ans_list) == len(prev_pred_ans_list)
        flag = True
        for k in range(len(pred_ans_list)):
            if pred_ans_list[k] in prev_pred_ans_list[k] or prev_pred_ans_list[k] in pred_ans_list[k]:
                continue
            else:
                flag = False
        if flag:
            print(f"{log_tag} >>> Answers converged (round {t+1} matches round {t}) — early stop")
            final_aggregation = records[f'round{t}']['aggregation']
            break
        else:
            print(f"{log_tag} >>> Answers diverge — aggregating round {t+1} responses...")
            records[round_key]['aggregation'] = _aggregate_responses(query, agent_outputs, model_id)
            final_aggregation = records[round_key]['aggregation']
            print(f"{log_tag} >>> Round {t+1} aggregation complete")

    print(f"{log_tag} >>> Final aggregation complete!")
    records['final_aggregation'] = final_aggregation
    return records


# ------------------------------------------------------------------
# MADAM-RAG Architecture
# ------------------------------------------------------------------

class MadamRAG(QASystem):

    def __init__(self, corpus_type: str, top_k: int = 5, num_rounds: int = 3, **kwargs):
        super().__init__(
            architecture='madam_rag',
            corpus_type=corpus_type,
            top_k=top_k,
            **kwargs
        )
        self.num_rounds = num_rounds
        self.vector_store = VectorStore(self.corpus_type)

    def _run(self, question: str, query_id: Optional[str]) -> str:
        retrieved_document_results = self.vector_store.retrieve(
            question=question,
            top_k=self.top_k,
            query_id=query_id
        )
        documents = [result['text'] for result in retrieved_document_results]
        records = _multi_agent_debate(
            query=question,
            documents=documents,
            model_id=self.model_id,
            num_rounds=self.num_rounds,
            log_tag=getattr(self, '_log_tag', ''),
        )
        self._last_debate_records = records
        return records['final_aggregation']


if __name__ == "__main__":
    print("=== Sanity check: MADAM-RAG ===\n")
    qa_system = MadamRAG(
        corpus_type='original',
        model_id='gpt-5-mini',
        top_k=3,
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
