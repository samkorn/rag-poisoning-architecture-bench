import os
import modal
import json


# Globals and constants
app = modal.App(image=modal.Image.debian_slim().pip_install('openai'))
MODEL_ID = 'gpt-5.2-2025-12-11'


# Primary function
correct_answer_prompt = """
Given the following question and reference passages, extract the shortest correct answer to the question. The answer should be a brief factual response — typically a name, date, number, place, or short phrase. Do not explain or elaborate. Respond with only the answer, no other text.

Example:
Question: When was the first manned moon landing?
Reference passages:
Apollo 11 was the spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on July 20, 1969.
The Apollo program was designed to land humans on the Moon and bring them safely back to Earth. It achieved its goal with the Apollo 11 mission, when astronauts Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above in the command module Columbia.
Answer:
July 20, 1969

Now complete this task:
Question: {question}
Reference passages:
{passages}
Answer:
"""

@app.function(secrets=[modal.Secret.from_name('openai-rag-poisoning')])
def craft_correct_answer(question_text: str, document_texts: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    prompt = correct_answer_prompt.format(question=question_text, passages=document_texts)
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{'role': 'user', 'content': prompt}],
        response_format={'type': 'text'},
        temperature=0.0
    )
    return response.choices[0].message.content


@app.local_entrypoint()
def main():
    # Parse original dataset
    print("Parsing NQ dataset...")
    queries: dict[str, str] = {}
    with open('original-datasets/nq/queries.jsonl', 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            queries[line_dict['_id']] = line_dict['text']

    documents: dict[str, dict[str, str]] = {}
    with open('original-datasets/nq/corpus.jsonl', 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            documents[line_dict['_id']] = {'title': line_dict['title'], 'text': line_dict['text']}
            
    query_id_to_document_ids_map: dict[str, set[str]] = {}
    with open('original-datasets/nq/qrels/test.tsv', 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            query_id, document_id, _ = line.split('\t')
            if query_id not in query_id_to_document_ids_map:
                query_id_to_document_ids_map[query_id] = set([document_id])
            else:
                query_id_to_document_ids_map[query_id].add(document_id)
    
    # Create question-documents pairs
    print("Creating question-documents pairs...")
    question_documents_pairs: list[tuple[str, str]] = []
    for query_id in queries.keys():
        question_text = queries[query_id]
        document_ids = query_id_to_document_ids_map[query_id]
        document_texts = '\n'.join([documents[document_id]['text'] for document_id in document_ids])
        question_documents_pairs.append((question_text, document_texts))
    
    # Craft correct answers
    print("Crafting correct answers...")
    correct_answers = list(craft_correct_answer.starmap(question_documents_pairs))
    mapped_correct_answers = list(zip(list(queries.keys()), correct_answers))
    os.makedirs('experiment-datasets', exist_ok=True)
    with open('experiment-datasets/nq-correct-answers.jsonl', 'w') as f:
        for query_id, correct_answer in mapped_correct_answers:
            f.write(json.dumps({'query_id': query_id, 'correct_answer': correct_answer}) + '\n')
    print("Correct answers saved to experiment-datasets/nq-correct-answers.jsonl")
