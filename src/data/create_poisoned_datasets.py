import os
import json


print("Copying datasets to experiment-datasets...")
# copy nq datasets to experiment-datasets, but for naive poisoning
os.system('cp -r original-datasets/nq experiment-datasets/nq-naive-poisoning')

# copy nq datasets to experiment-datasets, but for adversarial poisoning
os.system('cp -r original-datasets/nq experiment-datasets/nq-adversarial-poisoning')


# Parse datasets
print("Parsing datasets...")
queries: dict[str, str] = {}
with open('original-datasets/nq/queries.jsonl', 'r') as f:
    for line in f.readlines():
        line_dict = json.loads(line)
        queries[line_dict['_id']] = line_dict['text']

original_documents: dict[str, dict[str, str]] = {}
with open('original-datasets/nq/corpus.jsonl', 'r') as f:
    for line in f.readlines():
        line_dict = json.loads(line)
        original_documents[line_dict['_id']] = {'title': line_dict['title'], 'text': line_dict['text']}
        
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

incorrect_answers, poisoned_docs = {}, {}
with open('experiment-datasets/nq-incorrect-answers-poisoned-docs.jsonl', 'r') as f:
    for line in f.readlines():
        line_dict = json.loads(line)
        incorrect_answers[line_dict['query_id']] = line_dict['incorrect_answer']
        poisoned_docs[line_dict['query_id']] = line_dict['poisoned_doc']


# get single distinct document title per query
distinct_titles_per_query: dict[str, set[str]] = {}
for query_id in query_id_to_document_ids_map.keys():
    distinct_titles = set(
        original_documents[document_id]['title']
        for document_id in query_id_to_document_ids_map[query_id]
    )
    if len(distinct_titles) > 1:
        raise ValueError(f"Query {query_id} has multiple distinct titles: {distinct_titles}")
    distinct_titles_per_query[query_id] = list(distinct_titles)[0] # get only title


# Create poisoned datasets
print("Creating poisoned datasets...")
naive_poisoned_lines, adversarial_poisoned_lines = [], []
for query_id in poisoned_docs.keys():
    title = distinct_titles_per_query[query_id]
    # naive poisoning
    new_document_id = f'poisoned-naive-q:{query_id}'
    naive_poisoned_doc = poisoned_docs[query_id]
    naive_poisoned_line = {
        '_id': new_document_id,
        'title': title,
        'text': naive_poisoned_doc,
        'metadata': {}
    }
    naive_poisoned_lines.append(naive_poisoned_line)
    # adversarial poisoning
    new_document_id = f'poisoned-adversarial-q:{query_id}'
    query_text = queries[query_id]
    adversarial_poisoned_doc = f"{query_text[0].upper()}{query_text[1:]}? {poisoned_docs[query_id]}"
    adversarial_poisoned_line = {
        '_id': new_document_id,
        'title': title,
        'text': adversarial_poisoned_doc,
        'metadata': {}
    }
    adversarial_poisoned_lines.append(adversarial_poisoned_line)

# Write poisoned datasets
print("Writing poisoned datasets...")
with open('experiment-datasets/nq-naive-poisoning/corpus.jsonl', 'a') as f:
    for line in naive_poisoned_lines:
        f.write(json.dumps(line) + '\n')
with open('experiment-datasets/nq-adversarial-poisoning/corpus.jsonl', 'a') as f:
    for line in adversarial_poisoned_lines:
        f.write(json.dumps(line) + '\n')

print("Poisoned datasets written to experiment-datasets/nq-naive-poisoning/corpus.jsonl and experiment-datasets/nq-adversarial-poisoning/corpus.jsonl")
