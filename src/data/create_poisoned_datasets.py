"""Assemble poisoned corpus directories from the original NQ dataset.

Creates two poisoned corpus directories:

1. nq-naive-poisoning/         — naive poisoned docs (from nq-incorrect-answers-poisoned-docs.jsonl)
2. nq-corruptrag-ak-poisoning/ — CorruptRAG-AK poisoned docs (query prepended per p_i = p_i^s + p_i^h)

Each directory is a copy of original-datasets/nq/ with poisoned docs appended to corpus.jsonl.
"""

import json
import os
import shutil


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_NQ_DIR = os.path.join(DATA_DIR, 'original-datasets', 'nq')
EXPERIMENT_DIR = os.path.join(DATA_DIR, 'experiment-datasets')


def _safe_copy_original(output_dir: str) -> None:
    """Copy original NQ dataset to output_dir, archiving any existing copy first."""
    if os.path.exists(output_dir):
        backup_dir = output_dir + '-BACKUP'
        if os.path.exists(backup_dir):
            print(f"Removing old backup at {backup_dir}...")
            shutil.rmtree(backup_dir)
        print(f"Archiving existing {output_dir} -> {backup_dir}...")
        shutil.copytree(output_dir, backup_dir)
        shutil.rmtree(output_dir)
    print(f"Copying {ORIGINAL_NQ_DIR} -> {output_dir}...")
    shutil.copytree(ORIGINAL_NQ_DIR, output_dir)


def main():
    # --- Shared data parsing (used by both naive and CorruptRAG-AK) ---

    # Parse queries
    print("Parsing queries...")
    queries: dict[str, str] = {}
    with open(os.path.join(ORIGINAL_NQ_DIR, 'queries.jsonl'), 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            queries[line_dict['_id']] = line_dict['text']

    # Parse qrels for gold doc title lookup
    print("Parsing qrels...")
    query_id_to_document_ids_map: dict[str, set[str]] = {}
    with open(os.path.join(ORIGINAL_NQ_DIR, 'qrels', 'test.tsv'), 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            query_id, document_id, _ = line.split('\t')
            if query_id not in query_id_to_document_ids_map:
                query_id_to_document_ids_map[query_id] = set([document_id])
            else:
                query_id_to_document_ids_map[query_id].add(document_id)

    # Parse original corpus for doc titles
    print("Parsing original corpus for titles (this takes a moment)...")
    document_titles: dict[str, str] = {}
    with open(os.path.join(ORIGINAL_NQ_DIR, 'corpus.jsonl'), 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            document_titles[line_dict['_id']] = line_dict['title']

    # Get single distinct title per query (from gold docs)
    print("Resolving gold doc titles...")
    distinct_titles_per_query: dict[str, str] = {}
    for query_id in query_id_to_document_ids_map.keys():
        distinct_titles = set(
            document_titles[doc_id]
            for doc_id in query_id_to_document_ids_map[query_id]
        )
        if len(distinct_titles) > 1:
            raise ValueError(f"Query {query_id} has multiple distinct titles: {distinct_titles}")
        distinct_titles_per_query[query_id] = list(distinct_titles)[0]

    # --- Naive poisoning ---

    naive_output_dir = os.path.join(EXPERIMENT_DIR, 'nq-naive-poisoning')

    print("\n=== Building naive poisoned corpus ===")
    _safe_copy_original(naive_output_dir)

    # Parse naive poisoned docs
    poisoned_docs: dict[str, str] = {}
    with open(os.path.join(EXPERIMENT_DIR, 'nq-incorrect-answers-poisoned-docs.jsonl'), 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            poisoned_docs[line_dict['query_id']] = line_dict['poisoned_doc']

    # Build and append naive poisoned docs
    naive_poisoned_lines: list[dict] = []
    for query_id in poisoned_docs.keys():
        title = distinct_titles_per_query[query_id]
        naive_poisoned_lines.append({
            '_id': f'poisoned-naive-q:{query_id}',
            'title': title,
            'text': poisoned_docs[query_id],
            'metadata': {},
        })

    print(f"Appending {len(naive_poisoned_lines)} poisoned docs to corpus...")
    with open(os.path.join(naive_output_dir, 'corpus.jsonl'), 'a') as f:
        for line_dict in naive_poisoned_lines:
            f.write(json.dumps(line_dict) + '\n')

    total = sum(1 for _ in open(os.path.join(naive_output_dir, 'corpus.jsonl'), 'r'))
    print(f"Done. Naive corpus has {total:,} documents (expected ~2,684,920)")

    # --- CorruptRAG-AK poisoning ---

    crak_output_dir = os.path.join(EXPERIMENT_DIR, 'nq-corruptrag-ak-poisoning')
    ak_docs_path = os.path.join(EXPERIMENT_DIR, 'nq-corruptrag-ak-poisoned-docs.jsonl')

    print("\n=== Building CorruptRAG-AK poisoned corpus ===")
    _safe_copy_original(crak_output_dir)

    # Parse CorruptRAG-AK texts
    print("Parsing CorruptRAG-AK texts...")
    ak_texts: dict[str, str] = {}
    with open(ak_docs_path, 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            ak_texts[line_dict['query_id']] = line_dict['corruptrag_ak_text']

    # Build and append poisoned docs
    print("Building poisoned documents...")
    crak_poisoned_lines: list[dict] = []
    for query_id in ak_texts.keys():
        query_text = queries[query_id]
        title = distinct_titles_per_query[query_id]
        # Per CorruptRAG paper: p_i = p_i^s (query for retrieval) ⊕ p_i^h (AK text)
        poisoned_text = f'{query_text} {ak_texts[query_id]}'
        crak_poisoned_lines.append({
            '_id': f'poisoned-corruptrag-ak-q:{query_id}',
            'title': title,
            'text': poisoned_text,
            'metadata': {},
        })

    print(f"Appending {len(crak_poisoned_lines)} poisoned docs to corpus...")
    with open(os.path.join(crak_output_dir, 'corpus.jsonl'), 'a') as f:
        for line_dict in crak_poisoned_lines:
            f.write(json.dumps(line_dict) + '\n')

    # Final count
    total = sum(1 for _ in open(os.path.join(crak_output_dir, 'corpus.jsonl'), 'r'))
    print(f"Done. CorruptRAG-AK corpus has {total:,} documents (expected ~2,684,920)")


if __name__ == '__main__':
    main()
