"""Shared data utilities — query lookup, dataset paths, etc."""

import os
import json
import time
from typing import Literal


_DATA_BASE = os.path.dirname(__file__)
_QUERIES_PATH = os.path.join(_DATA_BASE, 'original-datasets', 'nq', 'queries.jsonl')
_QRELS_PATH = os.path.join(_DATA_BASE, 'original-datasets', 'nq', 'qrels', 'test.tsv')
_CORPUS_PATHS: dict[str, str] = {
    'original': os.path.join(_DATA_BASE, 'original-datasets', 'nq', 'corpus.jsonl'),
    'naive_poisoned': os.path.join(_DATA_BASE, 'experiment-datasets', 'nq-naive-poisoning', 'corpus.jsonl'),
    'adversarial_poisoned': os.path.join(_DATA_BASE, 'experiment-datasets', 'nq-adversarial-poisoning', 'corpus.jsonl'),
}

_title_to_doc_ids_by_corpus_type: dict[str, dict[str, set[str]]] = {}


def get_question_from_query_id(query_id: str) -> str:
    with open(_QUERIES_PATH, 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            if line_dict['_id'] == query_id:
                return line_dict['text']
    raise ValueError(f"Query ID '{query_id}' not found in queries")


def get_query_id_from_question(question: str) -> str:
    with open(_QUERIES_PATH, 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            if line_dict['text'] == question:
                return line_dict['_id']
    raise ValueError(f"Question '{question}' not found in queries")


def preload_title_to_doc_ids_map(corpus_type: str) -> None:
    global _title_to_doc_ids_by_corpus_type
    if corpus_type not in (_CORPUS_PATHS.keys()):
        raise ValueError(f"Corpus type '{corpus_type}' must be one of {_CORPUS_PATHS.keys()}")
    _load_title_to_doc_ids_map(corpus_type)


def get_all_relevant_doc_ids_for_retrieved_docs(
    corpus_type: str,
    retrieved_docs: list[dict[str, str]]
) -> list[dict[str, str]]:
    retrieved_doc_titles = set(doc['title'] for doc in retrieved_docs)
    title_to_doc_ids_map = _load_title_to_doc_ids_map(corpus_type)
    relevant_doc_ids = set()
    for title in retrieved_doc_titles:
        doc_ids_for_title = title_to_doc_ids_map[title]
        relevant_doc_ids.update(doc_ids_for_title)
    return list(relevant_doc_ids)


def _load_title_to_doc_ids_map(corpus_type: str) -> dict[str, set[str]]:
    global _title_to_doc_ids_by_corpus_type
    if corpus_type not in _title_to_doc_ids_by_corpus_type:
        print(f"Loading title -> doc IDs map for corpus type: {corpus_type}...")
        t0 = time.time()
        _title_to_doc_ids_by_corpus_type[corpus_type] = {}
        with open(_CORPUS_PATHS[corpus_type], 'r') as f:
            for line in f.readlines():
                line_dict = json.loads(line)
                doc_id, doc_title = line_dict['_id'], line_dict['title']
                if doc_title not in _title_to_doc_ids_by_corpus_type[corpus_type]:
                    _title_to_doc_ids_by_corpus_type[corpus_type][doc_title] = set([doc_id])
                else:
                    _title_to_doc_ids_by_corpus_type[corpus_type][doc_title].add(doc_id)
        print(f"  Title -> doc IDs map loaded: {len(_title_to_doc_ids_by_corpus_type[corpus_type]):,} titles ({time.time() - t0:.1f}s)\n")
    return _title_to_doc_ids_by_corpus_type[corpus_type]



if __name__ == "__main__":
    print("=== Sanity check: data utilities ===\n")

    print("Getting question from query ID...")
    t0 = time.time()
    get_question_from_query_id('test3443')
    print(f"Time taken: {time.time() - t0:.3f}s\n")
    
    print("Getting query ID from question...")
    t0 = time.time()
    get_query_id_from_question("where does junior want to go to find hope")
    print(f"Time taken: {time.time() - t0:.3f}s\n")
    
    print("Preloading title -> doc IDs map...")
    preload_title_to_doc_ids_map('original')
    print()

    print("Getting all relevant doc IDs for retrieved docs...")
    t0 = time.time()
    retrieved_docs = [
        {'title': "Minority interest"},
        {'title': "Chicago Fire (season 4)"},
    ]
    get_all_relevant_doc_ids_for_retrieved_docs('original', retrieved_docs)
    print(f"Time taken: {time.time() - t0:.3f}s\n")