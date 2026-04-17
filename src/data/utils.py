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
    'corruptrag_ak_poisoned': os.path.join(_DATA_BASE, 'experiment-datasets', 'nq-corruptrag-ak-poisoning', 'corpus.jsonl'),
}

_title_to_doc_ids_by_corpus_type: dict[str, dict[str, set[str]]] = {}

# Candidate paths for noise filter results (checked in order).
# Local: src/experiments/results/noise/
# Modal volume: /vol/results/noise/
_NOISE_RESULTS_PATHS = [
    os.path.join(_DATA_BASE, '..', 'experiments', 'results', 'noise'),
    '/vol/results/noise',
]


def _load_noise_question_ids() -> set[str]:
    """Load NOISE question IDs from the noise filter results directory.

    Checks local path first, then Modal volume path. Returns only 'full'
    NOISE (not partial). Raises if no results found.
    """
    for candidate in _NOISE_RESULTS_PATHS:
        noise_dir = os.path.normpath(candidate)
        if not os.path.isdir(noise_dir):
            continue

        exclusions = set()
        for fname in os.listdir(noise_dir):
            if not fname.endswith('.json') or fname == 'summary.json':
                continue
            fpath = os.path.join(noise_dir, fname)
            try:
                with open(fpath) as f:
                    r = json.load(f)
                if r.get('is_noise') and r.get('noise_type') == 'full':
                    exclusions.add(r['question_id'])
            except (json.JSONDecodeError, OSError, KeyError):
                continue

        if exclusions:
            return exclusions

    raise FileNotFoundError(
        f"No noise filter results found. Checked: {_NOISE_RESULTS_PATHS}"
    )


# Questions excluded from all judging and metrics because the target answer
# is also a plausible correct answer, making attack success unmeasurable.
# Cached on first call so the file IO only happens once per process, but
# loaded lazily so importing this module doesn't require noise data on disk
# (which keeps unit tests that don't exercise judging import-clean).
_NOISE_QUESTION_IDS_CACHE: set[str] | None = None


def get_noise_question_ids() -> set[str]:
    """Return the set of full-NOISE question IDs, loading + caching on first call."""
    global _NOISE_QUESTION_IDS_CACHE
    if _NOISE_QUESTION_IDS_CACHE is None:
        _NOISE_QUESTION_IDS_CACHE = _load_noise_question_ids()
    return _NOISE_QUESTION_IDS_CACHE


# Testing only — 3 known NOISE IDs from 41-question validation sample:
# NOISE_QUESTION_IDS = {'test3419', 'test2554', 'test2605'}


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


def load_title_to_doc_ids_map(corpus_type: str) -> dict[str, set[str]]:
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
    
    print("Loading title -> doc IDs map...")
    title_map = load_title_to_doc_ids_map('original')
    print(f"  'Minority interest' has {len(title_map['Minority interest'])} passages")
    print(f"  'Chicago Fire (season 4)' has {len(title_map['Chicago Fire (season 4)'])} passages\n")
