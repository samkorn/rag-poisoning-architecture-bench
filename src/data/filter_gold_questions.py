"""
Filter nq-questions.jsonl to queries where at least one gold-standard
document appears in top-10 clean retrieval.

Uses the same on-disk artifacts as ``src.embeddings.vector_store`` (query
embedding pickle, ``nq-original.faiss``, ``nq-original-doc-ids.pkl``) so
retrieval scores/ranks match experiment-time retrieval.

Why this script does not call ``VectorStore``:

- **Batch FAISS**: ``VectorStore.retrieve`` runs ``index.search`` on one query
  at a time. Here we stack all query vectors and call ``search`` once (or in
  few large blocks if you later chunk for memory). For thousands of
  queries that is orders of magnitude faster than per-query retrieval.
- **No corpus or embedder**: We only need neighbor *indices* mapped to doc
  IDs, then a set intersection with ``gold_doc_ids``. We never return passage
  text. ``VectorStore`` loads the full corpus into memory and constructs an
  ``Embedder`` (Torch) even when using precomputed query embeddings; that is
  wasted work and RAM for this offline filter.

Flow: load queries and embeddings → L2-normalize query matrix (same as
``VectorStore.retrieve``) → batch ``search`` → keep rows where top-K doc IDs
hit ``gold_doc_ids``.

Output:
    experiment-datasets/nq-questions-gold-filtered.jsonl

Usage:
    python src/data/filter_gold_questions.py
"""

import json
import os
import pickle
import time

import faiss
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(_DATA_DIR, 'vector-store')
# Filenames keep the legacy "nq-questions" prefix (the value is a list of
# query records — id + question text + answers + gold_doc_ids — but the
# files are referenced from analysis notebooks, paper tables, and
# upstream BEIR conventions, so the on-disk names are frozen).
QUERIES_PATH = os.path.join(_DATA_DIR, 'experiment-datasets', 'nq-questions.jsonl')
OUTPUT_PATH = os.path.join(_DATA_DIR, 'experiment-datasets', 'nq-questions-gold-filtered.jsonl')

TOP_K = 10


def main():
    t_start = time.time()

    # --- Load queries --------------------------------------------------------
    queries: list[dict] = []
    with open(QUERIES_PATH) as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"Loaded {len(queries):,} queries")

    # --- Load query embeddings -----------------------------------------------
    emb_path = os.path.join(VECTOR_STORE_DIR, 'nq-queries-embeddings.pkl')
    with open(emb_path, 'rb') as f:
        query_embeddings: dict[str, np.ndarray] = pickle.load(f)
    print(f"Loaded {len(query_embeddings):,} query embeddings")

    # --- Load FAISS index + doc-ID mapping -----------------------------------
    index_path = os.path.join(VECTOR_STORE_DIR, 'nq-original.faiss')
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index: {index.ntotal:,} vectors")

    doc_ids_path = os.path.join(VECTOR_STORE_DIR, 'nq-original-doc-ids.pkl')
    with open(doc_ids_path, 'rb') as f:
        doc_ids: list[str] = pickle.load(f)
    print(f"Loaded {len(doc_ids):,} doc IDs")

    # --- Batch retrieval -----------------------------------------------------
    # One matrix × one search call: FAISS scores each row independently; this
    # is equivalent to calling retrieve(..., query_id=...) per query but
    # avoids Python/FAISS round-trip overhead and matches our design goals
    # (see module docstring).
    vecs: list[np.ndarray] = []
    valid_indices: list[int] = []
    for i, query in enumerate(queries):
        query_id = query['query_id']
        if query_id in query_embeddings:
            vecs.append(query_embeddings[query_id])
            valid_indices.append(i)

    q_matrix = np.array(vecs, dtype=np.float32)
    # Inner-product index was built on L2-normalized doc vectors; queries must
    # be normalized the same way — mirrors VectorStore.retrieve pre-search.
    faiss.normalize_L2(q_matrix)

    print(f"Batch-searching {len(valid_indices):,} queries for top-{TOP_K}...")
    t0 = time.time()
    _scores, faiss_indices = index.search(q_matrix, TOP_K)
    print(f"  Search completed in {time.time() - t0:.1f}s")

    # --- Filter: gold doc in top-K -------------------------------------------
    filtered: list[dict] = []
    for search_idx, q_idx in enumerate(valid_indices):
        query = queries[q_idx]
        gold_doc_ids = set(query.get('gold_doc_ids', []))
        if not gold_doc_ids:
            continue

        # Map FAISS indices to doc IDs for this query's top-K results
        retrieved_doc_ids = [doc_ids[fi] for fi in faiss_indices[search_idx] if fi >= 0]
        if gold_doc_ids.intersection(retrieved_doc_ids):
            filtered.append(query)

    # --- Write output --------------------------------------------------------
    with open(OUTPUT_PATH, 'w') as f:
        for query in filtered:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')

    elapsed = time.time() - t_start
    pct = len(filtered) / len(queries) * 100
    print(f"\nFiltered: {len(filtered):,} / {len(queries):,} queries ({pct:.1f}%)")
    print(f"Output:   {OUTPUT_PATH}")
    print(f"Time:     {elapsed:.1f}s")


if __name__ == '__main__':
    main()
