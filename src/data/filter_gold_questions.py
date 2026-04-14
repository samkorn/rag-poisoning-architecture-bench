"""
Filter nq-questions.jsonl to questions where at least one gold-standard
document appears in top-10 clean retrieval.

Uses pre-computed query embeddings + the original FAISS index to perform
a single batch search, then checks each query's top-10 results against
its gold_doc_ids from qrels.

Output:
    experiment-datasets/nq-questions-gold-filtered.jsonl

Usage:
    cd workspace/data
    python filter_gold_questions.py
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
QUESTIONS_PATH = os.path.join(_DATA_DIR, 'experiment-datasets', 'nq-questions.jsonl')
OUTPUT_PATH = os.path.join(_DATA_DIR, 'experiment-datasets', 'nq-questions-gold-filtered.jsonl')

TOP_K = 10


def main():
    t_start = time.time()

    # --- Load questions ------------------------------------------------------
    questions: list[dict] = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"Loaded {len(questions):,} questions")

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
    # Build query matrix in question order, tracking which questions have
    # embeddings available.
    vecs: list[np.ndarray] = []
    valid_indices: list[int] = []
    for i, q in enumerate(questions):
        qid = q['query_id']
        if qid in query_embeddings:
            vecs.append(query_embeddings[qid])
            valid_indices.append(i)

    q_matrix = np.array(vecs, dtype=np.float32)
    faiss.normalize_L2(q_matrix)  # match VectorStore.retrieve behavior

    print(f"Batch-searching {len(valid_indices):,} queries for top-{TOP_K}...")
    t0 = time.time()
    _scores, faiss_indices = index.search(q_matrix, TOP_K)
    print(f"  Search completed in {time.time() - t0:.1f}s")

    # --- Filter: gold doc in top-K -------------------------------------------
    filtered: list[dict] = []
    for search_idx, q_idx in enumerate(valid_indices):
        q = questions[q_idx]
        gold_doc_ids = set(q.get('gold_doc_ids', []))
        if not gold_doc_ids:
            continue

        # Map FAISS indices to doc IDs for this query's top-K results
        retrieved_doc_ids = [doc_ids[fi] for fi in faiss_indices[search_idx] if fi >= 0]
        if gold_doc_ids.intersection(retrieved_doc_ids):
            filtered.append(q)

    # --- Write output --------------------------------------------------------
    with open(OUTPUT_PATH, 'w') as f:
        for q in filtered:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    elapsed = time.time() - t_start
    pct = len(filtered) / len(questions) * 100
    print(f"\nFiltered: {len(filtered):,} / {len(questions):,} questions ({pct:.1f}%)")
    print(f"Output:   {OUTPUT_PATH}")
    print(f"Time:     {elapsed:.1f}s")


if __name__ == '__main__':
    main()
