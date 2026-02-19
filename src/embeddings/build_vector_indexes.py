"""
Build 3 FAISS indexes (original, naive-poisoned, adversarial-poisoned) from
pre-computed Contriever embeddings.

Indexes are saved to workspace/data/vector-store/ as:
  - nq-original.faiss            + nq-original-doc-ids.pkl
  - nq-naive-poisoned.faiss      + nq-naive-poisoned-doc-ids.pkl
  - nq-adversarial-poisoned.faiss + nq-adversarial-poisoned-doc-ids.pkl

Memory strategy: the ~8GB original embeddings dict is loaded once, stacked +
normalized into a matrix, then the dict is freed. Poisoned indexes reuse the
original matrix and vstack the small poisoned vectors on top.
"""

import os
import time
import pickle
import numpy as np
import faiss


VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector-store')
EMBEDDING_DIM = 768

EMBEDDINGS_PATHS = {
    'original': os.path.join(VECTOR_STORE_DIR, 'nq-original-documents-embeddings.pkl'),
    'naive_poisoned': os.path.join(VECTOR_STORE_DIR, 'nq-naive-poisoned-documents-embeddings.pkl'),
    'adversarial_poisoned': os.path.join(VECTOR_STORE_DIR, 'nq-adversarial-poisoned-documents-embeddings.pkl'),
}

INDEX_PATHS = {
    'original': {
        'index': os.path.join(VECTOR_STORE_DIR, 'nq-original.faiss'),
        'doc_ids': os.path.join(VECTOR_STORE_DIR, 'nq-original-doc-ids.pkl'),
    },
    'naive_poisoned': {
        'index': os.path.join(VECTOR_STORE_DIR, 'nq-naive-poisoned.faiss'),
        'doc_ids': os.path.join(VECTOR_STORE_DIR, 'nq-naive-poisoned-doc-ids.pkl'),
    },
    'adversarial_poisoned': {
        'index': os.path.join(VECTOR_STORE_DIR, 'nq-adversarial-poisoned.faiss'),
        'doc_ids': os.path.join(VECTOR_STORE_DIR, 'nq-adversarial-poisoned-doc-ids.pkl'),
    },
}


def _load_pickle(path: str) -> dict:
    print(f"  Loading {os.path.basename(path)}...")
    t0 = time.time()
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"    Loaded {len(data):,} entries in {time.time() - t0:.1f}s")
    return data


def _normalize_embeddings_dict(embeddings_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Stack a {doc_id: embedding} dict into a normalized float32 matrix.

    Returns (matrix, doc_ids) where matrix[i] is the L2-normalized embedding
    for doc_ids[i].
    """
    doc_ids = list(embeddings_dict.keys())
    matrix = np.stack([embeddings_dict[did] for did in doc_ids]).astype(np.float32)
    faiss.normalize_L2(matrix)
    return matrix, doc_ids


def _save_index(index: faiss.IndexFlatIP, doc_ids: list[str], corpus_type: str) -> None:
    """Write a FAISS index + doc-id list to disk."""
    paths = INDEX_PATHS[corpus_type]
    faiss.write_index(index, paths['index'])
    with open(paths['doc_ids'], 'wb') as f:
        pickle.dump(doc_ids, f)
    print(f"  Saved to {paths['index']}")


if __name__ == '__main__':
    t0 = time.time()

    # Load original embeddings and convert to normalized matrix once
    print("\n=== Loading original embeddings ===")
    original_dict = _load_pickle(EMBEDDINGS_PATHS['original'])
    print("  Stacking + normalizing original matrix...")
    orig_matrix, orig_doc_ids = _normalize_embeddings_dict(original_dict)
    del original_dict  # free ~8 GB dict, keep ~8 GB matrix
    print(f"  Original matrix: {orig_matrix.shape}")

    # 1. Original index
    print("\n=== Building ORIGINAL index ===")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(orig_matrix)
    print(f"  Index size: {index.ntotal:,} vectors")
    _save_index(index, orig_doc_ids, 'original')
    del index

    # 2. Naive-poisoned index (original + naive poisoned docs)
    print("\n=== Building NAIVE-POISONED index ===")
    naive_dict = _load_pickle(EMBEDDINGS_PATHS['naive_poisoned'])
    naive_matrix, naive_doc_ids = _normalize_embeddings_dict(naive_dict)
    del naive_dict
    combined_matrix = np.vstack([orig_matrix, naive_matrix])
    combined_doc_ids = orig_doc_ids + naive_doc_ids
    print(f"  Combined: {orig_matrix.shape[0]:,} original + {naive_matrix.shape[0]:,} poisoned = {combined_matrix.shape[0]:,} total")
    del naive_matrix, naive_doc_ids
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(combined_matrix)
    print(f"  Index size: {index.ntotal:,} vectors")
    _save_index(index, combined_doc_ids, 'naive_poisoned')
    del index, combined_matrix, combined_doc_ids

    # 3. Adversarial-poisoned index (original + adversarial poisoned docs)
    print("\n=== Building ADVERSARIAL-POISONED index ===")
    adv_dict = _load_pickle(EMBEDDINGS_PATHS['adversarial_poisoned'])
    adv_matrix, adv_doc_ids = _normalize_embeddings_dict(adv_dict)
    del adv_dict
    combined_matrix = np.vstack([orig_matrix, adv_matrix])
    combined_doc_ids = orig_doc_ids + adv_doc_ids
    print(f"  Combined: {orig_matrix.shape[0]:,} original + {adv_matrix.shape[0]:,} poisoned = {combined_matrix.shape[0]:,} total")
    del adv_matrix, adv_doc_ids
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(combined_matrix)
    print(f"  Index size: {index.ntotal:,} vectors")
    _save_index(index, combined_doc_ids, 'adversarial_poisoned')
    del index, combined_matrix, combined_doc_ids

    del orig_matrix, orig_doc_ids
    print(f"\nAll indexes built in {time.time() - t0:.1f}s")
