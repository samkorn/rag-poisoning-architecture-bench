"""Build 3 FAISS indexes from pre-computed Contriever embeddings.

The indexes correspond to the three corpus variants used by the
bench: `original`, `naive_poisoned`, and `corruptrag_ak_poisoned`.

Prerequisites:
    Pre-computed Contriever embedding pickles in
    `src/data/vector-store/`:

      * `nq-original-documents-embeddings.pkl`
      * `nq-naive-poisoned-documents-embeddings.pkl`
      * `nq-corruptrag-ak-poisoned-documents-embeddings.pkl`

    Built by `src/embeddings/embed_datasets.py`.

Usage:
    python src/embeddings/build_vector_indexes.py

Output:
    Six files in `src/data/vector-store/`:

      * `nq-original.faiss` + `nq-original-doc-ids.pkl`
      * `nq-naive-poisoned.faiss` + `nq-naive-poisoned-doc-ids.pkl`
      * `nq-corruptrag-ak-poisoned.faiss` +
        `nq-corruptrag-ak-poisoned-doc-ids.pkl`

Notes:
    Memory strategy — the ~8GB original embeddings dict is loaded
    once, stacked and normalized into a matrix, then the dict is
    freed. Poisoned indexes reuse the original matrix and `vstack`
    the small poisoned vectors on top, avoiding a second full load.
"""

import os
import time
import pickle

import faiss
import numpy as np


VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector-store')
EMBEDDING_DIM = 768

EMBEDDINGS_PATHS = {
    'original': os.path.join(VECTOR_STORE_DIR, 'nq-original-documents-embeddings.pkl'),
    'naive_poisoned': os.path.join(VECTOR_STORE_DIR, 'nq-naive-poisoned-documents-embeddings.pkl'),
    'corruptrag_ak_poisoned': os.path.join(VECTOR_STORE_DIR, 'nq-corruptrag-ak-poisoned-documents-embeddings.pkl'),
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
    'corruptrag_ak_poisoned': {
        'index': os.path.join(VECTOR_STORE_DIR, 'nq-corruptrag-ak-poisoned.faiss'),
        'doc_ids': os.path.join(VECTOR_STORE_DIR, 'nq-corruptrag-ak-poisoned-doc-ids.pkl'),
    },
}


def _load_pickle(path: str) -> dict:
    """Load a pickled `{doc_id: embedding}` dict from disk with timing logs.

    Args:
        path: Absolute path to the pickle file.

    Returns:
        The unpickled dict. Entries are emitted to stdout as a
        progress trace.
    """
    print(f"  Loading {os.path.basename(path)}...")
    t0 = time.time()
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"    Loaded {len(data):,} entries in {time.time() - t0:.1f}s")
    return data


def _normalize_embeddings_dict(embeddings_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Stack a `{doc_id: embedding}` dict into a normalized float32 matrix.

    Args:
        embeddings_dict: Map from doc ID to its raw embedding
            vector.

    Returns:
        Tuple `(matrix, doc_ids)` where `matrix[i]` is the
        L2-normalized embedding for `doc_ids[i]`. Iteration order
        matches `embeddings_dict.keys()`.
    """
    doc_ids = list(embeddings_dict.keys())
    matrix = np.stack([embeddings_dict[did] for did in doc_ids]).astype(np.float32)
    faiss.normalize_L2(matrix)
    return matrix, doc_ids


def _save_index(index: faiss.IndexFlatIP, doc_ids: list[str], corpus_type: str) -> None:
    """Write a FAISS index and its doc-id list to disk.

    Args:
        index: FAISS index already populated via `index.add`.
        doc_ids: Parallel list of doc IDs aligned with the rows of
            the matrix added to `index`. Written separately because
            FAISS doesn't store string IDs natively.
        corpus_type: Which corpus the index was built for; selects
            the destination paths from `INDEX_PATHS`.
    """
    paths = INDEX_PATHS[corpus_type]
    faiss.write_index(index, paths['index'])
    with open(paths['doc_ids'], 'wb') as f:
        pickle.dump(doc_ids, f)
    print(f"  Saved to {paths['index']}")


def build_all_indexes() -> None:
    """Build all 3 FAISS indexes, loading the 8GB original embeddings once.

    Loads the original embedding dict, stacks and normalizes it
    into a matrix, drops the dict, then builds the original index
    and the two poisoned indexes by `vstack`-ing each set of
    poisoned vectors onto the normalized original matrix.
    """
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

    # 3. CorruptRAG-AK-poisoned index (original + corruptrag-ak poisoned docs)
    print("\n=== Building CORRUPTRAG-AK-POISONED index ===")
    crak_dict = _load_pickle(EMBEDDINGS_PATHS['corruptrag_ak_poisoned'])
    crak_matrix, crak_doc_ids = _normalize_embeddings_dict(crak_dict)
    del crak_dict
    combined_matrix = np.vstack([orig_matrix, crak_matrix])
    combined_doc_ids = orig_doc_ids + crak_doc_ids
    print(f"  Combined: {orig_matrix.shape[0]:,} original + {crak_matrix.shape[0]:,} poisoned = {combined_matrix.shape[0]:,} total")
    del crak_matrix, crak_doc_ids
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(combined_matrix)
    print(f"  Index size: {index.ntotal:,} vectors")
    _save_index(index, combined_doc_ids, 'corruptrag_ak_poisoned')
    del index, combined_matrix, combined_doc_ids

    del orig_matrix, orig_doc_ids
    print(f"\nAll indexes built in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    build_all_indexes()
