"""Retrieves top-K documents from pre-built FAISS indexes.

Supports 3 corpus types: `original`, `naive_poisoned`,
`corruptrag_ak_poisoned`. Uses pre-computed query embeddings when a
`query_id` is provided (fast path), or falls back to live Contriever
embedding (slow path) for ad-hoc questions.

Prerequisites:
    FAISS index, doc-ID pickle, and corpus jsonl for each corpus
    type in `src/data/vector-store/` and `src/data/`. Built by
    `src/embeddings/build_vector_indexes.py` and the data scripts.

Notes:
    `VectorStore` instances are cached by `corpus_type` — repeated
    `VectorStore('original')` calls return the same in-memory
    instance (corpus + index loaded once per process). On macOS,
    `KMP_DUPLICATE_LIB_OK` and a one-thread Torch setting suppress
    an OpenMP conflict between `faiss-cpu` and PyTorch.

    Example:
        vs = VectorStore('original')
        results = vs.retrieve("some question", top_k=5)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # faiss-cpu and PyTorch both ship OpenMP; suppress conflict
import json
import time
import pickle

import torch  # must be imported before faiss to avoid segfault on Apple Silicon
torch.set_num_threads(1)  # avoid OpenMP deadlock against faiss's libomp on macOS
import faiss
import numpy as np

from src.data.utils import get_question_from_query_id
from src.embeddings import Embedder


# Global path variables
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector-store')
EMBEDDINGS_PATHS = {
    'queries': os.path.join(VECTOR_STORE_DIR, 'nq-queries-embeddings.pkl'),
}
_DATA_BASE = os.path.join(os.path.dirname(__file__), '..', 'data')
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
CORPUS_PATHS = {
    'original': os.path.join(_DATA_BASE, 'original-datasets', 'nq', 'corpus.jsonl'),
    'naive_poisoned': os.path.join(_DATA_BASE, 'experiment-datasets', 'nq-naive-poisoning', 'corpus.jsonl'),
    'corruptrag_ak_poisoned': os.path.join(_DATA_BASE, 'experiment-datasets', 'nq-corruptrag-ak-poisoning', 'corpus.jsonl'),
}



class VectorStore:
    """Load and cache the corpus + FAISS index for a single corpus type.

    Per-`corpus_type` singleton — calling `VectorStore('original')`
    twice returns the same instance without reloading. Pre-computed
    query embeddings are class-level state, so they're loaded once
    and shared across every instance.

    Attributes:
        _instances: Class-level cache mapping `corpus_type` to its
            constructed `VectorStore`. Drives the singleton
            behavior in `__new__`.
        _query_embeddings: Class-level dict of pre-computed query
            embeddings, lazily loaded by the first instance and
            then reused by every subsequent instance.
        corpus_type: Which corpus this instance was constructed
            for (`original`, `naive_poisoned`, `corruptrag_ak_poisoned`).
        _initialized: Sentinel flag set on first `__init__` so the
            second `VectorStore('original')` call short-circuits
            without redoing the loads.
        _corpus: Map from doc ID to `{'title', 'text'}` for this
            corpus.
        _index: FAISS index for this corpus.
        _doc_ids: List of doc IDs aligned with the rows of `_index`
            (FAISS works on integer offsets; this list maps them
            back to string IDs).
        _embedder: `Embedder` used by the live-embedding fallback
            path when `retrieve` is called without a `query_id`.
    """

    _instances: dict[str, 'VectorStore'] = {}

    # Shared across all instances (loaded once)
    _query_embeddings: dict[str, np.ndarray] | None = None

    def __new__(cls, corpus_type: str) -> 'VectorStore':
        """Return the cached instance for `corpus_type` or create a fresh one.

        The actual data loading happens in `__init__`. This method
        is the gate that makes `VectorStore` a per-`corpus_type`
        singleton.

        Args:
            corpus_type: Which corpus to retrieve from.

        Returns:
            Either the previously constructed instance for this
            `corpus_type` or a freshly allocated one.
        """
        if corpus_type in cls._instances:
            return cls._instances[corpus_type]
        else:
            instance = super().__new__(cls)
            cls._instances[corpus_type] = instance
            return instance

    def __init__(self, corpus_type: str):
        """Load the corpus, FAISS index, and embedder for `corpus_type`.

        Short-circuits when `_initialized` is already set so the
        singleton's second-and-later constructions are no-ops.

        Args:
            corpus_type: Which corpus to retrieve from
                (`original`, `naive_poisoned`, `corruptrag_ak_poisoned`).

        Raises:
            ValueError: If `corpus_type` isn't one of the three
                supported values.
        """
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        if corpus_type not in INDEX_PATHS:
            corpus_err_msg = (
                f"corpus_type must be one of {list(INDEX_PATHS.keys())}, "
                f"got '{corpus_type}'"
            )
            raise ValueError(corpus_err_msg)
        self.corpus_type = corpus_type

        t0 = time.time()
        print(f"Loading VectorStore('{corpus_type}')...")

        if VectorStore._query_embeddings is None:
            self._load_query_embeddings()
        else:
            print("  Query embeddings already loaded.")

        print(f"  Loading corpus ({corpus_type}) for VectorStore...")
        self._corpus = self._load_corpus()

        print(f"  Loading index ({corpus_type}) for VectorStore...")
        self._index, self._doc_ids = self._load_index()
    
        print("  Loading vector embedding model...")
        self._embedder = Embedder(gpu=False)

        print(f"VectorStore('{corpus_type}') ready in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # VectorStore Helper Functions
    # ------------------------------------------------------------------

    @classmethod
    def _load_query_embeddings(cls):
        """Load pre-computed query embeddings into the class-level cache.

        Called on first `VectorStore` construction; subsequent
        instances see the populated cache and skip the load.
        """
        print("  Loading query embeddings...")
        t0 = time.time()
        with open(EMBEDDINGS_PATHS['queries'], 'rb') as f:
            cls._query_embeddings = pickle.load(f)
        print(f"    Query embeddings: {len(cls._query_embeddings):,} ({time.time() - t0:.1f}s)")

    def _load_corpus(self) -> dict[str, dict]:
        """Load the corpus jsonl into a `{doc_id: {'title', 'text'}}` map.

        Returns:
            Map from doc ID to its title and text. Used by
            `retrieve` to attach passage content to FAISS hits and
            by `get_document_from_doc_id` for direct lookup.
        """
        corpus_path = CORPUS_PATHS[self.corpus_type]
        t0 = time.time()
        docs: dict[str, dict] = {}
        with open(corpus_path, 'r') as f:
            for line in f.readlines():
                line_dict = json.loads(line)
                doc_id, title, text = line_dict['_id'], line_dict['title'], line_dict['text']
                docs[doc_id] = {'title': title, 'text': text}
        print(f"    Corpus ({self.corpus_type}): {len(docs):,} docs ({time.time() - t0:.1f}s)")
        return docs

    def _load_index(self) -> tuple[faiss.IndexFlatIP, list[str]]:
        """Load the FAISS index and its parallel doc-id list from disk.

        Returns:
            Tuple `(index, doc_ids)`. `index` is the loaded
            `IndexFlatIP`; `doc_ids[i]` is the string identifier
            for vector row `i` (FAISS returns integer offsets, so
            this list is what maps them back to corpus IDs).

        Raises:
            FileNotFoundError: If the index hasn't been built yet
                — points the caller at `build_vector_indexes.py`.
        """
        index_paths = INDEX_PATHS[self.corpus_type]
        if not os.path.exists(index_paths['index']):
            index_err_msg = (
                f"FAISS index not found at {index_paths['index']}. "
                "Run build_vector_indexes.py first."
            )
            raise FileNotFoundError(index_err_msg)
        t0 = time.time()
        index = faiss.read_index(index_paths['index'])
        with open(index_paths['doc_ids'], 'rb') as f:
            doc_ids = pickle.load(f)
        print(f"    FAISS index ({self.corpus_type}) loaded: {index.ntotal:,} vectors ({time.time() - t0:.1f}s)")
        return index, doc_ids
    
    def _embed_query_live(self, query: str) -> np.ndarray:
        """Embed a query on the fly when no `query_id` is provided.

        Mirrors the offline pipeline (float32 + L2 normalize) so the
        live-embedded vector lives in the same space as the
        pre-computed ones already stored in `_query_embeddings`.

        Args:
            query: Raw question text.

        Returns:
            Contiguous float32 array of shape `(1, D)`, L2-normalized
            in place.
        """
        query_vec = self._embedder.embed_single(query)
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vec) # in place normalization
        return query_vec

    # ------------------------------------------------------------------
    # VectorStore Public API
    # ------------------------------------------------------------------
    
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        query_id: str | None = None,
    ) -> list[dict]:
        """Retrieve the top-K passages for a question.

        Args:
            question: The query text. Used by the live-embedding
                fallback path when `query_id` isn't provided.
            top_k: Number of passages to return.
            query_id: When provided (e.g. `test0`), uses the
                pre-computed query embedding for speed. Otherwise
                falls back to live Contriever embedding via
                `_embed_query_live`.

        Returns:
            List of result dicts with keys `doc_id`, `title`,
            `text`, and `score` (cosine similarity, since the
            index is L2-normalized inner-product). Sorted by score
            descending.

        Raises:
            KeyError: If `query_id` is provided but isn't in the
                pre-computed query-embeddings cache.
        """
        if query_id is not None:
            # if query_id is provided, use the pre-computed query embedding
            if query_id in self._query_embeddings:
                q_emb = np.ascontiguousarray(
                    self._query_embeddings[query_id], dtype=np.float32
                ).reshape(1, -1)
                faiss.normalize_L2(q_emb)
            else:
                raise KeyError(f"query_id '{query_id}' not found in pre-computed query embeddings")
        else:
            # if query_id is not provided, embed the question live
            q_emb = self._embed_query_live(question)

        # search the index for the top-K documents
        scores, indices = self._index.search(q_emb, top_k)
        
        # format the results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            did = self._doc_ids[idx]
            doc = self._corpus.get(did, {'title': '', 'text': ''})
            results.append({
                'doc_id': did,
                'title': doc['title'],
                'text': doc['text'],
                'score': float(score),
            })
        return results
    

    def get_document_from_doc_id(self, doc_id: str) -> dict:
        """Look up a single document by ID, bypassing retrieval.

        Used by `AgenticRAG.get_document_by_id` and by `RLM` when
        expanding retrieved articles into their full passage list.

        Args:
            doc_id: Corpus document identifier.

        Returns:
            Dict with `title` and `text`.

        Raises:
            ValueError: If `doc_id` isn't in the loaded corpus.
        """
        if doc_id not in self._corpus:
            raise ValueError(f"doc_id '{doc_id}' not found in corpus '{self.corpus_type}'")
        return self._corpus[doc_id]


if __name__ == '__main__':

    print("=== Sanity check: retrieving for test0 across all 3 indexes ===\n")
    query_id = 'test0'
    for corpus_type in ['original', 'naive_poisoned', 'corruptrag_ak_poisoned']:
        print(f"--- {corpus_type} (top 5) ---")
        vs = VectorStore(corpus_type)
        print(f"Retrieving for query {query_id} in {corpus_type}...")
        t0 = time.time()
        question = get_question_from_query_id(query_id)
        results = vs.retrieve(question, top_k=5, query_id=query_id)
        print(f"  Retrieved in {time.time() - t0:.2f}s\n\nRESULTS:")
        for i, r in enumerate(results):
            is_poisoned = r['doc_id'].startswith('poisoned')
            marker = ' [POISONED]' if is_poisoned else ''
            print(f"  {i+1}. [{r['doc_id']}] (score={r['score']:.4f}){marker}")
            print(f"     {r['text'][:120]}...")
        print()

    print("=== Sanity check: retrieving for test0-4 for corruptrag_ak_poisoned index ===\n")
    vs = VectorStore('corruptrag_ak_poisoned')
    query_ids = ['test0', 'test1', 'test2', 'test3', 'test4']
    for query_id in query_ids:
        t0 = time.time()
        question = get_question_from_query_id(query_id)
        print(f"--- Query [{query_id}]: {question} (top 5) ---")
        results = vs.retrieve(question, top_k=5, query_id=query_id)
        print(f"  Retrieved in {time.time() - t0:.2f}s")
        for i, r in enumerate(results):
            is_poisoned = r['doc_id'].startswith('poisoned')
            marker = ' [POISONED]' if is_poisoned else ''
            print(f"  {i+1}. [{r['doc_id']}] (score={r['score']:.4f}){marker}")
            print(f"     {r['text'][:120]}...")
        print()
    
    # exit without waiting for large variables to be garbage collected
    os._exit(0)
