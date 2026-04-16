"""
experiments/experiment.py

Core experiment execution logic for RAG Poisoning Architecture Bench.

Defines:
  - ExperimentConfig: immutable specification for one experiment
  - QuestionResult: result for a single question within an experiment
  - run_single_question(): atomic unit of work (one question, one architecture)
  - run_question_batch(): batch processor with per-question checkpointing
  - split_questions(): utility for dividing work across workers

This module is pure Python (no Modal dependency). It is imported by the
orchestrator's Modal worker function.

Imports use package-qualified paths (``from src.architectures.X import Y``),
resolved via the editable install (``pip install -e .``).
"""

import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal

import threading

from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from timeout_decorator import timeout


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATTACK_TO_CORPUS: dict[str, str] = {
    'clean': 'original',
    'naive': 'naive_poisoned',
    'corruptrag_ak': 'corruptrag_ak_poisoned',
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Immutable specification for one of the 12 experiments."""

    experiment_id: str  # e.g. "vanilla_clean", "rlm_corruptrag_ak"
    architecture: Literal['vanilla', 'agentic', 'madam', 'rlm']
    attack_type: Literal['clean', 'naive', 'corruptrag_ak']
    k: Optional[int]  # None for RLM (uses topic-scoped context)
    n_poisoned: int = 1  # Fixed at 1 for primary experiment
    defensive: bool = False  # Deferred to Phase 2
    backbone_model: str = 'gpt-5-mini'
    # rlm_root_model: str = 'gpt-5.2'  # Only used when architecture="rlm" root agent
    reasoning_effort: Optional[str] = None  # e.g. "low", "medium"

    @property
    def corpus_type(self) -> str:
        """Map attack_type to the VectorStore corpus identifier."""
        return ATTACK_TO_CORPUS[self.attack_type]

    def to_dict(self) -> dict:
        d = asdict(self)
        d['corpus_type'] = self.corpus_type
        return d


@dataclass
class QuestionResult:
    """Result for a single question within an experiment."""

    experiment_id: str
    question_id: str  # e.g. "test0", "test3451"
    question_text: str
    correct_answer: str
    target_answer: Optional[str]  # Attacker's desired wrong answer (None for clean)

    # Core output
    system_answer: str  # The answer the architecture produced

    # Retrieval info
    retrieved_doc_ids: list[str] = field(default_factory=list)
    poison_retrieved: Optional[bool] = None  # Was poison in top-K?
    poison_rank: Optional[int] = None  # 1-indexed rank of poison if retrieved
    gold_doc_ranks: list[int] = field(default_factory=list)  # 1-indexed ranks of gold docs in retrieved results

    # Architecture-specific metadata
    metadata: dict = field(default_factory=dict)

    # Timing
    latency_seconds: float = 0.0

    # Error handling
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_log_tag(
    config: 'ExperimentConfig',
    query_id: str,
    question_num: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> str:
    """Build a bracketed tag for log lines.

    Examples::

        [madam k=3 clean test0]
        [madam k=3 clean test0 q=3/12]
    """
    parts = [config.architecture]
    if config.k is not None:
        parts.append(f'k={config.k}')
    parts.append(config.attack_type)
    parts.append(query_id)
    if question_num is not None and batch_size is not None:
        parts.append(f'q={question_num}/{batch_size}')
    return f"[{' '.join(parts)}]"


def is_poison_doc_id(doc_id: str) -> bool:
    """Return True if *doc_id* belongs to an injected poisoned document.

    Original NQ passages have IDs like "doc0", "doc12345".
    Poisoned documents start with "poisoned" (e.g. "poisoned-naive-q:test3").
    """
    return doc_id.startswith('poisoned')


def detect_poison_in_results(
    retrieved_docs: list[dict],
) -> tuple[bool, Optional[int]]:
    """Scan retrieval results for a poisoned document.

    Returns (poison_found, 1-indexed rank or None).
    """
    for i, doc in enumerate(retrieved_docs):
        if is_poison_doc_id(doc['doc_id']):
            return True, i + 1
    return False, None


def detect_gold_in_results(
    retrieved_docs: list[dict],
    gold_doc_ids: list[str],
) -> list[int]:
    """Find 1-indexed ranks of gold-standard documents in retrieval results.

    Returns sorted list of ranks (empty if no gold docs retrieved).
    """
    gold_set = set(gold_doc_ids)
    return [i + 1 for i, doc in enumerate(retrieved_docs)
            if doc['doc_id'] in gold_set]


def split_questions(question_ids: list[str], n_workers: int = 99) -> list[list[str]]:
    """Split question IDs into *n_workers* roughly-equal batches (round-robin).

    ~1150 / 99 ~ 11-12 questions per worker.
    """
    batches: list[list[str]] = [[] for _ in range(n_workers)]
    for i, qid in enumerate(question_ids):
        batches[i % n_workers].append(qid)
    return [b for b in batches if b]


# ---------------------------------------------------------------------------
# Retrieval capture
# ---------------------------------------------------------------------------

class RetrievalCapture:
    """Capture all VectorStore retrieval activity during a _run() call.

    Monkeypatches retrieve() and get_document_from_doc_id() on the VectorStore
    instance for the duration of the context manager.  Since VectorStore is a
    singleton, the architecture's own self.vector_store IS the same object, so
    the capture sees everything the architecture does — including Agentic RAG's
    tool-driven retrieval and RLM's title-group expansion.

    Usage::

        with RetrievalCapture(vector_store) as capture:
            answer = qa_system._run(question_text, query_id)
        # capture.retrieve_calls  — list of {kwargs, results} per retrieve() call
        # capture.doc_fetches     — list of doc_ids fetched via get_document_from_doc_id()
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.retrieve_calls: list[dict] = []
        self.doc_fetches: list[str] = []
        self._orig_retrieve = vector_store.retrieve
        self._orig_get_doc = vector_store.get_document_from_doc_id

    def __enter__(self):
        def capturing_retrieve(*args, **kwargs):
            result = self._orig_retrieve(*args, **kwargs)
            self.retrieve_calls.append({'kwargs': kwargs, 'results': result})
            return result

        def capturing_get_doc(doc_id):
            result = self._orig_get_doc(doc_id)
            self.doc_fetches.append(doc_id)
            return result

        self.vector_store.retrieve = capturing_retrieve
        self.vector_store.get_document_from_doc_id = capturing_get_doc
        return self

    def __exit__(self, *exc):
        self.vector_store.retrieve = self._orig_retrieve
        self.vector_store.get_document_from_doc_id = self._orig_get_doc
        return False


# ---------------------------------------------------------------------------
# QA system factory
# ---------------------------------------------------------------------------

def create_qa_system(config: ExperimentConfig):
    """Instantiate the QASystem subclass for *config*.

    Each architecture's constructor internally creates a VectorStore singleton
    for the appropriate corpus_type, so clean/poisoned index selection is
    handled automatically.
    """
    common_kwargs: dict = {'model_id': config.backbone_model}
    if config.reasoning_effort:
        common_kwargs['reasoning_effort'] = config.reasoning_effort

    if config.architecture == 'vanilla':
        from src.architectures.vanilla_rag import VanillaRAG
        return VanillaRAG(
            corpus_type=config.corpus_type,
            top_k=config.k,
            **common_kwargs,
        )

    elif config.architecture == 'agentic':
        from src.architectures.agentic_rag import AgenticRAG
        return AgenticRAG(
            corpus_type=config.corpus_type,
            top_k=config.k,
            **common_kwargs,
        )

    elif config.architecture == 'madam':
        from src.architectures.madam_rag import MadamRAG
        return MadamRAG(
            corpus_type=config.corpus_type,
            top_k=config.k,
            **common_kwargs,
        )

    elif config.architecture == 'rlm':
        # # RLM root agent uses gpt-5.2 for paper fidelity.
        # # top_k is forbidden — RLM uses full topic-scoped context internally.
        # rlm_kwargs: dict = {'model_id': config.rlm_root_model}
        # if config.reasoning_effort:
        #     rlm_kwargs['reasoning_effort'] = config.reasoning_effort
        from src.architectures.recursive_lm import RLM
        return RLM(corpus_type=config.corpus_type, **common_kwargs)

    else:
        raise ValueError(f"Unknown architecture: {config.architecture!r}")


# ---------------------------------------------------------------------------
# Core execution: single question
# ---------------------------------------------------------------------------

def _run_single_question(
    config: ExperimentConfig,
    question: dict,
    qa_system,
    log_tag: str,
) -> QuestionResult:
    """Inner execution: run one question with retry + timeout.

    Raises on failure — caller (run_single_question) catches and records errors.
    """
    query_id: str = question['query_id']
    question_text: str = question['question']
    correct_answer: str = question.get('correct_answer', '')
    target_answer: Optional[str] = (
        question.get('target_answer') if config.attack_type != 'clean' else None
    )

    # --- Run architecture with retrieval capture ------------------------
    print(f"{log_tag} Starting question: {question_text[:80]!r}")
    qa_system._log_tag = log_tag
    with RetrievalCapture(qa_system.vector_store) as capture:
        start = time.time()
        system_answer: str = qa_system._run(question_text, query_id)
        latency = time.time() - start
    print(f"{log_tag} Finished question ({latency:.1f}s)\n")

    # --- Extract retrieval metadata from capture ------------------------
    primary_results: list[dict] = []
    if capture.retrieve_calls:
        primary_results = capture.retrieve_calls[0]['results']
    retrieved_doc_ids = [d['doc_id'] for d in primary_results]

    # Poison detection in initial retrieval.
    poison_retrieved: Optional[bool] = None
    poison_rank: Optional[int] = None
    if config.attack_type != 'clean' and primary_results:
        poison_retrieved, poison_rank = detect_poison_in_results(
            primary_results
        )

    # Gold-standard document detection in initial retrieval.
    gold_doc_ids: list[str] = question.get('gold_doc_ids', [])
    gold_doc_ranks: list[int] = []
    if gold_doc_ids and primary_results:
        gold_doc_ranks = detect_gold_in_results(
            primary_results, gold_doc_ids
        )

    # --- Architecture-specific metadata from capture --------------------
    metadata: dict = {
        'architecture': config.architecture,
        'corpus_type': config.corpus_type,
        'k': config.k,
    }

    if config.architecture == 'vanilla':
        metadata['passages_text_length'] = sum(
            len(d['text']) for d in primary_results
        )

    elif config.architecture == 'agentic':
        metadata['n_retrieve_calls'] = len(capture.retrieve_calls)
        if len(capture.retrieve_calls) > 1:
            seen: set[str] = set()
            all_doc_ids: list[str] = []
            for call in capture.retrieve_calls:
                for d in call['results']:
                    if d['doc_id'] not in seen:
                        all_doc_ids.append(d['doc_id'])
                        seen.add(d['doc_id'])
            metadata['all_retrieved_doc_ids'] = all_doc_ids
        if capture.doc_fetches:
            metadata['doc_fetches'] = capture.doc_fetches

    elif config.architecture == 'madam':
        metadata['passages_text_length'] = sum(
            len(d['text']) for d in primary_results
        )

    elif config.architecture == 'rlm':
        metadata['context_doc_ids'] = capture.doc_fetches
        metadata['context_n_docs'] = len(capture.doc_fetches)
        if config.attack_type != 'clean':
            metadata['poison_in_context'] = any(
                is_poison_doc_id(did) for did in capture.doc_fetches
            )
        if gold_doc_ids:
            gold_set = set(gold_doc_ids)
            metadata['gold_in_context'] = any(
                did in gold_set for did in capture.doc_fetches
            )

    return QuestionResult(
        experiment_id=config.experiment_id,
        question_id=query_id,
        question_text=question_text,
        correct_answer=correct_answer,
        target_answer=target_answer,
        system_answer=system_answer,
        retrieved_doc_ids=retrieved_doc_ids,
        poison_retrieved=poison_retrieved,
        poison_rank=poison_rank,
        gold_doc_ranks=gold_doc_ranks,
        metadata=metadata,
        latency_seconds=latency,
    )


# Pre-build both timeout flavors at module load so retry-wraps-timeout ordering
# is preserved and we avoid per-call wrapping overhead. Auto-dispatch at call
# time based on thread context.
#   use_signals=True:  SIGALRM-based, main-thread only. Used locally.
#   use_signals=False: multiprocessing-based, works in non-main threads (Modal
#                      workers). Note: on macOS this requires picklable args,
#                      which the qa_system isn't — so signals must be used
#                      locally (mac defaults to spawn start method).
_run_single_question_retry_signal_timed = retry(
    stop=stop_after_attempt(2),
    reraise=True,
)(timeout(60*10, use_signals=True)(_run_single_question))

_run_single_question_retry_thread_timed = retry(
    stop=stop_after_attempt(2),
    reraise=True,
)(timeout(60*10, use_signals=False)(_run_single_question))


def run_single_question(
    config: ExperimentConfig,
    question: dict,  # {_id, text, answer, target_answer?, ...}
    qa_system,  # Pre-instantiated QASystem subclass
    question_num: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> QuestionResult:
    """Execute one question through the configured architecture.

    Delegates to _run_single_question (which has retry + timeout).
    **Never raises** — all errors are captured in QuestionResult.error so
    that one bad question cannot crash the worker or lose batch progress.
    """
    query_id: str = question['query_id']
    question_text: str = question['question']
    correct_answer: str = question.get('correct_answer', '')
    target_answer: Optional[str] = (
        question.get('target_answer')
        if config.attack_type != 'clean'
        else None
    )

    log_tag = make_log_tag(config, query_id, question_num, batch_size)

    try:
        _run_single_question_retry_timed = (
            _run_single_question_retry_signal_timed
            if threading.current_thread() is threading.main_thread()
            else _run_single_question_retry_thread_timed
        )
        return _run_single_question_retry_timed(config, question, qa_system, log_tag)
    except Exception as e:
        print(f"{log_tag} ERROR: {type(e).__name__}: {e}")
        return QuestionResult(
            experiment_id=config.experiment_id,
            question_id=query_id,
            question_text=question_text,
            correct_answer=correct_answer,
            target_answer=target_answer,
            system_answer='',
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


# ---------------------------------------------------------------------------
# Batch execution: one worker's workload
# ---------------------------------------------------------------------------

def run_question_batch(
    config: ExperimentConfig,
    question_ids: list[str],
    questions: dict[str, dict],  # query_id -> question dict
    results_dir: str,  # Root results directory (e.g. /vol/results)
    modal_volume=None,  # Optional Modal Volume — reload before / commit after each question
) -> dict:
    """Process a batch of questions for one experiment.

    Called by each of the 99 Modal worker containers.  Resources (VectorStore,
    QASystem) are loaded once per container, then questions are processed
    sequentially with immediate per-question JSON writes for checkpointing.

    If *modal_volume* is provided, ``volume.reload()`` is called before each
    question (to see results committed by other workers / retried attempts)
    and ``volume.commit()`` is called after each write (so partial progress
    survives preemption).

    Returns:
        {"completed": int, "skipped": int, "errors": int, "total": int}
    """
    exp_dir = os.path.join(results_dir, config.experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    # Commit so the directory survives the volume.reload() calls below —
    # otherwise reload reverts our uncommitted mkdir and the first write fails.
    if modal_volume is not None:
        modal_volume.commit()

    # create_qa_system() triggers VectorStore(corpus_type) internally,
    # which loads the FAISS index + corpus once (singleton).  Subsequent
    # VectorStore(corpus_type) calls from RetrievalCapture or anywhere
    # else return the cached instance instantly.
    qa_system = create_qa_system(config)

    completed = 0
    skipped = 0
    errors = 0
    batch_size = len(question_ids)

    for q_idx, query_id in enumerate(question_ids, 1):
        log_tag = make_log_tag(config, query_id, q_idx, batch_size)
        result_path = os.path.join(exp_dir, f'{query_id}.json')

        # Refresh volume view to see writes from other workers / retried attempts.
        if modal_volume is not None:
            modal_volume.reload()

        # --- Checkpoint: skip already-completed questions ---
        # Only skip if the result file exists AND contains a successful result.
        # Error results (from previous rate-limit failures etc.) are deleted
        # so they get retried on the next run.
        if os.path.exists(result_path):
            try:
                with open(result_path) as _f:
                    _prev = json.loads(_f.read())
                if _prev.get('error') is None:
                    skipped += 1
                    print(f"{log_tag} Skipping already completed question")
                    continue
                # Previous result was an error — delete and retry.
                print(f"{log_tag} Deleting errored result file and retrying...")
                os.remove(result_path)
            except (json.JSONDecodeError, OSError):
                # Corrupt file — delete and retry.
                print(f"{log_tag} Deleting corrupt result file and retrying...")
                os.remove(result_path)

        question = questions.get(query_id)
        if question is None:
            # Missing question — record error and continue
            print(f"{log_tag} Missing question in questions dict - recording error")
            errors += 1
            err_result = QuestionResult(
                experiment_id=config.experiment_id,
                question_id=query_id,
                question_text='',
                correct_answer='',
                target_answer=None,
                system_answer='',
                error=f"Question ID {query_id!r} not found in questions dict",
            )
            with open(result_path, 'w') as f:
                f.write(err_result.to_json())
            if modal_volume is not None:
                modal_volume.commit()
            continue

        result = run_single_question(config, question, qa_system, q_idx, batch_size)

        # Write immediately — per-question checkpoint
        with open(result_path, 'w') as f:
            f.write(result.to_json())
        if modal_volume is not None:
            modal_volume.commit()

        if result.error:
            errors += 1
        else:
            completed += 1

    return {
        'completed': completed,
        'skipped': skipped,
        'errors': errors,
        'total': len(question_ids),
    }
