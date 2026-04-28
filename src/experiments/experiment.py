"""Core experiment execution logic for the RAG Poisoning Architecture Bench.

Defines:

  * `ExperimentConfig` — immutable specification for one experiment.
  * `QuestionResult` — result for a single question within an
    experiment.
  * `run_single_question()` — atomic unit of work (one question, one
    architecture).
  * `run_question_batch()` — batch processor with per-question
    checkpointing.
  * `split_query_ids()` — utility for dividing work across workers.

This module is pure Python (no Modal dependency). It is imported by
the orchestrator's Modal worker function.

Notes:
    Imports use package-qualified paths (`from src.architectures.X
    import Y`), resolved via the editable install (`pip install -e .`).
    The function names `run_single_question` and `run_question_batch`
    are kept on the `question` side of the `query` vs `question`
    naming convention because renaming them would cascade through
    Modal log history; their parameters use `query_id` correctly
    (see CONVENTIONS.md).
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

from src.architectures.qa_system import QASystem


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
    """Specify one of the 12 experiment cells in the bench matrix.

    Attributes:
        experiment_id: Stable identifier used for the result
            directory (e.g. `vanilla_clean`, `rlm_corruptrag_ak`).
        architecture: Which RAG architecture to run.
        attack_type: Which attack condition to evaluate against.
        k: Top-K for retrieval. `None` for RLM, which uses
            topic-scoped context instead of fixed-K retrieval.
        n_poisoned: Number of injected poisoned documents per query.
            Fixed at 1 for the primary experiment.
        defensive: Reserved for Phase 2 defensive variants;
            currently unused.
        backbone_model: OpenAI model identifier shared across
            architectures.
        reasoning_effort: Optional reasoning-effort level passed to
            the architectures' OpenAI calls.
    """

    experiment_id: str  # e.g. "vanilla_clean", "rlm_corruptrag_ak"
    architecture: Literal['vanilla', 'agentic', 'madam', 'rlm']
    attack_type: Literal['clean', 'naive', 'corruptrag_ak']
    k: Optional[int]  # None for RLM (uses topic-scoped context)
    n_poisoned: int = 1  # Fixed at 1 for primary experiment
    defensive: bool = False  # Deferred to Phase 2
    backbone_model: str = 'gpt-5-mini'
    reasoning_effort: Optional[str] = None  # e.g. "low", "medium"

    @property
    def corpus_type(self) -> str:
        """Map `attack_type` to its `VectorStore` corpus identifier."""
        return ATTACK_TO_CORPUS[self.attack_type]

    def to_dict(self) -> dict:
        """Serialize to a plain dict, including the derived `corpus_type`.

        Used by the Modal worker to round-trip the config across
        process boundaries (Modal pickles plain dicts more cheaply
        than dataclass instances).

        Returns:
            Dict with every field plus a derived `corpus_type` key.
        """
        d = asdict(self)
        d['corpus_type'] = self.corpus_type
        return d


@dataclass
class QuestionResult:
    """Result for a single query within an experiment.

    The class name and the `question_id` / `question_text` field
    names are kept (rather than `QueryResult` / `query_id` /
    `question`) to preserve the on-disk JSON schema across all
    ~14k persisted result files. Per the project naming standard,
    "query" is the conceptual identifier; the JSON keys are the
    legacy schema names — see CONVENTIONS.md.

    Attributes:
        experiment_id: Identifier of the experiment cell this
            result belongs to.
        question_id: On-disk schema field; conceptually the
            `query_id`.
        question_text: Natural-language question text.
        correct_answer: Gold short-form answer.
        target_answer: Attacker's desired wrong answer; `None` for
            the clean condition.
        system_answer: The answer the architecture produced.
        retrieved_doc_ids: Doc IDs returned by the architecture's
            primary (initial) retrieval call.
        poison_retrieved: Whether a poisoned doc appeared in the
            primary retrieval. `None` for the clean condition.
        poison_rank: 1-indexed rank of the first poisoned doc in
            the primary retrieval, or `None` if none was retrieved.
        gold_doc_ranks: 1-indexed ranks of every gold doc that
            appeared in the primary retrieval (sorted ascending).
        metadata: Architecture-specific extras
            (`passages_text_length` for Vanilla/MADAM,
            `n_retrieve_calls` and `doc_fetches` for Agentic,
            `context_doc_ids` and `context_n_docs` for RLM, etc.).
        latency_seconds: Wall time of the `_run` call.
        error: `None` on success; otherwise the captured exception
            type, message, and traceback.
    """

    experiment_id: str
    question_id: str  # on-disk schema field; conceptually the query_id
    question_text: str  # the natural-language question text
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
        """Serialize to a JSON string for per-question result files.

        Uses `default=str` to gracefully stringify any unexpected
        non-JSON-native fields (e.g. exceptions captured into
        `error`).

        Returns:
            JSON string with no ASCII escaping.
        """
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

    Args:
        config: Experiment config; supplies architecture, k, and
            attack type.
        query_id: NQ test query identifier.
        question_num: 1-indexed position of this question within
            its batch. Optional.
        batch_size: Total number of questions in the batch.
            Optional. Both `question_num` and `batch_size` must be
            provided together to render the `q=i/N` suffix.

    Returns:
        Bracketed tag, e.g. `[madam k=3 clean test0]` or
        `[madam k=3 clean test0 q=3/12]`.
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
    """Return whether `doc_id` belongs to an injected poisoned document.

    Original NQ passages have IDs like `doc0`, `doc12345`. Poisoned
    documents are prefixed with `poisoned` (e.g.
    `poisoned-naive-q:test3`).

    Args:
        doc_id: Corpus document identifier.

    Returns:
        `True` if the ID is poisoned, `False` otherwise.
    """
    return doc_id.startswith('poisoned')


def detect_poison_in_results(
    retrieved_docs: list[dict],
) -> tuple[bool, Optional[int]]:
    """Scan retrieval results for the first poisoned document.

    Args:
        retrieved_docs: Result list from `VectorStore.retrieve`.

    Returns:
        Tuple `(poison_found, rank)`. `rank` is the 1-indexed
        position of the first poisoned doc, or `None` if none was
        found.
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

    Args:
        retrieved_docs: Result list from `VectorStore.retrieve`.
        gold_doc_ids: Doc IDs that count as gold for the query.

    Returns:
        Sorted list of 1-indexed ranks. Empty when no gold doc
        appeared in the retrieval.
    """
    gold_set = set(gold_doc_ids)
    return [i + 1 for i, doc in enumerate(retrieved_docs)
            if doc['doc_id'] in gold_set]


def split_query_ids(query_ids: list[str], n_workers: int = 99) -> list[list[str]]:
    """Split query IDs into roughly equal round-robin batches.

    For the bench's ~1150 questions and 99 workers, this yields
    11–12 queries per worker.

    Args:
        query_ids: All query IDs to distribute.
        n_workers: Number of worker batches to produce. Defaults
            to 99 (Modal's per-app concurrency limit).

    Returns:
        List of non-empty batches. Empty trailing batches (when
        `n_workers > len(query_ids)`) are omitted.
    """
    batches: list[list[str]] = [[] for _ in range(n_workers)]
    for i, query_id in enumerate(query_ids):
        batches[i % n_workers].append(query_id)
    return [b for b in batches if b]


# ---------------------------------------------------------------------------
# Retrieval capture
# ---------------------------------------------------------------------------

class RetrievalCapture:
    """Capture every `VectorStore` retrieval call during a `_run()` invocation.

    Monkeypatches `retrieve()` and `get_document_from_doc_id()` on
    the `VectorStore` instance for the duration of the context
    manager. Since `VectorStore` is a singleton, the architecture's
    own `self.vector_store` IS the same object — so the capture
    sees everything the architecture does, including Agentic RAG's
    tool-driven retrievals and RLM's title-group expansion.

    Example:
        with RetrievalCapture(vector_store) as capture:
            answer = qa_system._run(question_text, query_id)
        # capture.retrieve_calls  — list of {kwargs, results} per call
        # capture.doc_fetches     — list of fetched doc IDs

    Notes:
        Monkeypatching is preferred over the obvious alternatives:
        (a) Agentic RAG dispatches through PydanticAI's tool loop
        and RLM through the third-party `rlm` package, so a
        wrapper `VectorStore` would have to be plumbed through APIs
        we don't own; and (b) instrumenting `VectorStore` itself
        would put per-question state on a process-lifetime
        singleton, which breaks silently under the tenacity retry
        wrapping `_run_single_question`. Context-manager scoping
        makes the patch self-cleaning even on exception.

    Attributes:
        vector_store: The `VectorStore` instance being patched.
        retrieve_calls: One entry per `retrieve()` call, each a
            dict with `kwargs` and `results` keys.
        doc_fetches: Every doc ID passed to
            `get_document_from_doc_id()` during the capture window.
        _orig_retrieve: Saved reference to the unpatched
            `retrieve` method, restored on exit.
        _orig_get_doc: Saved reference to the unpatched
            `get_document_from_doc_id` method, restored on exit.
    """

    def __init__(self, vector_store):
        """Stash references to the original `VectorStore` methods.

        The patch isn't installed until `__enter__`, so constructing
        a `RetrievalCapture` outside a `with` block is harmless.

        Args:
            vector_store: The `VectorStore` instance to wrap.
        """
        self.vector_store = vector_store
        self.retrieve_calls: list[dict] = []
        self.doc_fetches: list[str] = []
        self._orig_retrieve = vector_store.retrieve
        self._orig_get_doc = vector_store.get_document_from_doc_id

    def __enter__(self):
        """Install the capturing wrappers and return `self`."""
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
        """Restore the original methods, even on exception."""
        self.vector_store.retrieve = self._orig_retrieve
        self.vector_store.get_document_from_doc_id = self._orig_get_doc
        return False


# ---------------------------------------------------------------------------
# QA system factory
# ---------------------------------------------------------------------------

def create_qa_system(config: ExperimentConfig):
    """Instantiate the `QASystem` subclass matching `config`.

    Each architecture's constructor internally creates a
    `VectorStore` singleton for the appropriate `corpus_type`, so
    clean/poisoned index selection is handled automatically.

    Args:
        config: Experiment specification. Drives which subclass is
            constructed and which `corpus_type` it loads.

    Returns:
        A ready-to-run `QASystem` subclass instance.

    Raises:
        ValueError: If `config.architecture` isn't one of the four
            supported values.
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
        from src.architectures.recursive_lm import RLM
        return RLM(corpus_type=config.corpus_type, **common_kwargs)

    else:
        raise ValueError(f"Unknown architecture: {config.architecture!r}")


# ---------------------------------------------------------------------------
# Core execution: single question
# ---------------------------------------------------------------------------

def _run_single_question(
    config: ExperimentConfig,
    query: dict,
    qa_system: QASystem,
    log_tag: str,
) -> QuestionResult:
    """Run one query and assemble its `QuestionResult`.

    Wraps `qa_system._run` in a `RetrievalCapture` context, then
    extracts retrieval-derived metadata (poison rank, gold ranks)
    and architecture-specific extras (doc fetches for Agentic and
    RLM, etc.).

    Args:
        config: Experiment configuration.
        query: Per-query record from `nq-questions.jsonl`. Expected
            keys: `query_id`, `question`, `correct_answer`,
            optional `target_answer`, optional `gold_doc_ids`.
        qa_system: Pre-instantiated `QASystem` subclass matching
            `config.architecture`.
        log_tag: Bracketed prefix from `make_log_tag` prepended to
            log lines.

    Returns:
        A populated `QuestionResult` (success path).

    Raises:
        Exception: Any error from the architecture or retrieval is
            propagated. The outer `run_single_question` wrapper
            catches it and records it in the result file's `error`
            field.

    Notes:
        Per-question outer timeout is enforced by the module-level
        `_run_single_question_retry_{signal,thread}_timed` wrappers
        (10 min, auto-dispatched by thread context). Per-LLM-call
        timeout lives on the OpenAI client itself (see
        `_LLM_CALL_TIMEOUT_SECONDS` in `src/architectures/utils.py`).
    """
    query_id: str = query['query_id']
    question_text: str = query['question']
    correct_answer: str = query.get('correct_answer', '')
    target_answer: Optional[str] = (
        query.get('target_answer') if config.attack_type != 'clean' else None
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
    gold_doc_ids: list[str] = query.get('gold_doc_ids', [])
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
    query: dict,  # {query_id, question, correct_answer, target_answer?, gold_doc_ids?}
    qa_system: QASystem,  # Pre-instantiated QASystem subclass
    question_num: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> QuestionResult:
    """Execute one query through the configured architecture, capturing errors.

    Wraps `_run_single_question` in retry + timeout (per-question
    10-min ceiling, signal-based on the main thread and
    multiprocessing-based on worker threads). On failure, returns
    a `QuestionResult` with the captured error rather than raising
    — one bad question cannot crash a worker or lose batch
    progress.

    Args:
        config: Experiment configuration.
        query: Per-query record (`{query_id, question,
            correct_answer, target_answer?, gold_doc_ids?}`).
        qa_system: Pre-instantiated `QASystem` subclass matching
            `config.architecture`.
        question_num: 1-indexed position of this query within the
            batch, used for log tags.
        batch_size: Total number of queries in the batch, used for
            log tags.

    Returns:
        A `QuestionResult` either on the success path or with
        `error` populated.

    Notes:
        Function name kept as `run_single_question` (rather than
        `run_single_query`) because it's referenced from Modal log
        history and downstream scripts; renaming would cascade —
        see CONVENTIONS.md.
    """
    query_id: str = query['query_id']
    question_text: str = query['question']
    correct_answer: str = query.get('correct_answer', '')
    target_answer: Optional[str] = (
        query.get('target_answer')
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
        return _run_single_question_retry_timed(config, query, qa_system, log_tag)
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
    query_ids: list[str],
    queries: dict[str, dict],  # query_id -> query record
    results_dir: str,  # Root results directory (e.g. /vol/results)
    modal_volume=None,  # Optional Modal Volume — reload before / commit after each query
) -> dict:
    """Process a batch of queries for one experiment, with checkpointing.

    Called by each Modal worker container. Resources (`VectorStore`,
    `QASystem`) are loaded once per container, then queries are
    processed sequentially with per-query JSON writes so that
    partial progress survives container preemption.

    When `modal_volume` is provided, the volume is reloaded before
    each query (to see results committed by other workers / retried
    attempts) and committed after each write (so partial progress
    is durable).

    Existing successful result files are skipped; existing error
    files are deleted and retried on the next pass; corrupt files
    are deleted and retried.

    Args:
        config: Experiment configuration.
        query_ids: Subset of query IDs this worker should process.
        queries: Map from `query_id` to its full per-query record.
            Records missing from this dict are recorded as errors
            rather than crashing the batch.
        results_dir: Root results directory (e.g. `/vol/results/`).
            One subdirectory per `experiment_id` is created
            underneath.
        modal_volume: Optional Modal Volume; when provided, reload
            and commit calls bracket each per-query write.

    Returns:
        Counts dict with keys `completed`, `skipped`, `errors`, and
        `total`.

    Notes:
        Function name kept as `run_question_batch` (rather than
        `run_query_batch`) because it's referenced from Modal log
        history and downstream scripts; renaming would cascade —
        see CONVENTIONS.md.
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
    qa_system: QASystem = create_qa_system(config)

    completed = 0
    skipped = 0
    errors = 0
    batch_size = len(query_ids)

    for q_idx, query_id in enumerate(query_ids, 1):
        log_tag = make_log_tag(config, query_id, q_idx, batch_size)
        result_path = os.path.join(exp_dir, f'{query_id}.json')

        # Refresh volume view to see writes from other workers / retried attempts.
        if modal_volume is not None:
            modal_volume.reload()

        # --- Checkpoint: skip already-completed queries ---
        # Only skip if the result file exists AND contains a successful result.
        # Error results (from previous rate-limit failures etc.) are deleted
        # so they get retried on the next run.
        if os.path.exists(result_path):
            try:
                with open(result_path) as _f:
                    _prev = json.loads(_f.read())
                if _prev.get('error') is None:
                    skipped += 1
                    print(f"{log_tag} Skipping already completed query")
                    continue
                # Previous result was an error — delete and retry.
                print(f"{log_tag} Deleting errored result file and retrying...")
                os.remove(result_path)
            except (json.JSONDecodeError, OSError):
                # Corrupt file — delete and retry.
                print(f"{log_tag} Deleting corrupt result file and retrying...")
                os.remove(result_path)

        query = queries.get(query_id)
        if query is None:
            # Missing query — record error and continue
            print(f"{log_tag} Missing query in queries dict - recording error")
            errors += 1
            err_result = QuestionResult(
                experiment_id=config.experiment_id,
                question_id=query_id,
                question_text='',
                correct_answer='',
                target_answer=None,
                system_answer='',
                error=f"Query ID {query_id!r} not found in queries dict",
            )
            with open(result_path, 'w') as f:
                f.write(err_result.to_json())
            if modal_volume is not None:
                modal_volume.commit()
            continue

        result = run_single_question(config, query, qa_system, q_idx, batch_size)

        # Write immediately — per-query checkpoint
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
        'total': len(query_ids),
    }
