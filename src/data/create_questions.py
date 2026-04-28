"""Build `nq-questions.jsonl`, the per-query record file used by workers.

Merges four local sources into one record per query:

  * `original-datasets/nq/queries.jsonl` (`_id`, `text`)
  * `original-datasets/nq/qrels/test.tsv` (gold doc-id mapping)
  * `experiment-datasets/nq-correct-answers.jsonl` (`query_id`,
    `correct_answer`)
  * `experiment-datasets/nq-incorrect-answers-poisoned-docs.jsonl`
    (`query_id`, `incorrect_answer`)

Prerequisites:
    All three input files above must exist (run `create_correct_answers.py`
    and `create_incorrect_answers_poisoned_docs.py` first).

Usage:
    python src/data/create_questions.py

Output:
    `src/data/experiment-datasets/nq-questions.jsonl` — one
    `{query_id, question, correct_answer, target_answer}` record per
    line.

Notes:
    The filename keeps the legacy `question` prefix even though the
    record is logically a query (id + question text + answers). The
    name is referenced from analysis notebooks, paper tables, and
    upstream BEIR conventions, so it's frozen — see CONVENTIONS.md.
"""

import json
import os

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_qrels(path: str) -> dict[str, list[str]]:
    """Load a BEIR qrels TSV into a `{query_id: [corpus_id, ...]}` mapping.

    Skips the header row. Multiple corpus IDs per query are
    accumulated in insertion order.

    Args:
        path: Path to the qrels TSV (header `query-id corpus-id score`).

    Returns:
        Map from query ID to its list of gold corpus IDs.
    """
    qrels: dict[str, list[str]] = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            query_id, corpus_id, _score = line.strip().split('\t')
            qrels.setdefault(query_id, []).append(corpus_id)
    return qrels


def main():
    """Merge the four input sources and write `nq-questions.jsonl`.

    Reads queries, correct answers, target (incorrect) answers, and
    gold qrels; sorts queries by trailing test-ID integer; writes
    one merged record per line.
    """
    # --- Load original queries ----------------------------------------------
    queries: dict[str, str] = {}
    with open(os.path.join(_DATA_DIR, 'original-datasets', 'nq', 'queries.jsonl')) as f:
        for line in f:
            line_dict = json.loads(line)
            queries[line_dict['_id']] = line_dict['text']
    print(f"Loaded {len(queries):,} queries")

    # --- Load correct answers -----------------------------------------------
    correct: dict[str, str] = {}
    with open(os.path.join(_DATA_DIR, 'experiment-datasets', 'nq-correct-answers.jsonl')) as f:
        for line in f:
            line_dict = json.loads(line)
            correct[line_dict['query_id']] = line_dict['correct_answer']
    print(f"Loaded {len(correct):,} correct answers")

    # --- Load target (incorrect) answers ------------------------------------
    targets: dict[str, str] = {}
    with open(os.path.join(_DATA_DIR, 'experiment-datasets', 'nq-incorrect-answers-poisoned-docs.jsonl')) as f:
        for line in f:
            line_dict = json.loads(line)
            targets[line_dict['query_id']] = line_dict['incorrect_answer']
    print(f"Loaded {len(targets):,} target answers")

    # --- Load gold-standard relevance judgments (qrels) ---------------------
    qrels = load_qrels(os.path.join(_DATA_DIR, 'original-datasets', 'nq', 'qrels', 'test.tsv'))
    print(f"Loaded qrels for {len(qrels):,} queries")

    # --- Merge and write ----------------------------------------------------
    query_ids = sorted(queries.keys(), key=lambda x: int(x.replace('test', '')))

    os.makedirs(os.path.join(_DATA_DIR, 'experiment-datasets'), exist_ok=True)

    # Filename keeps the "nq-questions" prefix; the file is a list of full
    # query records (id + question text + answers + gold_doc_ids).
    qjsonl_path = os.path.join(_DATA_DIR, 'experiment-datasets', 'nq-questions.jsonl')
    with open(qjsonl_path, 'w') as f:
        for query_id in query_ids:
            record = {
                'query_id': query_id,
                'question': queries[query_id],
                'correct_answer': correct.get(query_id, ''),
                'target_answer': targets.get(query_id),
                'gold_doc_ids': qrels.get(query_id, []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Wrote {qjsonl_path} ({len(query_ids):,} questions)")


if __name__ == '__main__':
    main()
