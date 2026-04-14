"""
Create nq-questions.jsonl for experiment workers.

Merges three local sources into the unified format that the Modal workers
expect:
    - original-datasets/nq/queries.jsonl             (_id, text)
    - experiment-datasets/nq-correct-answers.jsonl   (query_id, correct_answer)
    - experiment-datasets/nq-incorrect-answers-poisoned-docs.jsonl
                                                     (query_id, incorrect_answer)

Output (written to experiment-datasets/):
    nq-questions.jsonl  — {query_id, question, correct_answer, target_answer} per line

Usage:
    cd workspace/data
    python create_questions.py
"""

import json
import os


def load_qrels(path: str) -> dict[str, list[str]]:
    """Load qrels TSV into {query_id: [corpus_id, ...]} mapping."""
    qrels: dict[str, list[str]] = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            query_id, corpus_id, _score = line.strip().split('\t')
            qrels.setdefault(query_id, []).append(corpus_id)
    return qrels


def main():
    # --- Load original queries ----------------------------------------------
    queries: dict[str, str] = {}
    with open('original-datasets/nq/queries.jsonl') as f:
        for line in f:
            line_dict = json.loads(line)
            queries[line_dict['_id']] = line_dict['text']
    print(f"Loaded {len(queries):,} queries")

    # --- Load correct answers -----------------------------------------------
    correct: dict[str, str] = {}
    with open('experiment-datasets/nq-correct-answers.jsonl') as f:
        for line in f:
            line_dict = json.loads(line)
            correct[line_dict['query_id']] = line_dict['correct_answer']
    print(f"Loaded {len(correct):,} correct answers")

    # --- Load target (incorrect) answers ------------------------------------
    targets: dict[str, str] = {}
    with open('experiment-datasets/nq-incorrect-answers-poisoned-docs.jsonl') as f:
        for line in f:
            line_dict = json.loads(line)
            targets[line_dict['query_id']] = line_dict['incorrect_answer']
    print(f"Loaded {len(targets):,} target answers")

    # --- Load gold-standard relevance judgments (qrels) ---------------------
    qrels = load_qrels('original-datasets/nq/qrels/test.tsv')
    print(f"Loaded qrels for {len(qrels):,} queries")

    # --- Merge and write ----------------------------------------------------
    question_ids = sorted(queries.keys(), key=lambda x: int(x.replace('test', '')))

    os.makedirs('experiment-datasets', exist_ok=True)

    qjsonl_path = os.path.join('experiment-datasets', 'nq-questions.jsonl')
    with open(qjsonl_path, 'w') as f:
        for qid in question_ids:
            record = {
                'query_id': qid,
                'question': queries[qid],
                'correct_answer': correct.get(qid, ''),
                'target_answer': targets.get(qid),
                'gold_doc_ids': qrels.get(qid, []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Wrote {qjsonl_path} ({len(question_ids):,} questions)")


if __name__ == '__main__':
    main()
