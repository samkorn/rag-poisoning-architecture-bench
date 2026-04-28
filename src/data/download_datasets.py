"""Downloads BEIR datasets (NQ, MS MARCO, HotpotQA) into `src/data/original-datasets/`.

Skips any dataset whose target directory already exists, so re-running
is a no-op once the data is on disk.

Usage:
    python src/data/download_datasets.py

Output:
    `src/data/original-datasets/{nq,msmarco,hotpotqa}/` — each
    directory contains BEIR's `corpus.jsonl`, `queries.jsonl`, and
    `qrels/test.tsv`.

Notes:
    Adapted from PoisonedRAG. Only NQ is used by the rest of the
    pipeline; MS MARCO and HotpotQA are downloaded for parity with
    the upstream script but currently unused.
"""

from beir import util
import os

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Download and save dataset
datasets = ['nq', 
            'msmarco', 
            'hotpotqa']
for dataset in datasets:
    url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
    out_dir = os.path.join(_DATA_DIR, 'original-datasets')
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)

