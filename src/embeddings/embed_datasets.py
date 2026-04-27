import os
# quiet HF / transformers (must be done before importing transformers)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
import pickle

import modal
import transformers
import numpy as np


# Modal setup
app = modal.App('embed-datasets')
contriever_image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch', 'transformers', 'numpy', 'scikit-learn')
    .add_local_file('src/embeddings/embeddings.py', remote_path='/root/embeddings.py')
)
model_volume = modal.Volume.from_name('contriever-model', create_if_missing=True)
MODEL_DIR = '/vol/contriever'
EMBED_BATCH_SIZE = 1024


# facebook/contriever is a public model and does not require auth, so the
# from_pretrained calls below are unauthenticated. If you ever see an HF
# rate-limit (429) or authentication warning during the one-time download,
# the fix is to add a Modal Secret containing HF_TOKEN to this function's
# decorator and pass token=os.environ.get('HF_TOKEN') to both from_pretrained
# calls.
@app.function(
    image=contriever_image,
    volumes={MODEL_DIR: model_volume},
    timeout=60 * 10, # 10 minutes max
)
def download_model():
    """Download Contriever to the volume once. No-ops if already cached."""
    if os.path.exists(f'{MODEL_DIR}/config.json'):
        print("Model already cached on volume.")
        return
    from transformers import AutoTokenizer, AutoModel
    print("Downloading contriever to volume...")
    AutoTokenizer.from_pretrained('facebook/contriever').save_pretrained(MODEL_DIR)
    AutoModel.from_pretrained('facebook/contriever').save_pretrained(MODEL_DIR)
    model_volume.commit()
    print("Done.")


@app.function(
    image=contriever_image,
    gpu='L40S',
    volumes={MODEL_DIR: model_volume},
    timeout=60, # 1 minute max
)
def embed_text_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed a batch of text using the Contriever model."""
    from embeddings import Embedder
    embedder = Embedder(gpu=True, model_path=MODEL_DIR)
    return embedder.embed(texts)


@app.local_entrypoint()
def main():
    """Embed all datasets."""
    _EMBED_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = os.path.join(_EMBED_DIR, '..', 'data')

    # Download model (if not already cached)
    download_model.remote()

    # Create vector store directory (if it doesn't already exist)
    os.makedirs(os.path.join(_DATA_DIR, 'vector-store'), exist_ok=True)

    # Embed queries
    print("Embedding queries...")
    queries: dict[str, str] = {}
    with open(os.path.join(_DATA_DIR, 'original-datasets', 'nq', 'queries.jsonl'), 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            queries[line_dict['_id']] = line_dict['text']
    query_texts = list(queries.values())
    query_texts_batched = [
        query_texts[i:i + EMBED_BATCH_SIZE]
        for i in range(0, len(query_texts), EMBED_BATCH_SIZE)
    ]
    query_embeddings_batched = embed_text_batch.map(query_texts_batched)
    query_embeddings = [embedding for batch in query_embeddings_batched for embedding in batch]
    query_embeddings_dict = dict(zip(list(queries.keys()), query_embeddings))
    with open(os.path.join(_DATA_DIR, 'vector-store', 'nq-queries-embeddings.pkl'), 'wb') as f:
        pickle.dump(query_embeddings_dict, f)

    # Embed documents
    print("Embedding documents - this may take a while...")
    documents: dict[str, dict[str, str]] = {}
    with open(os.path.join(_DATA_DIR, 'original-datasets', 'nq', 'corpus.jsonl'), 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            documents[line_dict['_id']] = {'title': line_dict['title'], 'text': line_dict['text']}
    document_texts = [document['text'] for document in documents.values()]
    document_texts_batched = [
        document_texts[i:i + EMBED_BATCH_SIZE]
        for i in range(0, len(document_texts), EMBED_BATCH_SIZE)
    ]
    document_embeddings_batched = embed_text_batch.map(document_texts_batched)
    document_embeddings = [embedding for batch in document_embeddings_batched for embedding in batch]
    document_embeddings_dict = dict(zip(list(documents.keys()), document_embeddings))
    with open(os.path.join(_DATA_DIR, 'vector-store', 'nq-original-documents-embeddings.pkl'), 'wb') as f:
        pickle.dump(document_embeddings_dict, f)

    # Embed naively poisoned documents
    print("Embedding naively poisoned ('napo') documents...")
    napo_documents: dict[str, dict[str, str]] = {}
    with open(os.path.join(_DATA_DIR, 'experiment-datasets', 'nq-naive-poisoning', 'corpus.jsonl'), 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            if line_dict['_id'].startswith('doc'):
                continue
            napo_documents[line_dict['_id']] = {'title': line_dict['title'], 'text': line_dict['text']}
    napo_document_texts = [document['text'] for document in napo_documents.values()]
    napo_document_texts_batched = [
        napo_document_texts[i:i + EMBED_BATCH_SIZE]
        for i in range(0, len(napo_document_texts), EMBED_BATCH_SIZE)
    ]
    napo_document_embeddings_batched = embed_text_batch.map(napo_document_texts_batched)
    napo_document_embeddings = [embedding for batch in napo_document_embeddings_batched for embedding in batch]
    napo_document_embeddings_dict = dict(zip(list(napo_documents.keys()), napo_document_embeddings))
    with open(os.path.join(_DATA_DIR, 'vector-store', 'nq-naive-poisoned-documents-embeddings.pkl'), 'wb') as f:
        pickle.dump(napo_document_embeddings_dict, f)

    # Embed CorruptRAG-AK poisoned documents
    print("Embedding CorruptRAG-AK poisoned ('crak') documents...")
    crak_documents: dict[str, dict[str, str]] = {}
    with open(os.path.join(_DATA_DIR, 'experiment-datasets', 'nq-corruptrag-ak-poisoning', 'corpus.jsonl'), 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            if line_dict['_id'].startswith('doc'):
                continue
            crak_documents[line_dict['_id']] = {'title': line_dict['title'], 'text': line_dict['text']}
    crak_document_texts = [document['text'] for document in crak_documents.values()]
    crak_document_texts_batched = [
        crak_document_texts[i:i + EMBED_BATCH_SIZE]
        for i in range(0, len(crak_document_texts), EMBED_BATCH_SIZE)
    ]
    crak_document_embeddings_batched = embed_text_batch.map(crak_document_texts_batched)
    crak_document_embeddings = [embedding for batch in crak_document_embeddings_batched for embedding in batch]
    crak_document_embeddings_dict = dict(zip(list(crak_documents.keys()), crak_document_embeddings))
    with open(os.path.join(_DATA_DIR, 'vector-store', 'nq-corruptrag-ak-poisoned-documents-embeddings.pkl'), 'wb') as f:
        pickle.dump(crak_document_embeddings_dict, f)
