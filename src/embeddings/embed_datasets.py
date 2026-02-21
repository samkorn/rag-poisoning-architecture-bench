import os
# quiet HF / transformers (must be done before importing transformers)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
import modal
import pickle
import numpy as np
import transformers
from dotenv import load_dotenv; load_dotenv()


# Modal setup
app = modal.App('embed-datasets')
contriever_image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch', 'transformers', 'numpy', 'scikit-learn', 'python-dotenv')
    .add_local_file('embeddings.py', remote_path='/root/embeddings.py')
)
model_volume = modal.Volume.from_name('contriever-model', create_if_missing=True)
MODEL_DIR = '/vol/contriever'
EMBED_BATCH_SIZE = 1024


@app.function(
    image=contriever_image,
    secrets=[modal.Secret.from_name('huggingface-rag-poisoning')],
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
    AutoTokenizer.from_pretrained('facebook/contriever', token=os.environ.get('HF_TOKEN')).save_pretrained(MODEL_DIR)
    AutoModel.from_pretrained('facebook/contriever', token=os.environ.get('HF_TOKEN')).save_pretrained(MODEL_DIR)
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
    # Download model (if not already cached)
    download_model.remote()

    # Create vector store directory (if it doesn't already exist)
    os.makedirs('../data/vector-store', exist_ok=True)

    # Embed queries
    print("Embedding queries...")
    queries: dict[str, str] = {}
    with open('../data/original-datasets/nq/queries.jsonl', 'r') as f:
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
    with open('../data/vector-store/nq-queries-embeddings.pkl', 'wb') as f:
        pickle.dump(query_embeddings_dict, f)
    
    # Embed documents
    print("Embedding documents - this may take a while...")
    documents: dict[str, dict[str, str]] = {}
    with open('../data/original-datasets/nq/corpus.jsonl', 'r') as f:
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
    with open('../data/vector-store/nq-original-documents-embeddings.pkl', 'wb') as f:
        pickle.dump(document_embeddings_dict, f)
    
    # Embed naively poisoned documents
    print("Embedding naively poisoned ('napo') documents...")
    napo_documents: dict[str, dict[str, str]] = {}
    with open('../data/experiment-datasets/nq-naive-poisoning/corpus.jsonl', 'r') as f:
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
    with open('../data/vector-store/nq-naive-poisoned-documents-embeddings.pkl', 'wb') as f:
        pickle.dump(napo_document_embeddings_dict, f)
    
    # Embed poisonedrag poisoned documents
    print("Embedding poisonedrag poisoned ('adpo') documents...")
    adpo_documents: dict[str, dict[str, str]] = {}
    with open('../data/experiment-datasets/nq-poisonedrag-poisoning/corpus.jsonl', 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            if line_dict['_id'].startswith('doc'):
                continue
            adpo_documents[line_dict['_id']] = {'title': line_dict['title'], 'text': line_dict['text']}
    adpo_document_texts = [document['text'] for document in adpo_documents.values()]
    adpo_document_texts_batched = [
        adpo_document_texts[i:i + EMBED_BATCH_SIZE]
        for i in range(0, len(adpo_document_texts), EMBED_BATCH_SIZE)
    ]
    adpo_document_embeddings_batched = embed_text_batch.map(adpo_document_texts_batched)
    adpo_document_embeddings = [embedding for batch in adpo_document_embeddings_batched for embedding in batch]
    adpo_document_embeddings_dict = dict(zip(list(adpo_documents.keys()), adpo_document_embeddings))
    with open('../data/vector-store/nq-poisonedrag-poisoned-documents-embeddings.pkl', 'wb') as f:
        pickle.dump(adpo_document_embeddings_dict, f)
