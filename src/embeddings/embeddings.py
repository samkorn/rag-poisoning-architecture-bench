"""Contriever embedder for batch and single-text encoding.

Wraps `facebook/contriever` via `transformers`. Defines the
`Embedder` class used by `embed_datasets.py` (offline corpus and
query embedding) and `vector_store.py` (live query-embedding
fallback).

Usage:
    python src/embeddings/embeddings.py

Output:
    Sanity-check cosine similarities printed to stdout for two
    hand-picked sentence pairs. No files written.

Notes:
    The HF / transformers env vars at the top of this file silence
    noisy progress bars and must be set before `import transformers`.
    `_mean_pooling` ignores padded positions when averaging token
    embeddings — without the mask, padding would bias the pooled
    vector toward the zero-token region of the embedding space.
"""

import os
# quiet HF / transformers (must be done before importing transformers)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import torch
import transformers
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class Embedder:
    """Wrap a Hugging Face encoder for batch and single-text embedding.

    Loads tokenizer + model from a local path or HF hub on init,
    moves them to CPU or CUDA, and exposes `embed` (batch) and
    `embed_single` (one sentence). Used for both offline corpus
    encoding (`embed_datasets.py`) and the live query-embedding
    fallback in `VectorStore`.

    Attributes:
        device: Torch device the model lives on (`cpu` or `cuda`).
        tokenizer: HF tokenizer loaded from `model_path`.
        model: HF encoder loaded from `model_path`, in eval mode.
    """

    def __init__(self, gpu: bool = False, model_path: str = 'facebook/contriever'):
        """Load the tokenizer and encoder, move them to the chosen device.

        Args:
            gpu: When `True`, place the model on CUDA. Defaults to
                CPU because corpus embedding runs once on Modal GPU
                and live query embedding from `VectorStore` is rare
                enough that CPU is fine.
            model_path: Local path or HF hub repo for the encoder.
                Defaults to `facebook/contriever`, the model the
                bench was built against.
        """
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings over the sequence dim, ignoring padding.

        Args:
            token_embeddings: Per-token hidden states from the
                encoder, shape `(B, T, H)`.
            attention_mask: Per-token mask from the tokenizer, shape
                `(B, T)`. Padded positions are 0; real tokens are 1.

        Returns:
            Pooled sentence embeddings, shape `(B, H)`. Padded
            positions are zeroed out before averaging so they don't
            bias the pooled vector toward the zero-token region of
            the embedding space.
        """
        reshaped_mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
        masked_embeddings = token_embeddings * reshaped_mask          # zero out padded positions
        summed_embeddings = masked_embeddings.sum(dim=1)              # (B, H)
        embedding_counts = reshaped_mask.sum(dim=1).clamp(min=1e-9)   # (B, 1), avoid div-by-zero
        return summed_embeddings / embedding_counts

    def embed(self, sentences: list[str]) -> list[np.ndarray]:
        """Embed a batch of sentences.

        Tokenizes with padding and truncation, runs the encoder
        under `torch.no_grad`, then mean-pools and detaches each
        row to a numpy array.

        Args:
            sentences: Input sentences, all encoded together as one
                batch.

        Returns:
            One numpy embedding per input sentence, in input order.
        """
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            token_level_embeddings = self.model(**inputs)
        document_embeddings_array = self._mean_pooling(token_level_embeddings[0], inputs['attention_mask'])
        document_embeddings = [tensor.cpu().detach().numpy() for tensor in document_embeddings_array]
        return document_embeddings

    def embed_single(self, sentence: str) -> np.ndarray:
        """Embed a single sentence, returning its numpy vector.

        Args:
            sentence: Input sentence.

        Returns:
            Numpy embedding of shape `(1, H)`. Note the leading
            singleton batch dimension — callers that want a flat
            vector should `.squeeze()` or index `[0]`.
        """
        inputs = self.tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            token_level_embeddings = self.model(**inputs)
        document_embedding_array = self._mean_pooling(token_level_embeddings[0], inputs['attention_mask'])
        return document_embedding_array.cpu().detach().numpy()


if __name__ == "__main__":
    print("=== Sanity Check: embeddings ===")
    sentences = [
        "Where was Marie Curie born?",
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
        "Michael Collins was an American astronaut and the Command Module Pilot of the Apollo 11 mission."
    ]
    embedder = Embedder()
    embeddings = embedder.embed(sentences)
    print(embeddings)
    print("--------------------------------")
    print(embeddings[0].shape)
    embeddings = [embedding.reshape(1, -1) for embedding in embeddings]
    print(embeddings[0].shape)
    print("--------------------------------")
    print("Similarity between related sentences:")
    print(f"0<->1: {cosine_similarity(embeddings[0], embeddings[1])[0][0]:.4f}")
    print(f"0<->2: {cosine_similarity(embeddings[0], embeddings[2])[0][0]:.4f}")
    print(f"1<->2: {cosine_similarity(embeddings[1], embeddings[2])[0][0]:.4f}")
    print("--------------------------------")
    print("Similarity between unrelated sentences:")
    print(f"0<->3: {cosine_similarity(embeddings[0], embeddings[3])[0][0]:.4f}")
    print(f"1<->3: {cosine_similarity(embeddings[1], embeddings[3])[0][0]:.4f}")
    print(f"2<->3: {cosine_similarity(embeddings[2], embeddings[3])[0][0]:.4f}")
