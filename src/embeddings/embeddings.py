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
    def __init__(self, gpu: bool = False, model_path: str = 'facebook/contriever'):
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings over the sequence dim, ignoring padding."""
        reshaped_mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
        masked_embeddings = token_embeddings * reshaped_mask          # zero out padded positions
        summed_embeddings = masked_embeddings.sum(dim=1)              # (B, H)
        embedding_counts = reshaped_mask.sum(dim=1).clamp(min=1e-9)   # (B, 1), avoid div-by-zero
        return summed_embeddings / embedding_counts

    def embed(self, sentences: list[str]) -> list[np.ndarray]:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            token_level_embeddings = self.model(**inputs)
        document_embeddings_array = self._mean_pooling(token_level_embeddings[0], inputs['attention_mask'])
        document_embeddings = [tensor.cpu().detach().numpy() for tensor in document_embeddings_array]
        return document_embeddings
    
    def embed_single(self, sentence: str) -> np.ndarray:
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
