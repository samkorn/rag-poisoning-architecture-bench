import os
# quiet HF / transformers (must be done before importing transformers)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import torch
import transformers
import numpy as np
from dotenv import load_dotenv; load_dotenv()
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class Embedder:
    def __init__(self, gpu: bool = False, model_path: str = 'facebook/contriever'):
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=os.environ.get('HF_TOKEN'))
        self.model = AutoModel.from_pretrained(model_path, token=os.environ.get('HF_TOKEN')).to(self.device)
        self.model.eval()

    def _mean_pooling(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.) # zero out padding tokens
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None] # sum up token embeddings and divide by number of non-padding tokens
        return sentence_embeddings

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
