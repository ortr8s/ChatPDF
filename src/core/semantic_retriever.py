from typing import Iterable, List, Union
from sentence_transformers import SentenceTransformer
from torch import Tensor

import torch


class SemanticRetriever:
    def __init__(self, name: str) -> None:
        self.model = SentenceTransformer(name)
        self.corpus_embeddings: Union[List, Tensor] = []

    def update_corpus_embeddings(
        self,
        chunks_generator: Iterable[str],
        batch_size: int = 32
    ) -> None:
        batch_buffer: List[str] = []

        for chunk in chunks_generator:
            if isinstance(chunk, str): # skip control tokens
                continue

            text = "".join(chunk).replace("\n", " ")
            batch_buffer.append(text)

            if len(batch_buffer) >= batch_size:
                self._encode_batch(batch_buffer)
                batch_buffer = []

        if batch_buffer:
            self._encode_batch(batch_buffer)

    def _encode_batch(self, texts: List[str]) -> None:
        batch_embeddings = self.model.encode(
            texts,
            batch_size=len(texts),
            convert_to_tensor=True,
            show_progress_bar=False
        )

        if isinstance(self.corpus_embeddings, list):
            self.corpus_embeddings = batch_embeddings
        else:
            self.corpus_embeddings = torch.cat(
                (self.corpus_embeddings, batch_embeddings),
                dim=0
            )

    def encode(self, text: str) -> Tensor:
        return self.model.encode(text, convert_to_tensor=True)
