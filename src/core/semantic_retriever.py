from typing import Iterable, List, Union
from sentence_transformers import SentenceTransformer
from torch import Tensor
from src.utils.logger import Logger

import torch

logger = Logger(__name__)


class SemanticRetriever:
    def __init__(self, model_name: str, **kwargs) -> None:
        try:
            logger.log(f"Loading semantic model: {model_name}", "DEBUG")
            self.model = SentenceTransformer(model_name, **kwargs)
            self.corpus_embeddings: Union[List, Tensor] = []
            logger.log(f"Semantic model loaded: {model_name}", "INFO")
        except Exception as e:
            logger.log(f"Failed to load semantic model: {e}", "ERROR")
            raise

    def encode(self, text: str) -> Tensor:
        return self.model.encode(text, convert_to_tensor=True)

    def get_similar(self, query_embedding, corpus_embedding, n):
        scores = self.model.similarity(query_embedding, corpus_embedding)[0]
        indices = torch.topk(scores, k=n)[1]
        return indices

    def update_corpus_embeddings(
        self,
        chunks_generator: Iterable[str],
        batch_size: int = 32
    ) -> None:
        batch_buffer: List[str] = []
        chunk_count = 0

        try:
            logger.log("Starting corpus embedding generation", "DEBUG")
            for chunk in chunks_generator:
                if isinstance(chunk, str):
                    if chunk.startswith("<s>") or chunk.startswith("<e>"):
                        continue
                    text = chunk.replace("\n", " ").strip()
                    if text:
                        batch_buffer.append(text)
                        chunk_count += 1

                        if len(batch_buffer) >= batch_size:
                            self._encode_batch(batch_buffer)
                            batch_buffer = []

            if batch_buffer:
                self._encode_batch(batch_buffer)

            logger.log(
                f"Corpus embeddings generated: {chunk_count} chunks",
                "INFO"
            )
        except Exception as e:
            logger.log(f"Error generating embeddings: {e}", "ERROR")
            raise

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
