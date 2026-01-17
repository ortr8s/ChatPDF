from typing import List, Tuple
from src.utils.logger import Logger

import torch
from sentence_transformers import CrossEncoder

logger = Logger(__name__)


class ReRanker:
    def __init__(self, model_name: str, **kwargs):
        try:
            logger.log(f"Loading reranker model: {model_name}", "DEBUG")
            self.model = CrossEncoder(model_name, **kwargs)
            logger.log(f"Reranker model loaded: {model_name}", "DEBUG")
        except Exception as e:
            logger.log(f"Failed to load reranker model: {e}", "ERROR")
            raise

    def predict(
        self,
        query: str,
        documents: List[Tuple[int, str]],
        n_ans: int,
        batch_size: int = 32
    ) -> List[int]:
        if not documents:
            return []
        k = min(n_ans, len(documents))
        try:
            logger.log(
                f"Reranking {len(documents)} docs for query: {query[:50]}...",
                "DEBUG"
            )
            pairs = [[query, doc[1]] for doc in documents]
            scores = self.model.predict(
                pairs, 
                batch_size=batch_size,
                show_progress_bar=False
            )
            scores_tensor = torch.tensor(scores)
            _, top_indices = torch.topk(scores_tensor, k=k)
            result = [documents[i.item()][0] for i in top_indices]
            logger.log(f"Reranking complete, selected top {k}", "DEBUG")
            return result
        except Exception as e:
            logger.log(f"Error during reranking: {e}", "ERROR")
            raise
