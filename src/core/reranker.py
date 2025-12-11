from typing import List, Tuple

import torch
from sentence_transformers import CrossEncoder


class ReRanker:
    def __init__(self, model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def predict(
        self,
        query: str,
        documents: List[Tuple[int, str]],
        n_ans: int
    ) -> List[int]:
        if n_ans > len(documents):
            raise ValueError(
                "Number of answers cannot be bigger than number of documents"
            )

        pairs = [[query, doc[1]] for doc in documents]
        scores = self.model.predict(pairs)

        scores_tensor = torch.tensor(scores)
        _, top_indices = torch.topk(scores_tensor, k=n_ans)

        return [documents[i][0] for i in top_indices]
