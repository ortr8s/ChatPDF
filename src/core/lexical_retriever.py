from typing import List
from rank_bm25 import BM25Okapi


class LexicalRetriever:
    def __init__(self, lemmatizer):
        self.model = None
        self.lemmatizer = lemmatizer

    def build(self, corpus: List[str], lemmatize) -> None:
        tokenized_corpus = self.lemmatizer.lemmatize(corpus)
        self.model = BM25Okapi(tokenized_corpus)

    def search(self, query: str, n: int = 3) -> List[int]:
        if self.model is None:
            raise ValueError("The BM25 model has not been built. Call 'build' with a corpus first.")
        query_tokens = self.lemmatizer.tokenize_query(query)
        scores = self.model.get_scores(query_tokens)

        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

        return top_n_indices
