from typing import List
from rank_bm25 import BM25Okapi
from src.utils.logger import Logger

logger = Logger(__name__)


class LexicalRetriever:
    def __init__(self, lemmatizer=None):
        self.model = None
        if lemmatizer is None:
            from ..utils.lexical_utils import Lemmatizer
            lemmatizer = Lemmatizer()
        self.lemmatizer = lemmatizer

    def build(self, corpus: List[str]) -> None:
        if not corpus:
            raise ValueError("Corpus cannot be empty")
        try:
            logger.log(
                f"Building BM25 index for {len(corpus)} documents",
                "DEBUG"
                )
            tokenized_corpus = self.lemmatizer.lemmatize(corpus)
            self.model = BM25Okapi(tokenized_corpus)
            logger.log("BM25 index built successfully", "DEBUG")
        except Exception as e:
            logger.log(f"Error building BM25 index: {e}", "ERROR")
            raise

    def search(self, query: str, n: int = 3) -> List[int]:
        if self.model is None:
            msg = "BM25 model not built. Call build() first."
            raise ValueError(msg)
        try:
            query_tokens = self.lemmatizer.tokenize_query(query)
            scores = self.model.get_scores(query_tokens)

            top_n_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:n]
            return top_n_indices
        except Exception as e:
            logger.log(f"Error searching BM25 index: {e}", "ERROR")
            raise
