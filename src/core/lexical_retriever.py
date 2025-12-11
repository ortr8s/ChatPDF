from typing import List

from rank_bm25 import BM25Okapi


class LexicalRetriever:
    def __init__(self):
        self.model = None

    def build(self, tokenized_corpus: List[List[str]]):
        self.model = BM25Okapi(tokenized_corpus)

    def search(self, query_tokens: List[str], n: int = 3) -> List[List[str]]:
        return self.model.get_top_n(query_tokens, self.model.corpus, n=n)
