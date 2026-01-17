from typing import List, Tuple
from src.core.lexical_retriever import LexicalRetriever
from src.core.semantic_retriever import SemanticRetriever
from src.core.reranker import ReRanker
from .knowledge_base import KnowledgeBase


class SearchEngine:
    def __init__(
            self,
            lexical: LexicalRetriever,
            semantic: SemanticRetriever,
            reranker: ReRanker,
            config
            ):
        self.lexical = lexical
        self.semantic = semantic
        self.reranker = reranker
        self.config = config

    def search(
            self,
            query: str,
            kb: KnowledgeBase,
            top_k_lexical=None,
            top_k_semantic=None
            ) -> List[Tuple[int, str, str]]:
        if not kb.is_indexed:
            raise ValueError("KnowledgeBase is empty. Ingest documents first.")
        top_lex = top_k_lexical or self.config.get(
            "retrieval.top_k_lexical",
            5
        )
        lex_indices = self.lexical.search(query, n=top_lex)
        top_sem = top_k_semantic or self.config.get(
            "retrieval.top_k_semantic",
            5
        )
        query_emb = self.semantic.encode(query)
        sem_indices = self.semantic.get_similar(
            query_emb, kb.embeddings, top_sem
        ).tolist()
        merged_indices = list(dict.fromkeys(lex_indices + sem_indices))
        candidates = []
        for idx in merged_indices:
            if idx < len(kb.corpus):
                candidates.append(
                    (idx, kb.corpus[idx], kb.file_map.get(idx, "unknown"))
                    )

        return self._rerank(query, candidates)

    def _rerank(
            self,
            query: str,
            docs: List[Tuple[int, str, str]]
            ) -> List[Tuple[int, str, str]]:
        if not docs: 
            return []
        top_k = self.config.get("retrieval.rerank_top_k", 3)
        doc_tuples = [(idx, text) for idx, text, _ in docs]
        top_indices = self.reranker.predict(
            query,
            doc_tuples,
            min(top_k, len(docs))
            )
        lookup = {d[0]: d for d in docs}
        return [lookup[uid] for uid in top_indices if uid in lookup]
