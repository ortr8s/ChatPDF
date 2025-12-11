from typing import List, Dict

from src.core.lexical_retriever import LexicalRetriever
from src.core.semantic_retriever import SemanticRetriever
from src.core.reranker import ReRanker


class RAG:
    def __init__(
        self,
        reranker_model: str,
        semantic_retriever_model: str,
        tokenize_func
    ):
        self.semantic_retriever = SemanticRetriever(semantic_retriever_model)
        self.reranker = ReRanker(reranker_model)
        self.lexical_retriever = LexicalRetriever()
        self.tokenize_func = tokenize_func

        self.memory: List[str] = []
        self.file_map: Dict[int, str]  # start_index -> filename
    