from src.core.rag_components.knowledge_base import KnowledgeBase
from src.core.lexical_retriever import LexicalRetriever
from src.core.rag_components.ingestion_pipeline import IngestionPipeline
from src.core.rag_components.search_engine import SearchEngine
from src.core.reranker import ReRanker
from src.core.semantic_retriever import SemanticRetriever
from src.core.text_generator import Generator
from src.utils.lexical_utils import Lemmatizer
from src.utils.logger import Logger
from src.utils.config import get_config
from src.utils.prompt_utils import prepare_prompt, prepare_messages_with_few_shot

logger = Logger(__name__)


class RAG:
    def __init__(
        self,
        reranker_model: str,
        semantic_retriever_model: str,
        generator_model: str,
        tokenize_func,
        lemmatizer: Lemmatizer,
        cache_dir: str
    ):
        self.config = get_config()
        self.kb = KnowledgeBase(cache_dir, semantic_retriever_model)
        self.lemmatizer = lemmatizer or Lemmatizer()
        self.semantic_retriever = SemanticRetriever(
            semantic_retriever_model,
            device=self.config.get("biencoder.device", "auto")
        )
        self.reranker = ReRanker(
            model_name=reranker_model,
            device=self.config.get("reranker.device", "auto")
        )
        self.generator = Generator(
            model_name=generator_model,
            temperature=self.config.get("llm.temperature", 0.7),
            top_p=self.config.get("llm.top_p", 0.9),
        )
        self.lexical_retriever = LexicalRetriever(self.lemmatizer)
        self.ingestion = IngestionPipeline(tokenize_func, self.config)
        self.search_engine = SearchEngine(
            self.lexical_retriever,
            self.semantic_retriever,
            self.reranker,
            self.config
        )
        self._load_state()

    def _load_state(self):
        if self.kb.load():
            self.lexical_retriever.build(self.kb.corpus)
            self.semantic_retriever.corpus_embeddings = self.kb.embeddings

    def ingest_documents(self, dir_path: str):
        self.ingestion.run(
            dir_path,
            self.kb,
            self.semantic_retriever,
            self.lexical_retriever
        )
        return len(self.kb.corpus)

    def stream_response(self, query: str):
        try:
            top_docs = self.search_engine.search(query, self.kb)
            if not top_docs:
                yield "No relevant documents found."
                return
            sources = list(set([src for _, _, src in top_docs]))

            messages = prepare_messages_with_few_shot(
                system_prompt=self.config.get("llm.system_prompt"),
                documents=top_docs,
                query=query
            )

            for token in self.generator.stream_answer(messages):
                yield token
            yield {"__sources__": sources}
        except Exception as e:
            logger.log(f"Error in chat pipeline: {e}", "ERROR")
            yield f"Error: {str(e)}"
