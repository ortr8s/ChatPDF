import re
from typing import Optional

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
from src.utils.prompt_utils import (
    prepare_messages_with_few_shot,
    prepare_summary_messages
)

logger = Logger(__name__)

# Regex pattern to detect summarization intent
# Matches: "summarize document.pdf", "Summarize X.pdf"
SUMMARIZE_PATTERN = re.compile(
    r'^summarize\s+(?:document\s+)?(.+\.pdf)$',
    re.IGNORECASE
)


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

    def _detect_summarization_intent(self, query: str) -> Optional[str]:
        match = SUMMARIZE_PATTERN.match(query.strip())
        if match:
            filename = match.group(1).strip()
            logger.log(
                f"Detected summarization intent for: {filename}", "INFO"
            )
            return filename
        return None

    def _stream_summarization(self, filename: str):
        try:
            # Retrieve all chunks for the document
            chunks = self.kb.get_document_chunks(filename)

            if not chunks:
                yield f"File '{filename}' not found in the knowledge base. "
                yield "Please ensure the document has been ingested."
                return

            logger.log(
                f"Summarizing '{filename}' with {len(chunks)} chunks",
                "INFO"
            )

            # Get summarization system prompt from config
            default_prompt = (
                "You are a document summarization assistant. "
                "Provide a comprehensive summary of the provided document."
            )
            summarization_prompt = self.config.get(
                "llm.summarization_prompt",
                default_prompt
            )

            # Prepare messages for the LLM
            messages = prepare_summary_messages(
                system_prompt=summarization_prompt,
                chunks=chunks,
                filename=filename
            )

            # Stream the summarization response
            for token in self.generator.stream_answer(messages):
                yield token

            # Return source information
            yield {"__sources__": [filename]}
 
        except Exception as e:
            logger.log(f"Error during summarization: {e}", "ERROR")
            yield f"Error generating summary: {str(e)}"

    def stream_response(self, query: str, conversation_history: list = None):
        try:
            # Intent Detection: Check for summarization request
            summarize_filename = self._detect_summarization_intent(query)

            if summarize_filename:
                # Route to summarization pipeline
                logger.log(
                    f"Routing to summarization for: {summarize_filename}",
                    "DEBUG"
                )
                for token in self._stream_summarization(summarize_filename):
                    yield token
                return

            # Standard QA pipeline
            top_docs = self.search_engine.search(query, self.kb)
            if not top_docs:
                yield "No relevant documents found."
                return
            sources = list(set([src for _, _, src in top_docs]))

            messages = prepare_messages_with_few_shot(
                system_prompt=self.config.get("llm.system_prompt"),
                documents=top_docs,
                query=query,
                conversation_history=conversation_history
            )

            for token in self.generator.stream_answer(messages):
                yield token
            yield {"__sources__": sources}
        except Exception as e:
            logger.log(f"Error in chat pipeline: {e}", "ERROR")
            yield f"Error: {str(e)}"
