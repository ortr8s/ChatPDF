from typing import List, Dict, Tuple, Optional
from src.utils.logger import Logger

from src.core.lexical_retriever import LexicalRetriever
from src.core.semantic_retriever import SemanticRetriever
from src.core.reranker import ReRanker
from src.core.generation import Generator
from src.scraper.reader import get_chunks
from src.utils.lexical_utils import Lemmatizer
from src.utils.serializer import Serializer
from src.utils.config import get_config
from src.utils.prompt_utils import prepare_prompt

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
        config = get_config()
        self.config = config
        self.semantic_retriever = SemanticRetriever(
            semantic_retriever_model,
            device=self.config.get("biencoder.device", "auto")
        )
        self.reranker = ReRanker(
            model_name=reranker_model,
            device=self.config.get("reranker.device", "auto"))
        self.semantic_model_name = semantic_retriever_model
        self.lemmatizer = lemmatizer or Lemmatizer()
        self.lexical_retriever = LexicalRetriever(self.lemmatizer)

        self.generator = Generator(
            model_name=generator_model,
            temperature=config.get("llm.temperature", 0.7),
            top_p=config.get("llm.top_p", 0.9),
        )

        self.serializer = Serializer(cache_dir)

        self.tokenize_func = tokenize_func
        self.file_map: Dict[int, str] = {}
        self.corpus: List[str] = []
        self.is_indexed = False

    def load_from_cache(self) -> bool:
        try:
            logger.log("Checking for cached embeddings...", "INFO")
            embeddings, metadata = self.serializer.load_embeddings(
                self.semantic_model_name
            )

            if embeddings is None:
                logger.log("No valid cache found", "INFO")
                return False

            corpus = self.serializer.load_corpus()
            file_map = self.serializer.load_file_map()

            if corpus is None or file_map is None:
                logger.log(
                    "Incomplete cache (missing corpus or file map)",
                    "WARNING"
                )
                return False

            self.semantic_retriever.corpus_embeddings = embeddings
            self.corpus = corpus
            self.file_map = file_map

            logger.log(
                "Rebuilding BM25 index from cached corpus",
                "INFO"
            )
            self.lexical_retriever.build(self.corpus)

            self.is_indexed = True
            logger.log(
                f"Loaded {len(corpus)} documents from cache "
                f"({metadata.get('timestamp', 'unknown')})",
                "INFO"
            )
            return True

        except Exception as e:
            logger.log(
                f"Error loading from cache: {e}",
                "ERROR"
            )
            return False

    def ingest_documents(
        self,
        dir_path: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> int:
        logger.log(
            f"Starting document ingestion from {dir_path}",
            "INFO")
        chunk_count = 0
        current_file = None

        try:
            chunks_gen = get_chunks(
                dir_path, chunk_size, chunk_overlap, self.tokenize_func
            )

            for chunk in chunks_gen:
                if isinstance(chunk, str) and chunk.startswith("<s>"):
                    # Extract filename from <s>filename</s>
                    current_file = chunk[3:-4]
                    logger.log(
                        f"Processing file: {current_file}",
                        "DEBUG"
                    )
                    continue
                elif isinstance(chunk, str) and chunk.startswith("<e>"):
                    continue

                # Process actual text chunks (now strings, not lists)
                if isinstance(chunk, str) and chunk.strip():
                    self.corpus.append(chunk)
                    self.file_map[chunk_count] = current_file or "unknown"
                    chunk_count += 1

            # Build lexical index
            logger.log(f"Building BM25 index for {chunk_count} chunks", "INFO")
            self.lexical_retriever.build(self.corpus)

            # Build semantic embeddings
            logger.log("Generating semantic embeddings", "INFO")
            chunks_gen = get_chunks(
                dir_path, chunk_size, chunk_overlap, self.tokenize_func
            )
            self.semantic_retriever.update_corpus_embeddings(
                chunks_gen,
                batch_size=32
            )

            # Serialize embeddings for faster loading next time
            if self.config.get("cache.use_embeddings_cache", True):
                try:
                    logger.log("Caching embeddings to disk", "INFO")
                    self.serializer.save_embeddings(
                        self.semantic_retriever.corpus_embeddings,
                        self.corpus,
                        self.semantic_model_name
                    )
                    self.serializer.save_corpus(self.corpus)
                    self.serializer.save_file_map(self.file_map)
                    logger.log("Caching complete", "INFO")
                except Exception as e:
                    logger.log(f"Could not cache embeddings: {e}", "WARNING")

            self.is_indexed = True
            logger.log(f"Successfully ingested {chunk_count} chunks", "INFO")
            return chunk_count

        except Exception as e:
            logger.log(f"Error during document ingestion: {e}", "ERROR")
            raise

    def retrieve(
        self,
        query: str,
        top_k_lexical=None,
        top_k_semantic=None
    ) -> List[Tuple[int, str, str]]:
        if not self.is_indexed:
            msg = "Documents not indexed. Run ingest_documents first."
            logger.log(f"{msg}", "ERROR")
            raise ValueError()

        logger.log(f"Retrieving for query: {query}", "DEBUG")

        try:
            # Lexical retrieval (BM25)
            top_lexical = top_k_lexical or self.config.get(
                    "retrieval.top_k_lexical", 5
                )
            lexical_indices = self.lexical_retriever.search(
                query, n=top_lexical
            )
            # Semantic retrieval
            top_semantic = top_k_semantic or self.config.get(
                "retrieval.top_k_semantic", 5
            )
            query_embedding = self.semantic_retriever.encode(query)
            semantic_indices = self.semantic_retriever.get_similar(
                query_embedding,
                self.semantic_retriever.corpus_embeddings,
                top_semantic
            ).tolist()
            # Merge results (deduplicate)
            merged_indices = list(
                dict.fromkeys(lexical_indices + semantic_indices)
            )
            # Return with metadata
            results = []
            max_results = top_lexical + top_semantic
            for idx in merged_indices[:max_results]:
                if idx < len(self.corpus):
                    results.append((
                        idx,
                        self.corpus[idx],
                        self.file_map.get(idx, "unknown")
                    ))

            logger.log(f"Retrieved {len(results)} documents", "DEBUG")
            return results

        except Exception as e:
            logger.log(f"Error during retrieval: {e}", "ERROR")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Tuple[int, str, str]],
        top_k=None
    ) -> List[Tuple[int, str, str]]:
        if not documents:
            return []

        logger.log(f"Reranking {len(documents)} documents", "DEBUG")

        try:
            # Prepare data for reranker
            doc_tuples = [(idx, text) for idx, text, _ in documents]

            # Get reranked indices
            conf_top_k = self.config.get("retrieval.rerank_top_k")
            n_docs = min(top_k or conf_top_k, len(documents))
            top_doc_ids = self.reranker.predict(
                query, doc_tuples, n_docs
            )

            # Map document IDs back to positions in documents array
            # top_doc_ids are the chunk indices, need to find them in documents
            doc_id_to_pos = {doc[0]: pos for pos, doc in enumerate(documents)}
            reranked = [
                documents[doc_id_to_pos[doc_id]]
                for doc_id in top_doc_ids
                if doc_id in doc_id_to_pos
            ]
            logger.log(f"Reranked to top {len(reranked)} documents", "DEBUG")
            return reranked

        except Exception as e:
            logger.log(f"Error during reranking: {e}", "ERROR")
            raise

    def chat_stream(self, query: str, top_k_final: Optional[int] = None):
        logger.log(f"Processing chat query: {query}", "INFO")

        try:
            retrieved = self.retrieve(query=query)
            if not retrieved:
                yield "No relevant documents found."
                return

            # Use config value if top_k_final not specified
            if top_k_final is None:
                top_k_final = self.config.get("retrieval.rerank_top_k", 3)
            reranked = self.rerank(query, retrieved, top_k=top_k_final)
            sources = list(set([source for _, _, source in reranked]))
            user_prompt = prepare_prompt(reranked, query)
            messages = [
                {
                 "role": "system",
                 "content": self.config.get("llm.system_prompt")
                },
                {
                 "role": "user",
                 "content": f"{user_prompt}"
                }
            ]

            full_response_text = []
            for token in self.generator.generate_streaming(messages):
                full_response_text.append(token)
                yield token

            yield {"__sources__": sources}

        except Exception as e:
            logger.log(f"Error in chat pipeline: {e}", "ERROR")
            yield f"Error: {e}"
