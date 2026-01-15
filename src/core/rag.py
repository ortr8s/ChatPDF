from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from src.utils.logger import Logger

from src.core.lexical_retriever import LexicalRetriever
from src.core.semantic_retriever import SemanticRetriever
from src.core.reranker import ReRanker
from src.core.text_generator import Generator
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
        self.config = None
        self.semantic_retriever = None
        self.reranker = None
        self.generator = None
        self._prepare_llms(
            semantic_retriever_model,
            reranker_model,
            generator_model
        )
        self.semantic_model_name = semantic_retriever_model
        self.lemmatizer = lemmatizer or Lemmatizer()
        self.lexical_retriever = LexicalRetriever(self.lemmatizer)
        self.serializer = Serializer(cache_dir)
        self.tokenize_func = tokenize_func
        self.file_map: Dict[int, str] = {}
        self.ingested_file_names = []
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
            self.ingested_file_names = metadata.get("file_names", [])
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
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> int:
        logger.log(f"Starting ingestion for: {dir_path}", "INFO")
        files_to_process = self._determine_processing_strategy(dir_path)
        if files_to_process is not None and len(files_to_process) == 0:
            logger.log("Index is up to date. No processing needed.", "INFO")
            return len(self.corpus)
        chunks_gen = get_chunks(
            dir_path,
            chunk_size or self.config.get("retrieval.chunk_size", 512),
            chunk_overlap or self.config.get("retrieval.chunk_overlap", 64),
            self.tokenize_func,
            specific_files=files_to_process
        )
        new_text_chunks = self._consume_chunks(chunks_gen)
        self._update_indices(new_text_chunks)
        self._persist_to_cache()
        self.is_indexed = True
        logger.log(f"Ingestion complete. Total Corpus: {len(self.corpus)}", "INFO")
        return len(self.corpus)

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

    def stream_response(self, query: str, top_k_final: Optional[int] = None):
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
            for token in self.generator.stream_answer(messages):
                yield token
            yield {"__sources__": sources}
        except Exception as e:
            logger.log(f"Error in chat pipeline: {e}", "ERROR")
            yield f"Error: {e}"

    def _prepare_llms(self, semantic_retriever, reranker, generator):
        config = get_config()
        self.config = config
        self.semantic_retriever = SemanticRetriever(
            semantic_retriever,
            device=self.config.get("biencoder.device", "auto")
        )
        self.reranker = ReRanker(
            model_name=reranker,
            device=self.config.get("reranker.device", "auto")
        )
        self.generator = Generator(
            model_name=generator,
            temperature=config.get("llm.temperature", 0.7),
            top_p=config.get("llm.top_p", 0.9),
        )

    def _determine_processing_strategy(self, dir_path: str) -> Optional[List[str]]:
        if not self.config.get("cache.auto_invalidate", True):
            return None
        cache_info = self.serializer.get_cache_info()
        if not cache_info.get("cached"):
            return None
        try:
            cached_filenames = {Path(f).name for f in cache_info.get("file_names", [])}
            disk_files_map = {
                p.name: str(p) for p in Path(dir_path).glob("*.pdf")
            }
            current_filenames = set(disk_files_map.keys())
            new_files = list(current_filenames - cached_filenames)
            if not new_files and len(disk_files_map) == len(cached_filenames):
                logger.log("No new files found. Loading existing cache.", "INFO")
                self.load_from_cache()
                return []
            if new_files and self.load_from_cache():
                logger.log(f"Incremental Update: Found {len(new_files)} new files.", "INFO")
                return new_files
            logger.log("Cache invalid or load failed. Performing full ingestion.", "WARNING")
            return None

        except Exception as e:
            logger.log(f"Error determining strategy: {e}. Defaulting to full ingest.", "ERROR")
            return None

    def _consume_chunks(self, chunks_generator) -> List[str]:
        new_text_chunks = []
        chunk_count = len(self.corpus)
        current_file = None

        for chunk in chunks_generator:
            if isinstance(chunk, str):
                if chunk.startswith("<s>"):
                    current_file = chunk[3:-4]
                    if current_file not in self.ingested_file_names:
                        self.ingested_file_names.append(current_file)
                    logger.log(f"Processing file: {current_file}", "DEBUG")
                    continue
                elif chunk.startswith("<e>"):
                    continue
                elif chunk.strip():
                    self.corpus.append(chunk)
                    self.file_map[chunk_count] = current_file or "unknown"
                    chunk_count += 1
                    new_text_chunks.append(chunk)
        return new_text_chunks

    def _update_indices(self, new_text_chunks: List[str]):
        logger.log(f"Building BM25 index for {len(self.corpus)} chunks", "INFO")
        self.lexical_retriever.build(self.corpus)
        if not new_text_chunks:
            logger.log("No new chunks to embed.", "DEBUG")
            return

        logger.log(f"Generating embeddings for {len(new_text_chunks)} new chunks", "INFO")
        old_embeddings = self.semantic_retriever.corpus_embeddings
        self.semantic_retriever.update_corpus_embeddings(
            new_text_chunks,
            self.config.get("retrieval.batch_size", 32)
        )
        new_embeddings = self.semantic_retriever.corpus_embeddings
        if new_embeddings is None:
            logger.log("New embeddings generation returned None.", "WARNING")
            self.semantic_retriever.corpus_embeddings = old_embeddings
            return
        if old_embeddings is not None and len(old_embeddings) > 0:
            if old_embeddings.device != new_embeddings.device:
                old_embeddings = old_embeddings.to(new_embeddings.device)
            self.semantic_retriever.corpus_embeddings = torch.cat(
                [old_embeddings, new_embeddings], 
                dim=0
            )
        else:
            self.semantic_retriever.corpus_embeddings = new_embeddings

    def _persist_to_cache(self):
        if not self.config.get("cache.use_embeddings_cache", True):
            return

        try:
            logger.log("Saving state to cache...", "INFO")
            self.serializer.save_embeddings(
                self.semantic_retriever.corpus_embeddings,
                self.corpus,
                self.semantic_model_name,
                {"file_names": self.ingested_file_names}
            )
            self.serializer.save_corpus(self.corpus)
            self.serializer.save_file_map(self.file_map)
            logger.log("Cache save complete", "INFO")
        except Exception as e:
            logger.log(f"Failed to save cache: {e}", "WARNING")
