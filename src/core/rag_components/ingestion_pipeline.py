from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .knowledge_base import KnowledgeBase
from src.utils.logger import Logger
from src.core.semantic_retriever import SemanticRetriever
from src.core.lexical_retriever import LexicalRetriever
from src.scraper.reader import get_chunks
logger = Logger(__name__)


class IngestionPipeline:
    def __init__(self, tokenize_func, config):
        self.tokenize_func = tokenize_func
        self.config = config

    def run(self,
            dir_path: str,
            kb: KnowledgeBase,
            semantic_retriever: SemanticRetriever,
            lexical_retriever: LexicalRetriever
            ):
        logger.log(f"Starting ingestion for: {dir_path}", "INFO")
        files_to_process = self._determine_strategy(dir_path, kb)
        if files_to_process is None:
            logger.log("Full re-index triggered. Clearing old data.", "INFO")
            kb.corpus = []
            kb.ingested_file_names = []
            kb.embeddings = None
            kb.file_map = {}
            kb.is_indexed = False
        if files_to_process is not None and len(files_to_process) == 0:
            logger.log("Index up to date.", "INFO")
            return
        chunks_gen = get_chunks(
            dir_path,
            self.config.get("retrieval.chunk_size", 512),
            self.config.get("retrieval.chunk_overlap", 64),
            self.tokenize_func,
            specific_files=files_to_process
        )

        new_chunks, new_file_map = self._process_chunks(
            chunks_gen,
            len(kb.corpus),
            kb.ingested_file_names
        )
        if not new_chunks:
            return
        logger.log(f"Embedding {len(new_chunks)} new chunks", "INFO")
        semantic_retriever.update_corpus_embeddings(
            new_chunks,
            self.config.get("retrieval.batch_size", 32)
        )
        new_embeddings = semantic_retriever.corpus_embeddings
        full_corpus = kb.corpus + new_chunks
        lexical_retriever.build(full_corpus)
        kb.update_data(new_chunks, new_file_map, new_embeddings)
        kb.save()

    def _determine_strategy(
            self,
            dir_path: str,
            kb: KnowledgeBase
            ) -> Optional[List[str]]:
        disk_files = {p.name for p in Path(dir_path).glob("*.pdf")}
        if not kb.is_indexed:
            return None
        cached_files = set(kb.ingested_file_names)
        auto_invalidate = self.config.get("cache.auto_invalidate", True)
        if auto_invalidate:
            deleted_files = cached_files - disk_files
            if deleted_files:
                logger.log(
                    f"Cache invalidation: Detected deleted "
                    f"files {deleted_files}. Full reindex.",
                    "WARNING"
                )
                return None
        new_files = list(disk_files - cached_files)
        if new_files:
            logger.log(
                f"Found {len(new_files)} new files: {new_files}",
                "DEBUG"
            )
            return new_files
        if not auto_invalidate or len(disk_files) == len(cached_files):
            logger.log("Cache valid. Skipping ingestion.", "DEBUG")
            return []
        return []

    def _sanitize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        sanitized = []
        for char in text:
            code = ord(char)
            # Skip control characters (except newline, tab, space)
            if code < 32 and code not in (9, 10):
                continue
            # Skip DEL character
            if code == 127:
                continue
            # Skip surrogate pairs (U+D800 to U+DFFF) - invalid in isolation
            if 0xD800 <= code <= 0xDFFF:
                continue
            # Replace private use area characters with space
            if 0xE000 <= code <= 0xF8FF:
                sanitized.append(' ')
                continue
            sanitized.append(char)
        # Normalize whitespace
        result = " ".join("".join(sanitized).split())
        return result.strip()

    def _process_chunks(
            self,
            chunks_gen,
            start_idx: int,
            ingested_names: List
            ) -> Tuple[List[str], Dict[int, str]]:
        new_chunks = []
        new_map = {}
        current_file = None
        idx = start_idx
        for chunk in chunks_gen:
            if isinstance(chunk, str):
                if chunk.startswith("<s>"):
                    current_file = chunk[3:-4]
                    if current_file not in ingested_names:
                        ingested_names.append(current_file)
                    continue
                if chunk.startswith("<e>"):
                    continue
                clean_chunk = self._sanitize_text(chunk)
                if clean_chunk and len(clean_chunk) > 3:
                    new_chunks.append(clean_chunk)
                    new_map[idx] = current_file or "unknown"
                    idx += 1
        return new_chunks, new_map
