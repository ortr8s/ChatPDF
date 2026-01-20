import torch
from typing import Optional, Dict, List
from src.utils.serializer import Serializer
from src.utils.logger import Logger

logger = Logger(__name__)


class KnowledgeBase:
    def __init__(self, cache_dir: str, model_name: str):
        self.serializer = Serializer(cache_dir)
        self.model_name = model_name
        self.corpus: List[str] = []
        self.file_map: Dict[int, str] = {}
        self.embeddings: Optional[torch.Tensor] = None
        self.ingested_file_names: List[str] = []
        self.is_indexed = False

    def load(self) -> bool:
        try:
            logger.log("Loading KnowledgeBase from cache...", "INFO")
            embeddings, metadata = self.serializer.load_embeddings(
                self.model_name
                )
            if embeddings is None:
                return False
            corpus = self.serializer.load_corpus()
            file_map = self.serializer.load_file_map()
            if corpus is None or file_map is None:
                return False
            self.embeddings = embeddings
            self.corpus = corpus
            self.file_map = file_map
            self.ingested_file_names = metadata.get("file_names", [])
            self.is_indexed = True
            logger.log(f"Loaded {len(corpus)} docs from cache.", "INFO")
            return True
        except Exception as e:
            logger.log(f"Error loading cache: {e}", "ERROR")
            return False

    def save(self):
        try:
            logger.log("Saving KnowledgeBase to cache...", "INFO")
            self.serializer.save_embeddings(
                self.embeddings,
                self.corpus,
                self.model_name,
                {"file_names": self.ingested_file_names}
            )
            self.serializer.save_corpus(self.corpus)
            self.serializer.save_file_map(self.file_map)
        except Exception as e:
            logger.log(f"Failed to save cache: {e}", "WARNING")

    def update_data(
            self,
            new_chunks: List[str],
            new_file_map: Dict[int, str],
            new_embeddings: torch.Tensor
            ):
        self.corpus.extend(new_chunks)
        self.file_map.update(new_file_map)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        elif new_embeddings is not None:
            if self.embeddings.device != new_embeddings.device:
                self.embeddings = self.embeddings.to(new_embeddings.device)
            self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)

        self.is_indexed = True

    def get_document_chunks(self, filename: str) -> List[str]:
        try:
            # Normalize filename for comparison
            filename_lower = filename.lower().strip()

            # Find all chunk indices that belong to this file
            matching_indices = []
            for idx, mapped_filename in self.file_map.items():
                mapped_lower = mapped_filename.lower().strip()
                # Match exact or suffix (basename match)
                if (mapped_lower == filename_lower
                        or mapped_lower.endswith(filename_lower)
                        or filename_lower.endswith(mapped_lower)):
                    matching_indices.append(idx)

            if not matching_indices:
                logger.log(
                    f"No chunks found for file: {filename}", "WARNING"
                )
                return []

            # Sort indices to maintain chunk order
            matching_indices.sort()

            # Retrieve chunks in order
            chunks = []
            for idx in matching_indices:
                if idx < len(self.corpus):
                    chunks.append(self.corpus[idx])
                else:
                    logger.log(
                        f"Index {idx} out of range for corpus", "WARNING"
                    )

            logger.log(
                f"Retrieved {len(chunks)} chunks for file: {filename}",
                "INFO"
            )
            return chunks

        except Exception as e:
            logger.log(
                f"Error retrieving chunks for {filename}: {e}", "ERROR"
            )
            return []
