import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class Serializer:
    CACHE_VERSION = "1.0"

    def __init__(self, cache_dir: str = ".chatpdf_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using cache directory: {self.cache_dir}")

    def save_embeddings(
        self,
        embeddings: Tensor,
        corpus: list,
        model_name: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        if not isinstance(embeddings, Tensor):
            raise ValueError("Embeddings must be torch.Tensor")

        if len(embeddings) != len(corpus):
            raise ValueError(
                "Embeddings and corpus must have same length"
            )

        try:
            # Prepare data
            embeddings_np = (
                embeddings.cpu().detach().numpy()
                if isinstance(embeddings, Tensor)
                else embeddings
            )

            embeddings_file = (
                self.cache_dir / "embeddings.pt"
            )
            metadata_file = (
                self.cache_dir / "embeddings_metadata.json"
            )

            # Save embeddings
            logger.debug(
                f"Saving {len(embeddings)} embeddings "
                f"({embeddings_np.shape[1]} dims)"
            )
            torch.save(embeddings, embeddings_file)

            # Prepare metadata
            meta = {
                "version": self.CACHE_VERSION,
                "embeddings_model_name": model_name,
                "shape": list(embeddings_np.shape),
                "n_docs": len(corpus),
                "timestamp": datetime.now().isoformat(),
                "dtype": str(embeddings_np.dtype),
                **(metadata or {})
            }

            # Save metadata
            with open(metadata_file, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(
                f"Saved {len(corpus)} embeddings to {embeddings_file}"
            )
            return embeddings_file

        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise

    def load_embeddings(
        self,
        model_name: str
    ) -> Tuple[Optional[Tensor], Dict]:
        try:
            embeddings_file = (
                self.cache_dir / "embeddings.pt"
            )
            metadata_file = (
                self.cache_dir / "embeddings_metadata.json"
            )

            # Check files exist
            if not embeddings_file.exists():
                logger.debug("No cached embeddings found")
                return None, {}

            if not metadata_file.exists():
                logger.warning(
                    "Embeddings exist but metadata missing"
                )
                return None, {}

            # Load metadata
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Validate version
            if metadata.get("version") != self.CACHE_VERSION:
                logger.warning(
                    f"Cache version mismatch. "
                    f"Expected {self.CACHE_VERSION}, "
                    f"got {metadata.get('version')}"
                )
                return None, {}

            # Validate model name
            if metadata.get("embeddings_model_name") != model_name:
                logger.warning(
                    f"Model mismatch in cache. "
                    f"Expected {model_name}, "
                    f"got {metadata.get('embeddings_model_name')}. "
                    f"Regenerating..."
                )
                return None, {}

            # Load embeddings
            logger.debug("Loading cached embeddings from cache")
            embeddings = torch.load(embeddings_file)

            logger.info(
                f"Loaded {metadata.get('n_docs')} cached embeddings "
                f"from {embeddings_file.stat().st_size / 1e6:.1f}MB"
            )
            return embeddings, metadata

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None, {}

    def save_corpus(self, corpus: list) -> Path:
        try:
            corpus_file = self.cache_dir / "corpus.json"
            logger.debug(f"Saving {len(corpus)} corpus chunks")

            with open(corpus_file, "w") as f:
                json.dump(corpus, f)

            logger.info(
                f"Saved corpus to {corpus_file}"
            )
            return corpus_file

        except Exception as e:
            logger.error(f"Error saving corpus: {e}")
            raise

    def load_corpus(self) -> Optional[list]:
        """
        Load corpus from cache.

        Returns:
            Corpus list or None if not found
        """
        try:
            corpus_file = self.cache_dir / "corpus.json"

            if not corpus_file.exists():
                logger.debug("No cached corpus found")
                return None

            logger.debug("Loading cached corpus")
            with open(corpus_file, "r") as f:
                corpus = json.load(f)

            logger.info(f"Loaded {len(corpus)} corpus chunks")
            return corpus

        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            return None

    def save_file_map(
        self,
        file_map: Dict[int, str]
    ) -> Path:
        try:
            filemap_file = self.cache_dir / "file_map.json"
            logger.debug(f"Saving {len(file_map)} file mappings")

            with open(filemap_file, "w") as f:
                json.dump(file_map, f)

            return filemap_file

        except Exception as e:
            logger.error(f"Error saving file map: {e}")
            raise

    def load_file_map(self) -> Optional[Dict[int, str]]:
        try:
            filemap_file = self.cache_dir / "file_map.json"

            if not filemap_file.exists():
                logger.debug("No cached file map found")
                return None

            with open(filemap_file, "r") as f:
                file_map = json.load(f)
                # Convert string keys back to ints
                file_map = {int(k): v for k, v in file_map.items()}

            logger.info(f"Loaded {len(file_map)} file mappings")
            return file_map

        except Exception as e:
            logger.error(f"Error loading file map: {e}")
            return None

    def clear_cache(self) -> None:
        try:
            import shutil
            logger.warning(f"Clearing cache at {self.cache_dir}")
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise

    def cache_exists(self) -> bool:
        metadata_file = self.cache_dir / "embeddings_metadata.json"
        return metadata_file.exists()

    def get_cache_info(self) -> Dict:
        try:
            metadata_file = self.cache_dir / "embeddings_metadata.json"

            if not metadata_file.exists():
                return {"cached": False}

            with open(metadata_file, "r") as f:
                meta = json.load(f)

            return {
                "cached": True,
                "size_mb": (
                    sum(
                        f.stat().st_size
                        for f in self.cache_dir.glob("*")
                    ) / 1e6
                ),
                **meta
            }

        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {"cached": False, "error": str(e)}
