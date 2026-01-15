import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self._load_config()

    @classmethod
    def _load_config(cls) -> None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if not config_path.exists():
            logger.warning(
                f"Config file not found at {config_path}. "
                "Using defaults."
            )
            cls._config = cls._get_defaults()
            return
        try:
            with open(config_path, "r") as f:
                cls._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            cls._config = cls._get_defaults()

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        return {
            "models": {
                "semantic_retriever": "sentence-transformers/all-MiniLM-L6-v2",
                "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                "generator": "microsoft/Phi-3.5-mini-instruct",
            },
            "retrieval": {
                "chunk_size": 512,
                "chunk_overlap": 64,
                "batch_size": 32,
                "top_k_lexical": 5,
                "top_k_semantic": 5,
                "rerank_top_k": 3,
            },
            "reranker": {
                "device": "cuda",
            },
            "biencoder": {
                "device": "cuda",
            },
            "llm": {
                "system_prompt": (
                    "You are a knowledgeable and precise assistant. "
                    "Your goal is to answer the user's question using "
                    "ONLY the provided context documents.\n\n"
                    "Guidelines:\n"
                    "1. **Reliance on Context:** Answer the query "
                    "strictly based on the \"Numbered document list\" "
                    "provided in the user message. Do not use outside "
                    "knowledge or make up facts.\n"
                    "2. **Citations:** When you answer, cite the 'Source' "
                    "or 'Document ID' of the information "
                    "(e.g., \"According to the HR Handbook...\").\n"
                    "3. **Uncertainty:** If the provided documents do not "
                    "contain the answer, simply state: \"I cannot answer "
                    "this question based on the provided documents.\" "
                    "Do not attempt to guess.\n"
                    "4. **Tone:** Keep your answer professional, concise, "
                    "and direct."
                ),
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 0.9,
                "attn_implementation": "eager",
                "device": "cuda",
                "quantize": {
                    "enable": True,
                    "how_many_bits": 4,
                },
            },
            "cache": {
                "directory": ".chatpdf_cache",
                "use_embeddings_cache": True,
                "auto_invalidate": True,
            },
            "logging": {
                "level": "ERROR",
                "log_to_file": False,
                "log_file": "chatpdf.log",
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})

    def __repr__(self) -> str:
        return f"Config({self._config})"


def get_config() -> Config:
    return Config()
