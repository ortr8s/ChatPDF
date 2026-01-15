import torch
from transformers import logging as transformers_logging
from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from typing import Tuple, Any
from src.utils.logger import Logger

logger = Logger(__name__)


def init_pipeline(model_name: str, config) -> Tuple[Any, Any]:
    try:
        logger.log(f"Loading generation model: {model_name}", "DEBUG")
        if torch.cuda.is_available():
            device_name = "cuda"
        logger.log(f"Using device: {device_name}", "DEBUG")
        disable_transformer_logging(config)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "device_map": get_device(config, device_name),
            "trust_remote_code": False,
            "attn_implementation": config.get("llm.attn_implementation")
        }
        if device_name == "cuda":
            quant_config = get_quantization_config(config)
            if quant_config:
                model_kwargs["quantization_config"] = quant_config
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
            generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )
            logger.log(
                f"Generation model loaded: {model_name} on {device_name}",
                "INFO"
            )
        return tokenizer, generation_pipeline
    except Exception as e:
        logger.log(
            f"Failed to load generation model {model_name}: {e}",
            "ERROR"
        )
        raise


def disable_transformer_logging(config):
    current_log_level = config.get("logging.level", "INFO").upper()
    if current_log_level not in ["INFO", "DEBUG"]:
        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    else:
        transformers_logging.enable_progress_bar()
        transformers_logging.set_verbosity_info()


def get_device(config, available_name: str):
    conf_device = config.get("llm.device")
    if available_name == "cuda" and conf_device == "cuda":
        return "cuda"
    else:
        return conf_device


def get_quantization_config(config):
    if not config.get("llm.quantize.enable"):
        return None
    how_many_bits = config.get("llm.quantize.how_many_bits")
    if how_many_bits == 4:
        logger.log("Applying 4-bit NF4 Quantization (Speed Optimized)", "INFO")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif how_many_bits == 8:
        logger.log("Applying 8-bit Quantization", "INFO")
        return BitsAndBytesConfig(
            load_in_8bit=True
        )
    return None
