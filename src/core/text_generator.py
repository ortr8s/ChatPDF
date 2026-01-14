import torch
from transformers import pipeline, TextIteratorStreamer, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from threading import Thread
from src.utils.logger import Logger
from src.utils.config import get_config
logger = Logger(__name__)


class Generator:

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = None
        self.pipeline = None
        self.config = get_config()
        self._initialize_pipeline()

    def stream_answer(
        self,
        messages: list,
        max_new_tokens: int = 500
    ):
        if self.pipeline is None:
            raise RuntimeError("Generation pipeline not initialized")
        try:
            logger.log("Starting streaming generation", "DEBUG")
            if isinstance(messages, list) and self.tokenizer.chat_template:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif isinstance(messages, list):
                logger.log("No chat template found, using manual concat", "WARNING")
                prompt_text = "\n".join([m["content"] for m in messages])
            else:
                prompt_text = messages
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            generation_kwargs = {
                "text_inputs": prompt_text,
                "max_new_tokens": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": True,
                "streamer": streamer,
                "return_full_text": False,
                "use_cache": True,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            thread = Thread(
                target=self.pipeline,
                kwargs=generation_kwargs
            )
            thread.daemon = True
            thread.start()
            for token in streamer:
                yield token
            logger.log("Streaming generation complete", "DEBUG")
        except Exception as e:
            logger.log(f"Error during streaming generation: {e}", "ERROR")
            raise

    def _initialize_pipeline(self) -> None:
        try:
            logger.log(f"Loading generation model: {self.model_name}", "DEBUG")
            if torch.cuda.is_available():
                device_name = "cuda"
            logger.log(f"Using device: {device_name}", "DEBUG")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=False
            )
            self.tokenizer = tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {
                "device_map": self._get_device(device_name),
                "trust_remote_code": False,
                "attn_implementation": self.config.get("llm.attn_implementation")
            }
            if device_name == "cuda":
                if self.config.get("llm.quantize.enable"):
                    self._apply_quantization_config(model_kwargs)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer
                )
                logger.log(
                    f"Generation model loaded: {self.model_name} on {device_name}",
                    "INFO"
                )
        except Exception as e:
            logger.log(
                f"Failed to load generation model {self.model_name}: {e}",
                "ERROR"
            )
            raise

    def _get_device(self, available_name: str):
        conf_device = self.config.get("llm.device")
        if available_name == "cuda" and conf_device == "cuda":
            return "cuda"
        else:
            return conf_device

    def _apply_quantization_config(self, model_kwargs):
        how_many_bits = self.config.get("llm.quantize.how_many_bits")
        if how_many_bits == 4:
            logger.log("Applying 4-bit NF4 Quantization (Speed Optimized)", "INFO")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif how_many_bits == 8:
            logger.log("Applying 8-bit Quantization", "INFO")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
