import torch
from transformers import pipeline, TextIteratorStreamer, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from threading import Thread
from src.utils.logger import Logger

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

        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        try:
            logger.log(
                f"Loading generation model: {self.model_name}",
                "debug")

            device_name = "cuda" if torch.cuda.is_available() else "cpu"

            logger.log(
                f"Using device: {device_name}",
                "debug")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.tokenizer = tokenizer

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if device_name == "cuda":
                logger.log(
                    "Using 8-bit quantization for GPU",
                    "debug")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=200.0,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    attn_implementation="eager"
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer
                )

            else:
                logger.log(
                    "CUDA not found. Falling back to CPU (slower).",
                    "warning")

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
                self.model = model
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=True,
                )

            logger.log(
                f"Generation model loaded: {self.model_name} "
                f"on {device_name}",
                "info"
            )
        except Exception as e:
            logger.log(
                f"Failed to load generation model {self.model_name}: {e}",
                "error"
            )
            raise

    def generate_streaming(
        self,
        messages: list,
        max_new_tokens: int = 500
    ):
        if self.pipeline is None:
            raise RuntimeError("Generation pipeline not initialized")

        try:
            logger.log("Starting streaming generation", "debug")

            streamer = TextIteratorStreamer(
                self.tokenizer, # changed from pipeline.tokenizer
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": True,
                "streamer": streamer,
                "return_full_text": False,
                "use_cache": False,
            }

            thread = Thread(
                target=self.pipeline,
                args=(messages,),
                kwargs=generation_kwargs
            )
            thread.start()

            for token in streamer:
                yield token

            thread.join()
            logger.log("Streaming generation complete", "debug")

        except Exception as e:
            logger.log(f"Error during streaming generation: {e}", "error")
            raise

    def __repr__(self) -> str:
        return (
            f"Generator(model={self.model_name}, "
            f"temperature={self.temperature})"
        )
