from transformers import TextIteratorStreamer
from threading import Thread
from src.utils.logger import Logger
from src.utils.config import get_config
from src.utils.generator_utils import init_pipeline
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
        self.config = get_config()
        self.tokenizer, self.pipeline = init_pipeline(
            self.model_name,
            get_config()
        )

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
