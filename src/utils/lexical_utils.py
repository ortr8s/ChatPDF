from typing import List
import spacy

from src.utils.logger import Logger

logger = Logger(__name__)


class Lemmatizer:
    def __init__(self):
        try:
            logger.log("Loading spaCy English model", "DEBUG")
            self.nlp = spacy.load("en_core_web_sm")
            logger.log("spaCy model loaded: en_core_web_sm", "INFO")
        except OSError:
            logger.log(
                "spaCy model not found. Run: "
                "python -m spacy download en_core_web_sm", "ERROR"
            )
            raise

    def lemmatize(self, corpus: List[str]) -> List[List[str]]:
        logger.log(f"Lemmatizing {len(corpus)} documents", "DEBUG")
        tokenized_corpus = []
        # disable=['ner', 'parser'] speeds up lemmatization
        try:
            for doc in self.nlp.pipe(
                corpus,
                disable=["ner", "parser"],
                batch_size=50
            ):
                tokens = [
                    token.lemma_.lower()
                    for token in doc
                    if (
                        not token.is_stop
                        and not token.is_punct
                        and not token.is_space
                    )
                ]
                tokenized_corpus.append(tokens)

            logger.log(
                f"Lemmatized {len(tokenized_corpus)} documents", "DEBUG"
            )
            return tokenized_corpus
        except Exception as e:
            logger.log(f"Error during lemmatization: {e}", "ERROR")
            raise

    def tokenize_query(self, text: str) -> List[str]:
        try:
            doc = self.nlp(text)
            tokens = [
                token.lemma_.lower()
                for token in doc
                if (
                    not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                )
            ]
            return tokens
        except Exception as e:
            logger.log(f"Error tokenizing query: {e}", "ERROR")
            raise
