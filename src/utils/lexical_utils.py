from typing import List
import logging
import spacy

logger = logging.getLogger(__name__)


class Lemmatizer:
    def __init__(self):
        try:
            logger.debug("Loading spaCy English model")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded: en_core_web_sm")
        except OSError:
            logger.error(
                "spaCy model not found. Run: "
                "python -m spacy download en_core_web_sm"
            )
            raise

    def lemmatize(self, corpus: List[str]) -> List[List[str]]:
        logger.debug(f"Lemmatizing {len(corpus)} documents")
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

            logger.debug(f"Lemmatized {len(tokenized_corpus)} documents")
            return tokenized_corpus
        except Exception as e:
            logger.error(f"Error during lemmatization: {e}")
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
            logger.error(f"Error tokenizing query: {e}")
            raise
