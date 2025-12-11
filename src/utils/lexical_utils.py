from typing import List
import spacy


class Lemmatizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def lemmatize(self, corpus: List[str]) -> List[List[str]]:
        tokenized_corpus = []
        # disable=['ner', 'parser'] speeds up lematization
        for doc in self.nlp.pipe(corpus, disable=["ner", "parser"], batch_size=50):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            ]
            tokenized_corpus.append(tokens)

        return tokenized_corpus

    def tokenize_query(self, text: str) -> List[str]:
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        return tokens
