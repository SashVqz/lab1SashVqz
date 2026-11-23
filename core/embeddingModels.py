import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

class Word2VecClassifier:
    def __init__(self, view='lemmas',
                vectorSize=200,
                window=5,
                minCount=2,
                epochs=10,
                C=1.0,
                classWeight='balanced',
                randomSeed=42):

        self.view = view
        self.vectorSize = vectorSize
        self.window = window
        self.minCount = minCount
        self.epochs = epochs
        self.randomSeed = randomSeed
        self.classifier = LogisticRegression(C=C, class_weight=classWeight, max_iter=2000, random_state=randomSeed)

    def _tokens(self, corpus):
        return list(corpus[self.view]) if isinstance(corpus, pd.DataFrame) else list(corpus)

    def _embed(self, corpus):
        vectors = self.embeddings.wv
        features = np.zeros((len(corpus), self.vectorSize), dtype=np.float32)

        for i, tokens in enumerate(self._tokens(corpus)):
            known = [token for token in tokens if token in vectors.key_to_index]
            if known:
                features[i] = vectors[known].mean(axis=0)

        return features

    def fit(self, corpus, labels, validationData=None):
        self.embeddings = Word2Vec(
            sentences=self._tokens(corpus),
            vector_size=self.vectorSize,
            window=self.window,
            min_count=self.minCount,
            epochs=self.epochs,
            workers=1,
            seed=self.randomSeed,
        )
        self.classifier.fit(self._embed(corpus), np.asarray(labels))

        return self

    def predict(self, corpus):
        return self.classifier.predict(self._embed(corpus))

    def mostSimilar(self, word, topn=10):
        return self.embeddings.wv.most_similar(word, topn=topn)

    @property
    def classes(self):
        return self.classifier.classes_
