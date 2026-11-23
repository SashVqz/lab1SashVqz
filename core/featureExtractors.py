import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

WORD_PATTERN = re.compile(r'(?u)[@#]?\w+|_[A-Z]+_')

DEFAULT_BLOCKS = [
    {'view': 'raw', 'analyzer': 'word', 'ngramRange': (1, 2), 'minDf': 2},
    {'view': 'raw', 'analyzer': 'char', 'ngramRange': (2, 5), 'minDf': 3, 'maxFeatures': 500000},
    {'view': 'lemmas', 'analyzer': 'word', 'ngramRange': (1, 2), 'minDf': 2},
]

def _toTokens(item):
    return item if isinstance(item, list) else WORD_PATTERN.findall(item)

def _toText(item):
    return ' '.join(item) if isinstance(item, list) else item

def _wordNgrams(tokens, ngramRange):
    minN, maxN = ngramRange

    return [' '.join(tokens[i:i + n]) for n in range(minN, maxN + 1) for i in range(len(tokens) - n + 1)]

# TF-IDF over one or more views of the text (raw, lemmas)
class TfidfFeatures:
    def __init__(self, blocks=None, selectK=None):
        self.blocks = blocks if blocks is not None else DEFAULT_BLOCKS
        self.vectorizers = [self._buildVectorizer(block) for block in self.blocks]
        self.selector = SelectKBest(chi2, k=selectK) if selectK else None

    def _buildVectorizer(self, block):
        common = {'min_df': block.get('minDf', 2), 'max_features': block.get('maxFeatures'), 'sublinear_tf': True}

        if block['analyzer'] == 'char':
            return TfidfVectorizer(analyzer='char_wb', ngram_range=block['ngramRange'], **common)

        return TfidfVectorizer(analyzer=lambda item, ngramRange=block['ngramRange']: _wordNgrams(_toTokens(item), ngramRange), **common)

    def _views(self, corpus):
        return corpus if isinstance(corpus, pd.DataFrame) else pd.DataFrame({'lemmas': corpus})

    def _blockData(self, views, block):
        column = views[block['view']]

        return [_toText(item) for item in column] if block['analyzer'] == 'char' else list(column)

    def blockMatrices(self, corpus, fit=False):
        views = self._views(corpus)
        matrices = []

        for block, vectorizer in zip(self.blocks, self.vectorizers):
            data = self._blockData(views, block)
            matrices.append(vectorizer.fit_transform(data) if fit else vectorizer.transform(data))

        return matrices

    def _stack(self, corpus, fit):
        matrices = self.blockMatrices(corpus, fit)
        weighted = [matrix * block.get('weight', 1.0) for block, matrix in zip(self.blocks, matrices)]

        return sp.hstack(weighted).tocsr().astype(np.float32)

    def fitTransform(self, corpus, labels):
        features = self._stack(corpus, fit=True)
        if self.selector is not None:
            features = self.selector.fit_transform(features, labels)

        return features

    def transform(self, corpus):
        features = self._stack(corpus, fit=False)
        if self.selector is not None:
            features = self.selector.transform(features)

        return features

    @property
    def numFeatures(self):
        if self.selector is not None:
            return int(self.selector.get_support().sum())

        return sum(len(vectorizer.vocabulary_) for vectorizer in self.vectorizers)
