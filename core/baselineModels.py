import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from core.featureExtractors import TfidfFeatures

# Wang and Manning, 2012
class NbSvmClassifier:
    def __init__(self, C=0.3, alpha=1.0, classWeight='balanced', numJobs=2, randomSeed=42):
        self.C = C
        self.alpha = alpha
        self.classWeight = classWeight
        self.numJobs = numJobs
        self.randomSeed = randomSeed

    def _logRatio(self, features, labels, label):
        binary = (features > 0).astype(np.float32)
        positive = self.alpha + np.asarray(binary[labels == label].sum(axis=0)).ravel()
        negative = self.alpha + np.asarray(binary[labels != label].sum(axis=0)).ravel()

        return sp.diags(np.log((positive / positive.sum()) / (negative / negative.sum())).astype(np.float32))

    def _fitOne(self, features, labels, label):
        ratio = self._logRatio(features, labels, label)
        svm = LinearSVC(C=self.C, class_weight=self.classWeight, random_state=self.randomSeed)
        svm.fit(features @ ratio, (labels == label).astype(int))

        return ratio, svm

    def fit(self, features, labels):
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        self.estimators = Parallel(n_jobs=self.numJobs, prefer='threads')(delayed(self._fitOne)(features, labels, label) for label in self.classes_)

        return self

    def decision_function(self, features):
        return np.column_stack([svm.decision_function(features @ ratio) for ratio, svm in self.estimators])

    def predict(self, features):
        return self.classes_[self.decision_function(features).argmax(axis=1)]

# TF-IDF + linear classifier 
class TfidfClassifier:
    def __init__(self, featureBlocks=None,
                selectK=None,
                classifier='svm',
                C=0.5,
                classWeight='balanced',
                randomSeed=42):

        self.features = TfidfFeatures(featureBlocks, selectK)

        if classifier == 'svm':
            self.classifier = LinearSVC(C=C, class_weight=classWeight, random_state=randomSeed)
        elif classifier == 'nbsvm':
            self.classifier = NbSvmClassifier(C=C, classWeight=classWeight, randomSeed=randomSeed)
        elif classifier == 'logreg':
            self.classifier = LogisticRegression(C=C, class_weight=classWeight, max_iter=2000, random_state=randomSeed)
        else:
            raise ValueError(f"Unknown classifier '{classifier}'")

    def fit(self, corpus, labels, validationData=None):
        labels = np.asarray(labels)
        self.classifier.fit(self.features.fitTransform(corpus, labels), labels)
        return self

    def predict(self, corpus):
        return self.classifier.predict(self.features.transform(corpus))

    def decisionScores(self, corpus):
        return self.classifier.decision_function(self.features.transform(corpus))

    @property
    def classes(self):
        return self.classifier.classes_
