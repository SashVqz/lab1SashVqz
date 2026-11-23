import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from core.baselineModels import NbSvmClassifier
from core.featureExtractors import TfidfFeatures

class SvmBlockTuner:
    def __init__(self, featureBlocks, trainCorpus, trainLabels, validationCorpus, validationLabels, classifier='svm', classWeight='balanced', randomSeed=42):
        self.featureBlocks = featureBlocks
        self.classifier = classifier
        self.classWeight = classWeight
        self.randomSeed = randomSeed
        features = TfidfFeatures(featureBlocks)
        self.trainBlocks = features.blockMatrices(trainCorpus, fit=True)
        self.validationBlocks = features.blockMatrices(validationCorpus)
        self.trainLabels = np.asarray(trainLabels)
        self.validationLabels = np.asarray(validationLabels)
        self.history = []

    def _weights(self, params):
        return [params.get(f'weight{i}', 1.0) for i in range(len(self.featureBlocks))]

    def _stack(self, matrices, weights):
        return sp.hstack([matrix * weight for matrix, weight in zip(matrices, weights)]).tocsr().astype(np.float32)

    def _buildClassifier(self, C):
        if self.classifier == 'nbsvm':
            return NbSvmClassifier(C=C, classWeight=self.classWeight, numJobs=1, randomSeed=self.randomSeed)

        return LinearSVC(C=C, class_weight=self.classWeight, random_state=self.randomSeed)

    def objective(self, params):
        weights = self._weights(params)
        classifier = self._buildClassifier(10 ** params['logC'])
        classifier.fit(self._stack(self.trainBlocks, weights), self.trainLabels)
        predictions = classifier.predict(self._stack(self.validationBlocks, weights))
        score = f1_score(self.validationLabels, predictions, average='macro')
        self.history.append({**params, 'f1Macro': score})

        return 1 - score

    def bestParams(self, params):
        blocks = [{**block, 'weight': weight} for block, weight in zip(self.featureBlocks, self._weights(params))]

        return {'featureBlocks': blocks, 'classifier': self.classifier, 'C': 10 ** params['logC']}
