from utils.dataLoader import splitTrainValidationTest
from utils.metrics import calculateMetrics

class ClassificationPipeline:
    def __init__(self, model):
        self.model = model
        self.splits = None

    def _evaluate(self, corpus, labels, idx):
        predictions = self.model.predict(corpus.loc[idx])
        actuals = labels.loc[idx].to_numpy()
        
        return predictions, actuals, calculateMetrics(actuals, predictions)

    def run(self, corpus, labels, validationSize=0.15, testSize=0.15, randomSeed=42):
        self.splits = splitTrainValidationTest(labels, validationSize, testSize, randomSeed)
        trainIdx, validationIdx, _ = self.splits

        self.model.fit(corpus.loc[trainIdx], labels.loc[trainIdx], validationData=(corpus.loc[validationIdx], labels.loc[validationIdx]))
        
        return self._evaluate(corpus, labels, validationIdx)

    def test(self, corpus, labels):
        if self.splits is None:
            raise RuntimeError("error: test(...) /core/pipeline.py")

        return self._evaluate(corpus, labels, self.splits[2])

    def fitFull(self, corpus, labels):
        self.model.fit(corpus, labels)
        
        return self

    def predict(self, corpus):
        return self.model.predict(corpus)
