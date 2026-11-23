import numpy as np
from sklearn.metrics import f1_score

class MacroF1EarlyStopping:
    def __init__(self, validationFeatures, validationLabels, patience=2, batchSize=1024):
        self.validationFeatures = validationFeatures
        self.validationLabels = validationLabels
        self.patience = patience
        self.batchSize = batchSize

        self.bestF1 = -np.inf
        self.bestEpoch = 0
        self.bestWeights = None
        self.wait = 0

    def update(self, network, epoch):
        probabilities = network.predict(self.validationFeatures, batch_size=self.batchSize, verbose=0)
        validationF1 = float(f1_score(self.validationLabels, probabilities.argmax(axis=1), average='macro'))

        if validationF1 > self.bestF1:
            self.bestF1, self.bestEpoch, self.wait = validationF1, epoch + 1, 0
            self.bestWeights = network.get_weights()
        else:
            self.wait += 1

        return validationF1, self.wait >= self.patience

    def restoreBest(self, network):
        if self.bestWeights is not None:
            network.set_weights(self.bestWeights)
