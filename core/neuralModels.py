import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
from core.featureExtractors import TfidfFeatures
from core.callbacks import MacroF1EarlyStopping

# Galke and Scherp, 2022
class WideMlpClassifier:
    def __init__(self, featureBlocks=None,
                selectK=50000,
                hiddenUnits=1024,
                inputDropout=0.5,
                hiddenDropout=0.5,
                learningRate=1e-3,
                weightDecay=1e-4,
                labelSmoothing=0.0,
                batchSize=256,
                maxEpochs=20,
                patience=2,
                randomSeed=42):

        self.features = TfidfFeatures(featureBlocks, selectK)
        self.encoder = LabelEncoder()
        self.hiddenUnits = hiddenUnits
        self.inputDropout = inputDropout
        self.hiddenDropout = hiddenDropout
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.labelSmoothing = labelSmoothing
        self.batchSize = batchSize
        self.maxEpochs = maxEpochs
        self.patience = patience
        self.randomSeed = randomSeed

        self.network = None
        self.history = None
        self.bestEpoch = None

    def _buildNetwork(self, numFeatures, numClasses):
        keras.utils.set_random_seed(self.randomSeed)

        inputs = keras.Input(shape=(numFeatures,), sparse=True)
        x = keras.layers.Dense(self.hiddenUnits, activation='relu')(inputs)
        x = keras.layers.Dropout(self.hiddenDropout)(x)
        outputs = keras.layers.Dense(numClasses, activation='softmax')(x)

        network = keras.Model(inputs, outputs)
        network.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=self.learningRate, weight_decay=self.weightDecay),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=self.labelSmoothing),
            metrics=['accuracy'],
        )

        return network

    def _dropInputs(self, features, rng):
        if self.inputDropout <= 0:
            return features
        
        features = features.copy()
        keep = rng.random(features.nnz) >= self.inputDropout
        features.data = features.data * keep / (1 - self.inputDropout)
        features.eliminate_zeros()
        
        return features

    def fit(self, corpus, labels, validationData=None):
        labels = self.encoder.fit_transform(np.asarray(labels))
        features = self.features.fitTransform(corpus, labels)
        numClasses = len(self.encoder.classes_)
        self.network = self._buildNetwork(features.shape[1], numClasses)

        rng = np.random.default_rng(self.randomSeed)
        targets = keras.utils.to_categorical(labels, numClasses)
        earlyStopping, fitKwargs = None, {}

        if validationData is not None:
            validationFeatures = self.features.transform(validationData[0])
            validationLabels = self.encoder.transform(np.asarray(validationData[1]))
            earlyStopping = MacroF1EarlyStopping(validationFeatures, validationLabels, self.patience)
            fitKwargs['validation_data'] = (validationFeatures, keras.utils.to_categorical(validationLabels, numClasses))
            epochs = self.maxEpochs
        else:
            epochs = self.bestEpoch or self.maxEpochs

        self.history = {}
        for epoch in range(epochs):
            epochHistory = self.network.fit(self._dropInputs(features, rng), targets, batch_size=self.batchSize, verbose=0, **fitKwargs)
            for key, values in epochHistory.history.items():
                self.history.setdefault(key, []).extend(values)

            if earlyStopping is not None:
                validationF1, stop = earlyStopping.update(self.network, epoch)
                self.history.setdefault('val_f1', []).append(validationF1)
                if stop:
                    break

        if earlyStopping is not None:
            earlyStopping.restoreBest(self.network)
            self.bestEpoch = earlyStopping.bestEpoch
        
        return self

    def predictProba(self, corpus):
        return self.network.predict(self.features.transform(corpus), batch_size=1024, verbose=0)

    def predict(self, corpus):
        return self.encoder.inverse_transform(self.predictProba(corpus).argmax(axis=1))

    @property
    def classes(self):
        return self.encoder.classes_
