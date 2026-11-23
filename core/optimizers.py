import numpy as np
from joblib import Parallel, delayed

# Grey Wolf Optimizer (GWO) implementation for hyperparameter tuning used in my bachelor thesis. Based on the original paper by Mirjalili et al., 2014.
class GreyWolfOptimizer:
    def __init__(self, objectiveFunction, bounds, numWolves=10, maxIter=50, integerKeys=(), numJobs=1, randomSeed=42):
        self.fitnessFunc = objectiveFunction
        self.bounds = bounds
        self.dim = len(bounds)
        self.numWolves = numWolves
        self.maxIter = maxIter
        self.integerKeys = set(integerKeys)
        self.numJobs = numJobs
        self.rng = np.random.default_rng(randomSeed)
        
        self.lowerBound = np.array([b[0] for b in bounds.values()])
        self.upperBound = np.array([b[1] for b in bounds.values()])
        
        self.positions = self.rng.uniform(0, 1, (numWolves, self.dim)) * (self.upperBound - self.lowerBound) + self.lowerBound
        
        self.alphaPos = np.zeros(self.dim)
        self.alphaScore = float("inf")
        
        self.betaPos = np.zeros(self.dim)
        self.betaScore = float("inf")
        
        self.deltaPos = np.zeros(self.dim)
        self.deltaScore = float("inf")

    def _move(self, leaderPos, position, a):
        A = 2 * a * self.rng.random() - a
        C = 2 * self.rng.random()

        return leaderPos - A * abs(C * leaderPos - position)

    def optimize(self):
        for t in range(self.maxIter):
            self.positions = np.clip(self.positions, self.lowerBound, self.upperBound)
            scores = Parallel(n_jobs=self.numJobs, prefer='threads')(delayed(self.fitnessFunc)(self._arrayToParams(position)) for position in self.positions)

            for i, fitness in enumerate(scores):
                if fitness < self.alphaScore:
                    self.deltaScore = self.betaScore
                    self.deltaPos = self.betaPos.copy()
                    self.betaScore = self.alphaScore
                    self.betaPos = self.alphaPos.copy()
                    self.alphaScore = fitness
                    self.alphaPos = self.positions[i].copy()
                elif fitness < self.betaScore:
                    self.deltaScore = self.betaScore
                    self.deltaPos = self.betaPos.copy()
                    self.betaScore = fitness
                    self.betaPos = self.positions[i].copy()
                elif fitness < self.deltaScore:
                    self.deltaScore = fitness
                    self.deltaPos = self.positions[i].copy()
            
            a = 2 - t * (2 / self.maxIter)
            
            for i in range(self.numWolves):
                for j in range(self.dim):
                    X1 = self._move(self.alphaPos[j], self.positions[i, j], a)
                    X2 = self._move(self.betaPos[j], self.positions[i, j], a)
                    X3 = self._move(self.deltaPos[j], self.positions[i, j], a)
                    self.positions[i, j] = (X1 + X2 + X3) / 3
            
            print(f"Iter {t+1}/{self.maxIter} | best error (Alpha): {self.alphaScore}")
            
        return self._arrayToParams(self.alphaPos), self.alphaScore

    def _arrayToParams(self, positionArray):
        params = {}
        for key, val in zip(self.bounds.keys(), positionArray):
            params[key] = int(round(val)) if key in self.integerKeys else float(val)

        return params
