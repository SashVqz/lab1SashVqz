from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_COLUMN = 'tweet'
TARGET_COLUMN = 'target'
ID_COLUMN = 'id'

def loadTrain(dataDir=DATA_DIR):
    return pd.read_csv(Path(dataDir) / 'train.csv')

def loadTest(dataDir=DATA_DIR):
    return pd.read_csv(Path(dataDir) / 'test.csv')

def splitTrainValidationTest(labels, validationSize=0.15, testSize=0.15, randomSeed=42):
    trainIdx, holdoutIdx = train_test_split(labels.index, test_size=validationSize + testSize, random_state=randomSeed, stratify=labels)
    validationIdx, testIdx = train_test_split(holdoutIdx, test_size=testSize / (validationSize + testSize), random_state=randomSeed, stratify=labels.loc[holdoutIdx])
    
    return trainIdx, validationIdx, testIdx

def saveSubmission(ids, predictions, path):
    submission = pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: predictions})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return submission
