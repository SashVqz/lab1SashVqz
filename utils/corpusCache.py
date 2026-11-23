import gzip
import pickle
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

def loadOrNormalize(texts, normalizer, cacheName, cacheDir=CACHE_DIR):
    cachePath = Path(cacheDir) / f'{cacheName}.pkl.gz'

    if cachePath.exists():
        with gzip.open(cachePath, 'rb') as file:
            cached = pickle.load(file)

        return pd.Series(cached['tokens'], index=cached['index'])

    tokens = pd.Series(normalizer.normalizeCorpus(texts), index=getattr(texts, 'index', None))
    cachePath.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(cachePath, 'wb') as file:
        pickle.dump({'index': tokens.index.tolist(), 'tokens': tokens.tolist()}, file)

    return tokens
