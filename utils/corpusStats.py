from collections import Counter
import emoji
import pandas as pd
from utils.dataLoader import TEXT_COLUMN, TARGET_COLUMN

def classDistribution(df):
    counts = df[TARGET_COLUMN].value_counts()
    return pd.DataFrame({'count': counts, 'share': (counts / counts.sum()).round(4)})

def lengthStats(df):
    lengths = df[TEXT_COLUMN].str.split().str.len()
    return lengths.groupby(df[TARGET_COLUMN]).describe().round(1)

def noiseStats(df):
    text = df[TEXT_COLUMN]
    
    features = pd.DataFrame({
        'url': text.str.contains(r'https?://', regex=True),
        'mention': text.str.contains(r'@\w+', regex=True),
        'hashtag': text.str.contains(r'#\w+', regex=True),
        'emoji': text.map(lambda tweet: emoji.emoji_count(tweet) > 0),
    })

    summary = features.groupby(df[TARGET_COLUMN]).mean()
    summary.loc['all'] = features.mean()
    return summary.round(3)

def topTokensByClass(df, pattern, topN=10):
    tokens = df[TEXT_COLUMN].str.lower().str.findall(pattern)
    topTokens = {}
    
    for label, group in tokens.groupby(df[TARGET_COLUMN]):
        counter = Counter(token for tweetTokens in group for token in tweetTokens)
        topTokens[label] = [token for token, _ in counter.most_common(topN)]
    
    return pd.DataFrame(topTokens)
