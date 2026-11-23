import re
import emoji
import pandas as pd
import spacy

ENTITY_LABELS = {'PER', 'ORG', 'LOC'}
NEGATION_WORDS = {'no', 'ni', 'nunca', 'jamás', 'tampoco', 'nada', 'nadie'}

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\b[\w.+-]+@([\w-]+)\.[\w.-]+\b')
SOCIAL_TAG_PATTERN = re.compile(r'[@#]\w+')
REPEATED_CHAR_PATTERN = re.compile(r'([^\W\d_])\1{2,}')
PLACEHOLDER_PATTERN = re.compile(r'^_[A-Z]+_$')
EMOJI_TAG_PATTERN = re.compile(r':[^\s:]+:')
PROTECTED_TOKEN_PATTERN = re.compile(r'^(?:[@#]\w+|_[A-Z]+_|:[^\s:]+:)$')

def applyRegex(text):
    text = URL_PATTERN.sub(' _URL_ ', text)
    text = EMAIL_PATTERN.sub(lambda match: f' _EMAIL_ {match.group(1)} ', text)
    text = SOCIAL_TAG_PATTERN.sub(lambda match: f' {match.group(0)} ', text)
    text = REPEATED_CHAR_PATTERN.sub(r'\1\1', text)
    text = emoji.replace_emoji(text, replace=lambda char, data: f" {emoji.demojize(char, language='es')} ")

    return ' '.join(text.split())

def lightNormalize(text):
    return applyRegex(text).lower()

def buildViews(texts, lemmaTokens):
    return pd.DataFrame({'raw': [lightNormalize(text) for text in texts], 'lemmas': list(lemmaTokens)}, index=lemmaTokens.index)

class TextNormalizer:
    def __init__(self, removeStopwords=True, keepNegation=True, keepEmojis=True, mergeEntities=True, spacyModel='es_core_news_sm'):
        self.removeStopwords = removeStopwords
        self.keepNegation = keepNegation
        self.keepEmojis = keepEmojis
        self.mergeEntities = mergeEntities
        self.nlp = spacy.load(spacyModel)
        self.nlp.tokenizer.token_match = PROTECTED_TOKEN_PATTERN.match

    def _isMergeable(self, entity):
        if entity.label_ not in ENTITY_LABELS or len(entity) < 2:
            return False
        
        return all(token.is_alpha for token in entity)

    def _keepToken(self, token):
        if PROTECTED_TOKEN_PATTERN.match(token.text):
            return True
        
        if self.keepEmojis and EMOJI_TAG_PATTERN.fullmatch(token.text):
            return True
        
        if token.ent_type_ in ENTITY_LABELS:
            return True
        
        if token.is_punct or token.is_space:
            return False
        
        if token.is_stop and self.removeStopwords:
            return self.keepNegation and token.lower_ in NEGATION_WORDS
        
        return token.is_alpha

    def _tokenToString(self, token):
        if PLACEHOLDER_PATTERN.match(token.text):
            return token.text
        
        if SOCIAL_TAG_PATTERN.fullmatch(token.text) or EMOJI_TAG_PATTERN.fullmatch(token.text) or token.ent_type_ in ENTITY_LABELS:
            return token.text.lower().replace(' ', '_')
        
        return token.lemma_.lower().replace(' ', '_')

    def _processDoc(self, doc):
        if self.mergeEntities:
            with doc.retokenize() as retokenizer:
                for entity in doc.ents:
                    if self._isMergeable(entity):
                        retokenizer.merge(entity)
        
        return [self._tokenToString(token) for token in doc if self._keepToken(token)]

    def normalize(self, text):
        return self._processDoc(self.nlp(applyRegex(text)))

    def normalizeCorpus(self, texts, batchSize=1000, numProcesses=1):
        cleanTexts = (applyRegex(text) for text in texts)
        docs = self.nlp.pipe(cleanTexts, batch_size=batchSize, n_process=numProcesses)
        
        return [self._processDoc(doc) for doc in docs]
