from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger', 'maxent_ne_chunker'])
from nltk.corpus import stopwords
import re


def tokenize(text) -> [str]:
    
    # Remove non word characters
    text = re.sub(r'[^\w]', ' ', text)
    
    # Create tokens from words
    tokens = word_tokenize(text)
    
    # Lemmatize, normalize case, and remove leading/trailing white space
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    final_tokens = [lemmatizer.lemmatize(token).strip().lower() 
                        for token in tokens 
                            if token not in stop_words and len(token) > 2]
    return final_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) == 0:
                return False
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

class ResponseLengthExtractor(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self
    
    def response_length(self, text):
        return len(text)
    
    def transform(self, X):
        X_length = pd.Series(X).apply(self.response_length)
        return pd.DataFrame(X_length)
