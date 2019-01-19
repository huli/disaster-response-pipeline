import pytest
from feature_extractor import StartingVerbExtractor
from feature_extractor import tokenize

def test_starting_verb_extractor():
    assert all(StartingVerbExtractor().transform(['obtain a new value'])[0]) == True

def test_tokenize_sentence():
    assert tokenize('Hi, this is my first tweet') == ['first', 'tweet']

def test_tokenize_remove_special_chars():
    assert tokenize('This is absolute %&ç*%"*%ç') == ['this', 'absolute']

def test_tokenize_remove_stopwords():
    assert tokenize('those am is') == []