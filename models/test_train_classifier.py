import pytest
from train_classifier import tokenize

def test_tokenize_sentence():
    assert tokenize('Hi, this is my first tweet') != None