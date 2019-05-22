import numpy as np
import pandas as pd
import nltk
from nltk import tokenize
from nltk.stem.snowball import SnowballStemmer  
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter

def tokenizar(sentenca):        
    """Esta função aplica tokenização em uma sentença

    Parameters:
    sentence (string): um texto

    Returns:
    array: um array de palavras

   """
    # Implementa a sua solução aqui
    tokens = tokenize.word_tokenize(sentenca, language='english')
    return tokens


def aplicar_stemming(words):        
    """Esta função aplica radicalização (stemming) de um conjunto de palavras

    Parameters:
    words (array): um array de strings com palavras nas quais será aplicado stemming.    

    Returns:
    array: um array de palavras stemmizadas.

   """
    # Implementa a sua solução aqui
    stemmer = SnowballStemmer('english')        
    words_stem = [ stemmer.stem(word) for word in words ]    
    return words_stem


def remover_stopwords(words):
    """Esta função aplica remoção de stopwords de um conjunto de palavras

    Parameters:
    words (array): um array de strings com palavras

    Returns:
    array: um array de palavras sem as stopwords
   """
    # Implementa a sua solução aqui
    from nltk.corpus import stopwords 
    stopwords = stopwords.words('english')        
    words_new = [word for word in words if word not in stopwords]    
    return words_new


def remover_pontuacao(words):
    """Esta função aplica remoção pontuacao em uma frase

    Parameters:
    words (array): um array de strings com tokens    

    Returns:
    array: um array de palavras tokens sem a pontuação
   """
    # Implemente a sua solução aqui
    from string import punctuation
    words_new = [word for word in words if word not in punctuation]  
    return words_new

def preprocessar_sentenca(sentenca):
    """Esta função preprocessa um conjunto de sentenças

    Parameters:
    sentenca (string): uma sentenca

    Returns:
    array: um array de palavras tokens pré-processados
   """    
    x = sentenca 
    x = tokenizar(x)
    x = remover_pontuacao(x)
    x = remover_stopwords(x)
    x = aplicar_stemming(x)
    
    return x



def get_word_scores(words, dicionario):
    """Esta função extrai os scores de um conjunto de palavras

    Parameters:
    words (array): conjunto de palavras para extrair os scores
    dicionario (disc): dicionários de palavras, onde cada palavra tem um valor de score associado

    Returns:
    array: um array de valores de scores
   """
    keys = dicionario.keys()    
    word_scores = [ dicionario[word] for word in words if word in keys]
    return word_scores

def predict_from_words(words, dicionario):
    """Esta função realiza a predição a partir de um conjunto de palavras

    Parameters:
    words (array): conjunto de palavras
    dicionario (disc): dicionários de palavras, onde cada palavra tem um valor de score associado

    Returns:
    array: um array de valores de scores
   """
    scores = get_word_scores( words, dicionario)
    score = sum(scores)
    return 1 if score > 0 else -1 if score < 0 else 0

def predict_from_sentenca(sentenca, dicionario, preprocessar_fn):
    """Esta função realiza a predição a partir de um conjunto de palavras

    Parameters:
    words (array): conjunto de palavras
    dicionario (disc): dicionários de palavras, onde cada palavra tem um valor de score associado
    preprocessar_fn (function): função de preprocesamento de sentencas
    Returns:
    array: um array de valores de scores
   """
    words = preprocessar_fn(sentenca)
    return predict_from_words(words, dicionario)