#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import numpy as np


# In[2]:


morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)


# In[3]:


def preprocess_txt(line):
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    return spls


# In[4]:


def embed_txt(txt, idfs, midf, model):
    n_ft = 0
    vector_ft = np.zeros(100)
    for word in txt:
        if word in model:
            vector_ft += model[word] * idfs.get(word, midf)
            n_ft += idfs.get(word, midf)
    return vector_ft / n_ft


# In[ ]:




