#!/usr/bin/env python
# coding: utf-8

# In[1]:


from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


# In[ ]:


import os, sys

module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[ ]:


import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import annoy
from gensim.models import Word2Vec, FastText
import pickle
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd


# In[ ]:


from preprocess_func import morpher, sw, exclude, preprocess_txt, embed_txt


# In[ ]:


FTmodel = FastText.load("../data/bot_trained/ft_model")


# In[ ]:


FTmodel_movies = FastText.load("../data/bot_trained/ft_model_imdb")


# In[ ]:


with open('../data/bot_trained/lr_vect_tfidf_anecd.pkl', 'rb') as ctl:
    countvect_anecd, tfidf_vect_anecd, lr_anecd = pickle.load(ctl)


# In[ ]:


idfs_anecd = {v[0]: v[1] for v in zip(tfidf_vect_anecd.vocabulary_, tfidf_vect_anecd.idf_)}
midf_anecd = np.mean(tfidf_vect_anecd.idf_)


# #load logostic regression model for anecdote classification
# with open("../data/bot_trained/lr_model_anecd.sav", "rb") as mod:
#     anecd_lr_model = pickle.load(mod)
# #anecd_lr_model = pickle.load(open('../data/bot_trained/lr_model_anecd.sav', 'rb'))

# In[ ]:


with open('../data/bot_trained/lr_vect_tfidf_imdb.pkl', 'rb') as idb:
    countvect_imdb, tfidf_vect_imdb, lr_imdb = pickle.load(idb)


# In[ ]:


idfs_imdb = {v[0]: v[1] for v in zip(tfidf_vect_imdb.vocabulary_, tfidf_vect_imdb.idf_)}
midf_imdb = np.mean(tfidf_vect_imdb.idf_)


# #load logostic regression model for imdb classification
# #imdb_lr_model = pickle.load(open('../data/bot_trained/lr_model_imdb.sav', 'rb'))
# with open("../data/bot_trained/lr_model_imdb.sav", "rb") as im:
#     imdb_lr_model = pickle.load(im)

# *loading fasttext annoy trees with indexes and vectors*

# In[ ]:


ft_index = annoy.AnnoyIndex(100 ,'angular')
ft_index.load('../data/bot_trained/speaker.ann')


# In[ ]:


ft_index_anecd = annoy.AnnoyIndex(100 ,'angular')
ft_index_anecd.load('../data/bot_trained/anecd.ann')


# In[ ]:


ft_index_imdb = annoy.AnnoyIndex(100 ,'angular')
ft_index_imdb.load('../data/bot_trained/imdb.ann')


# *loading dicts with indexes and replies*

# In[ ]:


with open("../data/bot_trained/index_speaker.pkl", "rb") as f:
    index_loaded = pickle.load(f)


# In[ ]:


with open("../data/bot_trained/index_anecd.pkl", "rb") as p:
    index_anecd_loaded = pickle.load(p)


# In[ ]:


with open("../data/bot_trained/index_imdb.pkl", "rb") as db:
    index_imdb_loaded = pickle.load(db)


# In[2]:


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def startCommand(update: Update, context: CallbackContext):
    update.message.reply_text('Hi!')


# def startCommand2(bot, update):
#     bot.send_message(chat_id=update.message.chat_id, text='Добрый день')
# 
# 

# In[ ]:


def textMessage(update, context):
    
    input_txt = preprocess_txt(update.message.text)
    vect = countvect_imdb.transform([" ".join(input_txt)])
    prediction_imdb = lr_imdb.predict(vect)
    
    if prediction_imdb[0] == 1:
        vect_ft = embed_txt(input_txt, idfs_imdb, midf_imdb, FTmodel_movies.wv)
        ft_index_imdb_val = ft_index_imdb.get_nns_by_vector(vect_ft, 3) 
        for item in ft_index_imdb_val:
            title, link = index_imdb_loaded[item]
            context.bot.send_message(chat_id=update.message.chat_id, text="title: {} link: {}".format(title, link))
        return
    
    else:
        vect_anecd = countvect_anecd.transform([" ".join(input_txt)])
        prediction_anecd = lr_anecd.predict(vect_anecd)
        if prediction_anecd[0] == 1:
            vect_ft = embed_txt(input_txt, idfs_anecd, midf_anecd, FTmodel.wv)
            ft_index_anecd_val = ft_index_anecd.get_nns_by_vector(vect_ft, 3) 
            for item in ft_index_anecd_val:
                text = index_anecd_loaded[item]
                context.bot.send_message(chat_id=update.message.chat_id, text=text)
            return
             
        else:
            vect_ft = embed_txt(input_txt, {}, 1, FTmodel.wv)
            ft_index_val, distances = ft_index.get_nns_by_vector(vect_ft, 1, include_distances=True)
            if distances[0] > 0.2:
                print(distances[0])
                context.bot.send_message(chat_id=update.message.chat_id, text="Моя твоя не понимать, спробуй ще")
                return
            context.bot.send_message(chat_id=update.message.chat_id, text=index_loaded[ft_index_val[0]])
        


# In[ ]:


bot_token = "5872854434:AAEY_bCIlhxY6LcLfx2cIiyEXWgSrW_eNIk"


# In[ ]:


updater = Updater(bot_token, use_context=True)
dispatcher = updater.dispatcher


# In[ ]:


start_command_handler = CommandHandler('start', startCommand)
text_message_handler = MessageHandler(Filters.text & ~Filters.command, textMessage)
dispatcher.add_handler(start_command_handler)
dispatcher.add_handler(text_message_handler)
updater.start_polling(clean=True)
updater.idle()

