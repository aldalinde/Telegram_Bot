#!/usr/bin/env python
# coding: utf-8

# In[1]:


from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


# In[2]:


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, context: CallbackContext):
    update.message.reply_text('Hi!')

def echo(update: Update, context: CallbackContext):
    txt = update.message.text
    
    update.message.reply_text('Ваше сообщение! ' + update.message.text)


# In[ ]:


bot_token = "5872854434:AAEY_bCIlhxY6LcLfx2cIiyEXWgSrW_eNIk"


# In[3]:


updater = Updater(bot_token, use_context=True)
dispatcher = updater.dispatcher

# on different commands - answer in Telegram
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

# Start the Bot
updater.start_polling()
updater.idle()

