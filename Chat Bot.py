#!/usr/bin/env python
# coding: utf-8

# In[35]:


# import libraries
import json
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from textblob import TextBlob
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from textblob import TextBlob
import os
from autocorrect import Speller


# In[36]:


# load data
data = json.loads(open('data.json').read())


# In[37]:


intents = data['interactionModel']['languageModel']


# In[38]:


# create set of words and labels
words = []
labels = []
documents = []
punc = ['?', '!', ',', '.']

for intent in intents['intent']: 
    for sample in intent['samples']:
        word_list = nltk.word_tokenize(sample)
        words.extend(word_list)
        documents.append((word_list, intent['name']))
        if intent['name'] not in labels:
            labels.append(intent['name'])


# In[39]:


# lemmatize all words
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()
words = [lem.lemmatize(word.lower()) for word in words if word not in punc]

# sort words and remove duplicates
words = sorted(set(words))
labels = sorted(set(labels))

# Create files storing words and labels for personal reference
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('labels.pkl', 'wb'))


# In[40]:


# load best model
best_model = load_model('best_model.h5')


# In[41]:


print('Model summary: \n')
print(best_model.summary())


# In[42]:


# preprocess input
def preprocess_input(text):
    text_token = nltk.word_tokenize(text)
    text_token = [lem.lemmatize(word.lower()) for word in text_token]
    return text_token

# create bag of words
def bow(text_token):
    bag = []
    for word in words:
        if word in text_token:
            bag.append(1)
        else:
            bag.append(0)
    return bag

# return the predicted label 
def predict_label(bag):
    prediction = best_model.predict(np.array(bag).reshape((1, len(words))))
    max_value = np.max(prediction)
    if max_value > 0.7:
        predict_label = labels[np.argmax(prediction)]
    else:
        predict_label = 'NA'
    return predict_label


# In[43]:


# response dictionary
responses = { 
        'cancel' : {
            'en':"Ok, it's cancelled. Once you're ready, please type *`daily`* to fill out your follow-up reports.",

            'es': "Ok, has cancelado tu seguimiento diario. Por favor escribe *`daily`* para empezar cuando desees."
        },
        'chicken' : {
            'en': "Here comes Arthur!\n\n*[Launch Rubber Chicken]({{{url}}})* 🐔\n\n_Pro tip: you can change the last part of Arthur's URL in your browser in order to create a unique room.",

            'es': "¡Aquí viene Arthur!\n\n[Abrir Rubber Chicken]({{{url}}}) 🐔\n\n_Pro tip: puedes cambiar la última parte de la URL de Arthur en tu navegador y así tendrás una sala única._"

        },
        'daily' : {
            'en':  "No response",
            'es':  "No hay respuesta"
        },
        'dailystatus' : {
            'en': "No response",
            'es': "No hay respuesta"
        },
        'dashboard' : {
            'en': "Ok, please *[follow this link]({{{url}}})* to access the web dashboard.",

            'es': "Ok, *[abre este enlace]({{{url}}})* para abrir la web y dashboard."

        },
        'datetime' : {
            'en':  "The time information for your user is:",
            'es': "La información horaria de tu usuario es:"

        },
        'disable' : {
            'en': "Ok, it's done. You have disconnected and disabled all private messaging between you and me.\nYou can still use the [web application]({{{url}}}).\n\nIf you want to talk to me again, just type *`enable`*",

            'es': "Listo, está hecho. Has desconectado y deshabilitado todos los mensajes privados entre tu y yo.\nPor cierto, puedes seguir utilizando mi [aplicación web]({{{url}}}).\n\nSi deseas volver a hablar conmigo simplemente escribe *`enable`*."

        },
        'enable' : {
            'en': "It seems that you had enabled me previously, I'm ready to help you.\nPlease type *`help`* to know more about how I can help you.",
                    # 8. "Hey! Welcome back, I'm ready to help you.\nYou can *[login into the web application]({{{url}}})* to check your settings."
            'es': "Parece que ya me habías habilitado previamente, estoy listo para ayudarte.\nPor favor, escribe *`help`* para saber más sobre cómo puedo ayudarte."
                    # 8. "¡Hola! Bienvenido de nuevo, estoy listo para ayudarte.\nPor favor inicia sesión en mi *[aplicación web]({{{url}}})* para confirmar su configuración."
        },
        'extras' : {
            'en': "Hey, your wish is my command, and these are the extra commands",

            'es': "Hola, puedes usar estos comandos extras para interactuar conmigo:"

        },
        'feedback' : {
            'en': "Hey *[user]*, it'd be great to get some feedback from you. If it's something important, I recommend you to visit our support center.",

            'es': "Hola *[user]*!, Sería grandióso recibir tu feedback. Si es algo importante, Te recomiendo visitar nuestro centro de soporte."

        },
        'hello' : {
            'en': "Hey [user]! Have a great day and remember I'm so happy about being your friend.\nType *`help`* to learn more about me.",

            'es': "¡Hola [user]! Ten un excelente día y recuerda que estoy muy feliz de ser tu amigo.\nEscribe *`help`* para aprender más sobre mí."

        },
        'help' : {
            'en': "Hey, your wish is my command, and these are my available commands:",

            'es': "Hola, puedes usar estos comandos para interactuar conmigo:"
        },
        'kudos' : {
            'en': "kudos response: ",
            'es':"kudos response: "
        },
        'news' : {
            'en': "Take a look at *[my blog](https://www.dailybot.co/blog/)* to read about tips, key learnings and other content that our team is sharing.",

            'es': "Echa un vistazo a *[mi blog](https://www.dailybot.com/blog/)* (en inglés) para leer sobre trucos, aprendizajes y otro contenido que comparte nuestro equipo."

        },
        'report' : {
            'en':'No response',
            'es':"No hay respuesta"
        },
        'resetdaily' : {
            'en': "This command has been changed, please type *`help`* to see all current commands.",

            'es': "Este comando ha cambiado, por favor escribe *`help`* para ver todos los comandos actuales."

        },
        'settings' : {
            'en': "This is a quick overview of your settings. You can manage these settings on my *[Web Application]({{{url}}})*:",

            'es': "Este es un resúmen de tu configuración. Puedes gestionar la configuración en mi *[aplicación web]({{{url}}})*:"

        },
        'skipdaily' : {
            'en':"no response",
            'es':"No hay respuesta"
        },
        'snooze' : {
            'en': "Ok, I’ll remind you in *{{time}} minutes*",

            'es': "Ok, te recordaré en *{{time}} minutos*."

        },
        'textformat' : {
            'en':"Hi there! The following is a list of the different formats I support.\nYou can also see this *[support article](https://www.dailybot.com)* to know more about how to build responses with these formats.",

            
            'es': "Hola! La siguiente es una lista de los diferentes formatos que soporto.\nPuedes verlo aquí *[artículo de apoyo](https://www.dailybot.com)* para conocer más acerca de como construir respuestas con estos formatos."

        },
        'thanks' : {
            'en': ["Oh, don't you even mention it. I'm always happy to help.",
                                "It's always my pleasure.",
                                "You are so very welcome.",
                                "It's no trouble at all. 😊",
                                "Anytime! it's always my pleasure. 😊"],
            'es': ["¡Con muuucho gusto!",
                                 "Siempre es un placer.",
                                 "Estoy para ayudarte.",
                                 "A ti por tu tiempo. 😊",
                                 "¡Cuando me necesites! siempre es un placer. 😊"]
        },
        'timezone' : {
            'en': "The time information for the typed timezone is:",

            'es': "La información de la zona horaria introducida es:"

        },
        'NA': {
            'en': "Sorry for not understanding. Please type *`help`* for me to help you.",
            'es': "Lo siento por no entender. Por favor escribe *`help`* para ayudarte."
        }
        
        
    }


# In[44]:


# generate response
def generate_response(query_intent):

        return responses[query_intent]


# In[45]:


# Create list of queries with no response
try:
    no_intent_queries = open("no_intent_queries.txt", "x")
except:
    with open("no_intent_queries.txt", "w") as file:
            file.truncate(0)
            file.close()


# In[46]:


queries_list = ""


# In[47]:


# execute chatbot 

print('Insert command here (type "quit" to stop chatbot)\n')

while 1 >0:
    
    input_text = input()
    if input_text=='quit':
        break
    if len(input_text) < 3:
        while len(input_text) <3:
            input_text = input_text + " "
    #input_text = check_en(input_text)
    prediction = predict_label(np.array(bow(preprocess_input(input_text))))
    response = generate_response(prediction)
    if prediction == 'NA':
            queries_list = queries_list + input_text+'\n'
    # perform spell check before identifying language
    
    
    # identify language of query
    if TextBlob(input_text).detect_language() == 'en':
        if prediction == 'thanks':
            print(random.choice(response['en']))
        else:
            print(response['en'])
    elif TextBlob(input_text).detect_language() == 'es' :
        if prediction == 'thanks':
            print(random.choice(response['es'])+'\n')
        else:
            print(response['es']+'\n')
    else:
        print("Sorry for not understanding. Please type *`help`* for me to help you.")
    print(' ')
        


# In[15]:


# write all unidentified queries in the text file
with open("no_intent_queries.txt", "w") as file:
            file.write(queries_list)
            file.close()


# In[ ]:




