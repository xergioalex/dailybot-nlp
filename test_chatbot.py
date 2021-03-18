#!/usr/bin/env python
# coding: utf-8

# In[39]:


# libraries
import json
import nltk
import spacy
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
from spellchecker import SpellChecker
import dateparser
# import NLP library
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[40]:


# Load Data

# English Data
data_en = json.loads(open('data_en.json').read())

# Spanish Data
data_es = json.loads(open('data_es.json').read())


# In[41]:


intents_en = data_en['interactionModel']['languageModel']

intents_es = data_es['interactionModel']['languageModel']


# In[42]:


# Create a list of tuples that assigns a list of words to an intent in English
words_en = []
labels_en = []
documents_en = []
punc_en = ['?', '!', ',', '.']

for intent in intents_en['intent']: 
    for sample in intent['samples']:
        word_list_en = nltk.word_tokenize(sample)
        words_en.extend(word_list_en)
        documents_en.append((word_list_en, intent['name']))
        if intent['name'] not in labels_en:
            labels_en.append(intent['name'])


# In[43]:


# Create a list of tuples that assigns a list of words to an intent in Spanish
words_es = []
labels_es = []
documents_es = []
punc_es = ['?', '!', ',', '.']

for intent in intents_es['intent']: 
    for sample in intent['samples']:
        word_list_es = nltk.word_tokenize(sample)
        words_es.extend(word_list_es)
        documents_es.append((word_list_es, intent['name']))
        if intent['name'] not in labels_es:
            labels_es.append(intent['name'])


# In[44]:


lem = WordNetLemmatizer()


# turn words to lower case and to base form
words_en = [lem.lemmatize(word.lower()) for word in words_en if word not in punc_en]
words_es = [lem.lemmatize(word.lower()) for word in words_es if word not in punc_es]


# sort words and remove duplicates in English
words_en = sorted(set(words_en))
labels_en = sorted(set(labels_en))

# sort words and remove duplicates in Spanish
words_es = sorted(set(words_es))
labels_es = sorted(set(labels_es))


# In[45]:


# Load the Spanish and English Models

model_en = load_model('model_en.h5')
model_es = load_model('model_es.h5')


# In[46]:


# performs preprocessing on inputted text (lower casing and lemmatization)
def preprocess_input(text):
    text_token = nltk.word_tokenize(text)
    text_token = [lem.lemmatize(word.lower()) for word in text_token]
    return text_token

# coverts preprocessed text to a matrix form (English)
def bow_en(text_token):
    bag = []
    for word in words_en:
        if word in text_token:
            bag.append(1)
        else:
            bag.append(0)
    return bag

# coverts preprocessed text to a matrix form (Spanish)
def bow_es(text_token):
    bag = []
    for word in words_es:
        if word in text_token:
            bag.append(1)
        else:
            bag.append(0)
    return bag

# inputs user query in matrix form and returns a predicted label (English)
def predict_label_en(bow_en):
    prediction = model_en.predict(np.array(bow_en).reshape((1, len(words_en))))
    max_value = np.max(prediction)
    if max_value > 0.9:
        predict_label = labels_en[np.argmax(prediction)]
    else:
        predict_label = 'NA'
    return predict_label


# inputs user query in matrix form and returns a predicted label (Spanish)
def predict_label_es(bow_es):
    prediction = model_es.predict(np.array(bow_es).reshape((1, len(words_es))))
    max_value = np.max(prediction)
    if max_value > 0.9:
        predict_label = labels_es[np.argmax(prediction)]
    else:
        predict_label = 'NA'
    return predict_label


# In[47]:


# inputs the user intent and returns the appropriate response (English)
def generate_response_en(query_intent, user):
    
    
    responses_en = { 
        'cancel' : "Ok, it's cancelled. Once you're ready, please type *`daily`* to fill out your follow-up reports.",

        'chicken' : "Here comes Arthur!\n\n*[Launch Rubber Chicken]({{{url}}})* 🐔\n\n_Pro tip: you can change the last part of Arthur's URL in your browser in order to create a unique room.",

        'daily' :  "No response",
            
        'dailystatus' : "No response",
            
        'dashboard' :"Ok, please *[follow this link]({{{url}}})* to access the web dashboard.",

        'datetime' :  "The time information for your user is:",
            
        'disable' : "Ok, it's done. You have disconnected and disabled all private messaging between you and me.\nYou can still use the [web application]({{{url}}}).\n\nIf you want to talk to me again, just type *`enable`*",


        'enable' : ["It seems that you had enabled me previously, I'm ready to help you.\nPlease type *`help`* to know more about how I can help you.",
                    
                    "Hey! Welcome back, I'm ready to help you.\nYou can *[login into the web application]({{{url}}})* to check your settings."],
            
        'extras' :"Hey, your wish is my command, and these are the extra commands",
            
        'feedback' : "Hey {}, it'd be great to get some feedback from you. If it's something important, I recommend you to visit our support center.".format(user.first_name),

        'hello' : "Hey {}! Have a great day and remember I'm so happy about being your friend.\nType *`help`* to learn more about me.".format(user.first_name),

        'help' : "Hey, your wish is my command, and these are my available commands:",

        'kudos' :  "kudos response: ",
        
        'news' :  "Take a look at *[my blog](https://www.dailybot.co/blog/)* to read about tips, key learnings and other content that our team is sharing.",

        'report' : 'No response',
        
        'resetdaily' :"This command has been changed, please type *`help`* to see all current commands.",

        'settings' : "This is a quick overview of your settings. You can manage these settings on my *[Web Application]({{{url}}})*:",

        'skipdaily' :"no response",
        
        'snooze' : "Ok, I’ll remind you in *{{time}} minutes*",

        'textformat' : "Hi there! The following is a list of the different formats I support.\nYou can also see this *[support article](https://www.dailybot.com)* to know more about how to build responses with these formats.",

        'thanks' : ["Oh, don't you even mention it. I'm always happy to help.",
                                "It's always my pleasure.",
                                "You are so very welcome.",
                                "It's no trouble at all. 😊",
                                "Anytime! it's always my pleasure. 😊"],
       
        'timezone' : "The time information for the typed timezone is:",

        'NA':  "Sorry for not understanding. Please type *`help`* for me to help you.",
        
        'meeting': "Set an appointment with whom?"
        
        
    }

    return responses_en[query_intent]

    
# inputs the user intent and returns the appropriate response (Spanish)
def generate_response_es(query_intent, user):
    
    responses_es = { 
        'cancel' : "Ok, has cancelado tu seguimiento diario. Por favor escribe *`daily`* para empezar cuando desees.",
    
        'chicken' : "¡Aquí viene Arthur!\n\n[Abrir Rubber Chicken]({{{url}}}) 🐔\n\n_Pro tip: puedes cambiar la última parte de la URL de Arthur en tu navegador y así tendrás una sala única._",

        'daily' :  "No hay respuesta",
        
        'dailystatus' :  "No hay respuesta",
        
        'dashboard' : "Ok, *[abre este enlace]({{{url}}})* para abrir la web y dashboard.",

        'datetime' : "La información horaria de tu usuario es:",

        'disable' : "Listo, está hecho. Has desconectado y deshabilitado todos los mensajes privados entre tu y yo.\nPor cierto, puedes seguir utilizando mi [aplicación web]({{{url}}}).\n\nSi deseas volver a hablar conmigo simplemente escribe *`enable`*.",

        'enable' : "Parece que ya me habías habilitado previamente, estoy listo para ayudarte.\nPor favor, escribe *`help`* para saber más sobre cómo puedo ayudarte.",
                    # 8. "¡Hola! Bienvenido de nuevo, estoy listo para ayudarte.\nPor favor inicia sesión en mi *[aplicación web]({{{url}}})* para confirmar su configuración."
        
        'extras' : "Hola, puedes usar estos comandos extras para interactuar conmigo:",

        
        'feedback' : "Hola {}!, Sería grandióso recibir tu feedback. Si es algo importante, Te recomiendo visitar nuestro centro de soporte.".format(user.first_name),

        
        'hello' :  "¡Hola {}! Ten un excelente día y recuerda que estoy muy feliz de ser tu amigo.\nEscribe *`help`* para aprender más sobre mí.".format(user.first_name),

        
        'help' : "Hola, puedes usar estos comandos para interactuar conmigo:",
        
        'kudos' :"kudos response: ",
        
        'news' : "Echa un vistazo a *[mi blog](https://www.dailybot.com/blog/)* (en inglés) para leer sobre trucos, aprendizajes y otro contenido que comparte nuestro equipo.",

        
        'report' : "No hay respuesta",
        
        'resetdaily' : "Este comando ha cambiado, por favor escribe *`help`* para ver todos los comandos actuales.",

        'settings' : "Este es un resúmen de tu configuración. Puedes gestionar la configuración en mi *[aplicación web]({{{url}}})*:",

        'skipdaily' : "No hay respuesta",
        
        'snooze' : "Ok, te recordaré en *{{time}} minutos*.",

        'textformat' : "Hola! La siguiente es una lista de los diferentes formatos que soporto.\nPuedes verlo aquí *[artículo de apoyo](https://www.dailybot.com)* para conocer más acerca de como construir respuestas con estos formatos.",

        'thanks' : ["¡Con muuucho gusto!",
                                 "Siempre es un placer.",
                                 "Estoy para ayudarte.",
                                 "A ti por tu tiempo. 😊",
                                 "¡Cuando me necesites! siempre es un placer. 😊"],
    
        'timezone' :  "La información de la zona horaria introducida es:",

        'NA': "Lo siento por no entender. Por favor escribe *`help`* para ayudarte.",
        
        'meeting': "Set an appointment with whom?"
    }

    return responses_es[query_intent]


# In[48]:


# Create list of queries with no response
try:
    no_intent_queries = open("no_intent_queries.txt", "x")
    queries_list = ""
except:
    file = open("no_intent_queries.txt","r+")
    file.truncate(0)
    file.close()
    queries_list = ""


# In[49]:


from imp import reload
import UserObjects

reload(UserObjects)

User = UserObjects.User
Settings = UserObjects.Settings
DailyReport = UserObjects.DailyReport
Appointment = UserObjects.Appointment

# create aashish object
aashish_user = User(first_name="Aashish", last_name="Nair", language="English")
aashish_settings = Settings(is_enabled=True, given_feedback=False)
aashish_daily_report = DailyReport(is_completed=True)
aashish_meeting = Appointment(member=None, time=None)

aashish_user.settings = aashish_settings
aashish_user.daily_report = aashish_daily_report
aashish_user.appointment = aashish_meeting

# create sergio object
sergio_user = User(first_name="Sergio", last_name="Alex", language="Spanish")
sergio_settings = Settings(is_enabled=True, given_feedback=False)
sergio_daily_report = DailyReport(is_completed=True)
sergio_meeting = Appointment(member=None, time=None)

sergio_user.settings = sergio_settings
sergio_user.daily_report = sergio_daily_report
sergio_user.appointment = sergio_meeting


# In[50]:


# chatbot function
def chatbot(user):
    
    
    queries_list = ""
    
    print('Insert command here (type "quit" to stop chatbot)\n')

    while 1 >0:
        
        input_text = input()
        if input_text=='quit':
            return queries_list
        
        if len(input_text) < 3:
            while len(input_text) <3:
                input_text = input_text + " "
                
                
#----------------------------------------------------- English Bot -----------------------------------------------------                


        if user.language =='English':

            spell = Speller(lang='en')
            if input_text not in intents_en:
                input_text = spell(input_text)
                
            prediction_en = predict_label_en(np.array(bow_en(preprocess_input(input_text))))

            # thanks intent
            if prediction_en == 'thanks':
                response_en = random.choice(generate_response_en(prediction_en, user))
                print(response_en)
                print(' ')

            # enable intent
            elif prediction_en == 'enable':
                response_en = generate_response_en(prediction_en, user)
                if user.settings.is_enabled: 
                    response_en = response_en[0]
                else:
                    user.settings.enable()
                    response_en = response_en[1]
                print(response_en)
                print(' ')

            # disable intent                   
            elif prediction_en == 'disable':
                user.settings.disable()
                response_en = generate_response_en(prediction_en, user)
                print(response_en)
                print(' ')

            elif prediction_en == 'skipdaily':
                response_en = generate_response_en(prediction_en, user)
                user.daily_report.skip_daily()
                print(response_en)
                print(' ')

            elif prediction_en == 'cancel':
                response_en = generate_response_en(prediction_en, user)
                user.daily_report.cancel()
                print(response_en)
                print(' ')

            elif prediction_en == 'meeting':
                response_en = generate_response_en(prediction_en, user)
                print("Bot: ",response_en)
                print(' ')
                appointment_with = input()
                print('Which day?')
                time = ""
                appointment_day = input()
                time = time + appointment_day + " "
                print('What time?')
                appointment_time = input()
                time = time + appointment_time
                
                date = dateparser.parse(time)
                user.appointment = Appointment(member=appointment_with, time=date)
                
                print('Appointment scheduled at {} on {} with the following people: {}'.format(appointment_day, appointment_time, appointment_with))
                print(' ')
                
            elif prediction_en == 'meeting status':
                if vars(user.appointment)['time'] is None:
                    print('You have no meetings')
                else:
                    user.appointment.show_appointments()
                
            else:
                response_en = generate_response_en(prediction_en, user)
                print(response_en)
                print(' ')
            if prediction_en == 'NA':
                queries_list = queries_list + input_text+ '\n'
                
                
                
#----------------------------------------------------- Spanish Bot -----------------------------------------------------                

        elif user.language == 'Spanish':
        
            spell = Speller(lang='es')
            if input_text not in intents_en:
                input_text = spell(input_text)
           
            prediction_es = predict_label_es(np.array(bow_es(preprocess_input(input_text))))

            # thanks intent
            if prediction_es == 'thanks':
                response_es = random.choice(generate_response_es(prediction_es, user))
                print(response_es)
                print(' ')

            # enable intent
            elif prediction_es == 'enable':
                response_es = generate_response_es(prediction_es, user)
                if user.settings.is_enabled: 
                    response_es = response_es[0]
                else:
                    user.settings.enable()
                    response_es = response_es[1]
                print(response_es)
                print(' ')

            # disable intent                   
            elif prediction_es == 'disable':
                user.settings.disable()
                response_es = generate_response_es(prediction_es, user)
                print(response_es)
                print(' ')

            elif prediction_es == 'skipdaily':
                response_es = generate_response_es(prediction_es, user)
                user.daily_report.skip_daily()
                print(response_es)
                print(' ')

            elif prediction_es == 'cancel':
                response_es = generate_response_es(prediction_es, user)
                user.daily_report.cancel()
                print(response_es)
                print(' ')

            elif prediction_es == 'meeting':
                response_es = generate_response_es(prediction_es, user)
                print("Bot: ",response_es)
                print(' ')
                appointment_with = input()
                print('Which day?')
                time = ""
                appointment_day = input()
                time = time + appointment_day + " "
                print('What time?')
                appointment_time = input()
                time = time + appointment_time
                
                date = dateparser.parse(time)
                user.appointment = Appointment(member=appointment_with, time=date)
                
                print('Appointment scheduled at {} on {} with the following people: {}'.format(appointment_day, appointment_time, appointment_with))
                print(' ')
                
            elif prediction_es == 'meeting status':
                if vars(user.appointment)['time'] is None:
                    print('You have no meetings')
                else:
                    user.appointment.show_appointments()
                
            else:
                response_es = generate_response_es(prediction_es, user)
                print(response_es)
                print(' ')
            if prediction_es == 'NA':
                queries_list = queries_list + input_text+ '\n'


# In[53]:

# Use "aashish_user" as parameter for English and "sergio_user" as parameter for Spanish
queries_list = chatbot(sergio_user)


# In[54]:


# add "NA" queries to the .txt file
file = open('no_intent_queries.txt', 'a')
# Append 'hello' at the end of file
file.write(queries_list)
# Close the file
file.close()


# In[ ]:





# In[ ]:





# In[ ]:




