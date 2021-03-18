# DailyBot-Third-Prototype


## Introduction

This is my third prototype of the DailyBot chatbot. The final microservice that is expected to be built should have the following requirements:

1. Be trained with a base of predefined set of intentions.
2. Recognize the intent of a received message.
3. Support English and Spanish
4. Provide an intention-based response generator

During the construction of the chatbot, other features were added to the scope of the project. These include:

1. A feature that saves user queries that are not identified by the bot
2. A "meeting" intent that allows you to arrange meetings
3. The incorporation of user data in the chatbot's response

This prototype aims to address all of these requirements. 

## Files

Here is a rundown on the files in the repository.

The "data_en" and "data_es" files are json files that store the data used to train the models. The models are saved in the repo as "models_en.h5" and "models_es.h5". 

The "test_chatbot.py" file is a python file that allows users to input queries for the chatbot. It will naturally require the proper libraries and modules to run correctly. The dependencies for the code is recorded in the "requirements.txt" file. 

The "UserObjects.py" file is a python file that creates classes and objects that are used to imitate user data. The classes in this file are imported into the code used to run the chatbot. 

Finally, the "Chatbot.ipynb" file is a comprehensive python notebook file that shows all the steps taken to build the chatbot and utilize it. It covers the machine learning and natural language processing techniques used to build the model and train it to classify user intent. It then shows how the chatbot uses the model in tandem with its response generator.

## Running the Code

Running the code should be straightforward. For users wishing to simply try the chatbot, running the "test_chatbot.py" will be enough. It should be noted that the libraries in the "requirements.txt" file should be installed prior to running the code. Furthermore, the code requires the data from the json files as well as the models, so running the code will require downloading/cloning the whole repo. 

The code in "test_chatbot.py" currently uses data from 2 fake user objects: "aashish_user" and "sergio_user". To test the chatbot in English, set line 536 to "queries_list = chatbot(aashish_user)". To test the chatbot in Spanish, set line 536 to "queries_list = chatbot(sergio_user)".



## Conclusion

Overall, the prototype brings us much closer towards realizing the standards initially set from DailyBot at the start of this project. 

There are some noteworthy limitations to this prototype, though. For example, the Spanish data is not as comprehensive as the English data and needs more sentence patterns added to it. Next, the user objects used when running the bot are merely imitations and do not replicate the DailyBot's user database. Finally, the idea of using a "meeting" intent needs to be revisited. From the perspective of DailyBot employees, it is important to consider what the most important role of such an intent is. 

With that being said, the chatbot has developed sustainability. If new intents and sentence patterns are added, the model can be retrained and still accurately predict user intent. Also, although the bot is not currently using DailyBot's data, replacing data from the current classes and objects with the actual user data should be a simple task. It does not need a practitioner of machine learning and natural language processing. 
>>>>>>> e66e5acd58fed19420a8132634c079ea4b67f847
