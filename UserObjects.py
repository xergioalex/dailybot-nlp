#!/usr/bin/env python
# coding: utf-8

# In[27]:


class User:
    def __init__(self, first_name, last_name, language):
        self.first_name = first_name
        self.last_name = last_name
        self.language = language
        
    
    def introduce_self(self):
        print('My name is {} {}'.format(self.first_name, self.last_name))
        
    def show_language(self):
        print('My language is ',self.language)

        
    def set_to_Spanish(self):
        self.language = "Spanish"
        
    def set_to_English(self):
        self.language = "English"


# In[28]:


class DailyReport:
    def __init__(self, is_completed):
        self.is_completed = is_completed
        
    def skip_daily(self):
        self.is_completed = True
    
    def cancel(self):
        self.is_completed = False
        
    def show_status(self):
        if self.is_completed == True:
            print('Daily Report Completed')
        else:
            print('Daily Report Not Completed')


# In[29]:


class Settings:
    def __init__(self, is_enabled, given_feedback):
        self.is_enabled = is_enabled
        self.given_feedback = given_feedback
        
    def cancel(self):
        self.given_feedback = False
        
    def disable(self):
        self.is_enabled = False
        
    def enable(self):
        self.is_enabled = True
        
    def show_enabled_status(self):
        if self.is_enabled:
            print('Is enabled')
        else:
            print('Not enabled')
        
        


# In[ ]:





# In[ ]:





# In[30]:


class Datetime:
    def __init__(self, date, time):
        date = self.date
        time = self.time
        


# In[1]:


class Appointment:
    def __init__(self, member, time):
        self.member = member # a list of members
        self.time = time # datetime variable
    
    # show appointments
    def show_appointments(self):
        print('Appointments: \n')
        print('At {} with {}'.format(self.time, self.member))
    
    # add appointment
    
        
        


# In[32]:


aashish_user = User("Aashish", "Nair", "English")
aashish_daily_report = DailyReport("False")
aashish_settings = Settings(is_enabled=True, given_feedback=False)


# In[33]:


aashish_user.daily_report = aashish_daily_report
aashish_user.settings = aashish_settings


# In[ ]:




