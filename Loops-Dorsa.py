#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df1 = pd.read_csv("ticketmaster.csv")
df1.head(5)


# In[3]:


df1.columns


# In[ ]:





# In[124]:


# Connect and create database

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost", port = "3306",
  user="root", password = "Ryan20@gtown"
)

mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE loopsdb2")


# In[125]:


# Connect to database
mydb = mysql.connector.connect(
  host="localhost", port = "3306",
  user="root", password = "Ryan20@gtown", database = "loopsdb2"
)


# In[126]:


# Create table

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Ryan20@gtown",
  database="loopsdb2"
)

mycursor = mydb.cursor()

mycursor.execute("CREATE TABLE ticketmaster9 (id INT AUTO_INCREMENT PRIMARY KEY, title LONGTEXT, sentiment VARCHAR(255), product_entities LONGTEXT)")


# In[128]:


# Add titles into table

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Ryan20@gtown",
  database="loopsdb2"
)

mycursor = mydb.cursor()

sql = "INSERT INTO ticketmaster9 (title) VALUES (%s)"
val = []
for i in df1.index:   
    val.append(str(df1['title'][i]))
val = [x for x in zip(*[iter(val)])]
mycursor.executemany(sql, val)
mydb.commit()


# In[129]:


# Access data from table

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Ryan20@gtown",
  database="loopsdb2"
)

mycursor = mydb.cursor()
sql = "SELECT title FROM ticketmaster9"
mycursor.execute(sql)

myresult = mycursor.fetchall()


# In[130]:


# Check data in table

len(myresult)


# In[131]:


myresult[1:5]


# In[132]:


df1['title'][1:5]


# In[121]:


# Check maximum number of tokens, since BERT/RoBERTa model only considers sequences
#  lengths of up to 512 tokens (uses truncation if needed) 
lengths=[]
for i in range(0, len(df_title_list)-1):
    #print(df_title_list[i])
    lengths.append(len(str(df_title_list[i]).split()))

max(lengths)


# 1. Classify titles into sentiment categories using pretrained RoBERTa model from Sentence Transformers package
# 

# In[119]:


# Get BERT embeddings for sentiments and titles

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('stsb-roberta-large')
model.max_seq_length


# In[133]:


sentences = ['Need Help Understanding', 'Application Error', 'New Feature Request', 'Terminate Service']

for i in range(0, len(myresult) - 1):
    sentences.append(myresult[i][0])
    
sentence_embeddings = model.encode(sentences)


# In[134]:


# Assign sentiment based on which has the closest cosine similarity to each title

from sentence_transformers import util
import torch
import numpy as np
cosine_matrix_test1 = np.empty((0, 4), int)

# First calculate all cosine similarities
for title_ind in range(0, len(myresult) - 1):
    curr_row = []
    for sentiment_ind in range(0, 4):
        curr_similarity = util.pytorch_cos_sim(sentence_embeddings[title_ind + 4], sentence_embeddings[sentiment_ind])
        curr_row.append(curr_similarity)
    curr_row_final = np.array(curr_row)
    cosine_matrix_test1 = np.vstack((cosine_matrix_test1, curr_row_final))


# Assign highest cosine similarity
indices_top_sentiment =[]
for row in cosine_matrix_test1:
    indices_top_sentiment.append(np.argmax(row))
            


# In[135]:


print(indices_top_sentiment)


# In[92]:


# Insert sentiments into database table

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Ryan20@gtown",
  database="loopsdb2"
)

mycursor = mydb.cursor()

sql = "INSERT INTO ticketmaster9 (sentiment) VALUES (%s)"
val = []
for i in range(0, len(indices_top_sentiment) - 1):   
    val.append(str(sentences[indices_top_sentiment[i]]))
val = [x for x in zip(*[iter(val)])]
mycursor.executemany(sql, val)
mydb.commit()


# 2. Find Product Entities using pretrained model from Flair package
# 
# -Categories: Person, Norp, Fac, Org, Gpe, Loc, Product, Event, Work_Of_Art, Law, Language, Date, Time, Percent, Money, Quantity, Ordinal, Cardinal
# -Model: word embeddings (bidrectional character language model) inputted into bidirectional lstm 
# -Data: telephone conversations, newswire, newsgroups, broadcast news, broadcast conversation, weblogs

# In[96]:


from flair.models import SequenceTagger
model = SequenceTagger.load('ner-ontonotes-fast') 

from flair.data import Sentence
pos = []
for i in range(0, len(myresult) - 1):
    sentence_curr = Sentence(myresult[i][0])
    model.predict(sentence_curr)
    pos.append(sentence_curr.to_tagged_string())


# In[97]:


pos


# In[ ]:


# Insert titles with entities embedded into database table

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Ryan20@gtown",
  database="loopsdb2"
)

mycursor = mydb.cursor()

sql = "INSERT INTO ticketmaster9 (product_entities) VALUES (%s)"
val = []
for i in range(0, len(pos) - 1):   
    val.append(str(pos))
val = [x for x in zip(*[iter(val)])]
mycursor.executemany(sql, val)
mydb.commit()


# In[ ]:




