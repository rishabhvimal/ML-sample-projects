# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:12:31 2019

@author: risvimal
"""

# importing all libaries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, r2_score, accuracy_score,classification_report
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Loading data
train = pd.read_csv(r'C:\Users\risvimal\Desktop\New folder\training_1.csv')
test = pd.read_csv(r'C:\Users\risvimal\Desktop\New folder\test (1).csv')

# Checking Data length
length = train.shape[0]
length_test = test.shape[0]

# Data preprocessing on train set
for i in range(length):
    print('the value of i:' + str(i))
    train.text[i] = train.text[i].split()
    
# Data preprocessing on test set
for i in range(length_test):
    print('the value of i:' + str(i))
    test.text[i] = test.text[i].split()

# Upscaling data as it is highly imbalanced
train_majority = train[train['category']==0]
train_minority = train[train['category']==1]

train_minority_upsampled = resample(train_minority,replace =True, n_samples=3348,random_state=111)
train_upsampled = pd.concat([train_majority,train_minority_upsampled])

# Creating vocabalary of words to prepare count vector
vocab = {}
counter = 0

new = train.text

for i in range(length):
    print('the value of i:' + str(i))
    a = new[i]
    for j in range(len(a)):
        if a[j] in vocab:
            continue
        else:
            vocab[a[j]] = counter
            counter += 1

# Adding word which are missing from train set but present in test set
new1 = test.text

for i in range(length_test):
    a = new1[i]
    for j in range(len(a)):
        if a[j] in vocab:
            continue
        else:
            vocab[a[j]] = counter
            counter += 1
# resetting the upsampled data indexes
train_upsampled.reset_index(inplace = True)

# creating count vector
count_vector = []
for i in range(len(train_upsampled)):
    print('the value of i:' + str(i))
    blank = [0]*len(vocab)
    a = train_upsampled.text[i]
    print(a)
    for j in range(len(a)):
        word = a[j]
        blank[vocab[word]]+=1
    count_vector.append(blank)
    
# transform count to frequencies with TF-idf
tfidf = TfidfTransformer()
X = tfidf.fit_transform(count_vector).toarray()
y = train_upsampled.category.values

# Dividing train data in train set and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model on train set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# making prediction on validation set
y_pred = classifier.predict(X_test)

# Evaluating the model
cm = confusion_matrix(y_test, y_pred)
print(cm)

score_r2 = r2_score(y_test,y_pred)
print(score_r2)

score_f1 = f1_score(y_test,y_pred)
print(score_f1)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

print(classification_report(y_test,y_pred))

# creating count vector for test data
count_vector_test = []
for i in range(length_test):
    blank = [0]*len(vocab)
    a = new1[i]
    for j in range(len(a)):
        word = a[j]
        blank[vocab[word]]+=1
    count_vector_test.append(blank)

# transform count to frequencies with TF-idf
X_out = tfidf.transform(count_vector_test).toarray()

# making prediction on test set
test_pred = classifier.predict(X_out)

# making function for file submission
def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'id':test.id,'category':test_pred})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

#making final csv file
make_submission(test_pred, 'final_submission')