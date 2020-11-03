# -*- coding: utf-8 -*-
"""
@author: yukti
"""

"""
The dataset classifies mails as either spam or ham
"""
import re
import nltk
import pandas as pd
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# the doc words is tab separated not space separated

# STEP-1 : IMPORTING THE DATASET FOR SPAM CLASSIFICATION

# \t to divide into two columns- label(spam/ham) and mail message 
# creating message dataset
msg = pd.read_csv('SpamCollection',sep='\t',names=["label","mail"])
print("1. Importing done...")

# STEP-2 : DATA CLEANING AND PREPROCESSING
ps = PorterStemmer()
lem =  WordNetLemmatizer()
corpus = []
# removing all stop words and stemming words
for i in range(len(msg)):
    s = re.sub('[^a-zA-Z]',' ',msg['mail'][i])
    # converting the case to lower
    s = s.lower()
    # splitting sentence into words
    s = s.split()
    # removing stop words and applying stemming 
    #s = [ps.stem(w) for w in s if not w in stopwords.words('english')]
    s = [lem.lemmatize(w) for w in s if not w in set(stopwords.words('english'))]
    s = ' '.join(s)
    corpus.append(s)

# Data is huge, so here cleaning takes the maximum time 
print("2. Dataset cleaning done...")

# STEP-3: BAG OF WORDS MODEL
    
from sklearn.feature_extraction.text import CountVectorizer
# initialising the countvectorizer object
#cv = CountVectorizer()
# selecting the most frequent columns, as some words will be very less frequent than others
cv = CountVectorizer(max_features=3000) # check using X.shape

# independent feature 
X = cv.fit_transform(corpus).toarray()

# converting to dummy variable fo categorical variables using pandas
y = pd.get_dummies(msg['label'])
# avoiding dummy variable trap : removing one column from two as only one is required for whole info (whether ham or spam)
y = y.iloc[:,1].values
# y is the dependent feature 

print("3. Bag of words model done...")

# STEP-4: SPLITTING TRAINING AND TEST DATA
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state = 0)

print("4. Splitting train and test data done...")

# STEP-5: TRAINING MODEL USING NAIVE BAYES CLASSIFIER (good for nlp)
from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB().fit(X_train,y_train)

print("5. Training using Naive Bayes done...")

# STEP-6: PREDICT USING MODEL AND TEST DATA
y_predicted = spam_detection_model.predict(X_test)

print("6. Prediction done...")
# STEP-7: CONFUSION MATRIX TO CHECK ACCURACY OF PREDICTION
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test,y_predicted)

print("7. Confusion matrix done...")
 

# STEP-8: ACCURACY SCORE
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_predicted)
print("ACCURACY SCORE OF SPAM CLASSIFICATION : ",accuracy)










