# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:26:49 2021

@author: SHIVAM
"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pickle

data = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
corpus = []
for i in range(0, 1000):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
    
# Creating the pickle for both the Bag of Words model as well as the Classifier model 
pickle.dump(cv, open('bow-transform.pkl', 'wb'))
filename = 'restaurant-reviews-classifier.pkl'
pickle.dump(classifier, open(filename, 'wb'))




    
    



