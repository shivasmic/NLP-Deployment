# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 17:31:27 2021

@author: SHIVAM
"""

from flask import Flask, render_template, request
import pickle

filename = 'restaurant-reviews-classifier.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('bow-transform.pkl', 'rb'))

webapp = Flask(__name__)

@webapp.route('/')
def home():
    return render_template('index.html')

@webapp.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        test = cv.transform(data).toarray()
        predicted = classifier.predict(test)
        return render_template('result.html', prediction = predicted)
    
if __name__ == '__main__':
    webapp.run(debug = True)
    

