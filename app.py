import os
import nltk
from textblob import TextBlob
from wordcloud import  STOPWORDS
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

from nltk.data import find

def setup_nltk():
    try:
        # Check if stopwords, wordnet, and punkt data are already downloaded
        find('corpora/stopwords.zip')
        find('corpora/wordnet.zip')
        find('tokenizers/punkt.zip')
        print("NLTK data already downloaded.")
    except LookupError:
        # If any of the data is not downloaded, download them
        print("Downloading NLTK data...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')

setup_nltk()  # Call this function to check and download NLTK data if necessary

stop_words=set(STOPWORDS)
stop_words=stop_words.union(set(stopwords.words('english')))
stop_words.add('im')
stop_words.add('u')
stop_words.add('got')
def clean_text(text):
    text = text.str.lower()    # lowercase 
    text = text.str.strip()  # removing all spaces (useless)
     
    for a in text.index:
        #print(round(a/pd.Series(text.index).iloc[-1],3), end='\r')
        text[a]=GoogleTranslator(source='auto', target='en').translate(text[a])
        text[a] = re.sub(r"[-()\"#/@;:{}`+=~|_.!?,'0-9]", " ", text[a])        # getting rid of special characters and numbers
        text[a] = ' '.join( [w for w in text[a].split() if len(w)>1] )
        text[a] = re.sub(r' +', ' ', text[a])
        text[a] = re.sub(r'[^a-zA-Z_]', ' ', text[a])
        text[a]=re.sub('bakwaas','useless',text[a])
        text[a]=re.sub('not so good','bad',text[a])
        text[a]=re.sub('not good','bad',text[a])
        text[a]=re.sub('not at all good','bad',text[a])
    text = text.apply(lambda x: " ".join(lemmatizer.lemmatize(y) for y in x.split() ))
    text = text.apply(lambda x: " ".join(y for y in x.split() if y not in stop_words))  # removing stopwords
    text=text.apply(lambda x: " ".join(word_tokenize(x)))
    text = text.apply(lambda x: " ".join(y for y in x.split() if wordnet.synsets(y)))   # keeping words with some meaning, can be refined for respective business over time  
    return(text)

import pickle
from flask import Flask, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections.abc import Sequence
# Initialize Flask app
app = Flask(__name__)


# Change this to vectorizer file iteself
# Load the trained model from local directory
with open("/tmp/bert_model_100.pkl", "rb") as model_file:
    bert_model = pickle.load(model_file)
with open('/tmp/sentiment_analyzer.pkl','rb') as f:
    sentiment_analyzer=pickle.load(f)
bert_mapping={-1: 'Payment, Money and Slow-loading Related Issues',
 0: 'Positive Review',
 1: 'Negative Review',
 2: 'Payment, Money and Refund Related',
 3: 'Vague',
 4: 'Performance related Review',
 5: 'App load/opening related issues and reviews',
 6: 'Seat type/booking, preference and food related reviews',
 7: 'Poor Review',
 8: 'Update/Upgrade related reviews',
 9: 'Account, Login Credentials/Password, Registration related reviews',
 10: 'Service Related Review', # mention it can be positive or negative, hence we derive sentiment analysis P/N
 11: 'user experience related review', # P/N
 12: 'Generic Postive Review',
 13: 'Positive Review',
 14: 'Positive Review speaking of usefullness/helpful/informative features',
 15: 'Advertisement related problems and reviews',
 16: 'Negative Review',
 17: 'Technical problem in opening/accessing mobile app and data',
 18: 'Positive review about user friendly UI/UX',
 19: 'Generic Reviews', #issue P/N
 20: 'Generic Review about online platform',#issue P/N
 21: 'Login related issues/reviews',
 22: 'Reviews related to technical bugs/glitches',
 23: 'General Poor Review', # "low price" replace in clean_func # issue
 24: 'Station Names and other boarding details related reviews',
 25: 'Email/OTP/Mobile Number Verification and validation related reviews',
 26: 'General Poor Reviews',
 27: 'Vague',
 28: 'Generic Reviews',#issue P/N
 29: 'App crash and emergency/tatkal feature related reviews',
 30: 'Reviews addressing India or States of India, Indian Railways Dept. and their services',
 31: 'App repsonsiveness and Customer Care and Complaints related reviews',
 32: 'Reviews addressing rating of the app and its services',
 33: 'Internet connection related reviews',
 34: 'Negative Reviews',
 35: 'Negative Reviews',
 36: 'Negative Reviews',
 37: 'App hanging related reviews',
 38: 'Train and IRCTC services- related reviews addressing general and cleanliness related problems',
 39: 'Different Error popups in the application and mobile permissions related reviews',
 40: 'Journey and Travel related reviews',
  41: 'Negative Reviews', #related to download
 42: 'Reliability related positive reviews',
 43: 'Negative Reviews',#irrititating related
 44: 'Interface Related Reviews',
 45: 'Notifications and Advertisements related reviews',
 46: 'Searching feature and current ticket/seats status related reviews',
 47: 'Vague reviews with negative possibility',
 48: 'Negative feedback on App development process and other UI related reviews',
 49: 'Various/New Features related reviews',
 50: 'Master List (feature) rleated reviews',
  51: 'Negative Reviews',
 52: 'Unauthorized Device/Access and Invalid Requests related reviews',
 53: 'Negative reviews',
 54: 'Generic Reviews', #P/N
 55: 'Ads related reviews',
 56: 'Positive Reviews',
 57: 'Platform/System related general reviews',
 58: 'Negative Reviews',
 59: 'Train chart rleated reviews',
 60: 'ticket confirmation related reviews',
  61: 'Pin, Captcha and biometric related problems/reviews',
 62: 'App buffereing related issues',
 63: 'Postive Reviews',
 64: 'Review related to duration/years of the application',
 65: 'User satisfaction related reviews',
 66: 'Ticket and Hidden/Extra charges related reviews',
 67: 'Vague',
 68: 'General/Average Review',
 69: 'User disatisfaction/satisfaction related reviews',
 70: 'Generic Reviews',
 71: 'General Reviews referring to app support',
 72: 'Vague (Slight Negative) Reviews',
 73: 'Railway coaches related reviews',
 74: 'Reviews referring to smoothness of app experience',
 75: 'General Positive Reviews',
 76: 'General user experience of different facilities reviews',
 77: 'Different Quotas Related Reviews/Issues',
 78: 'Vague',
 79: 'General experience/comfort related reviews',
 80: 'Information/Ticket Details related Reviews',
 81: 'Vague (spelling error)',
 82: 'Negative Reviews', # spelling west-waste
 83: 'Generic Reviews',
 84: 'Vague',
 85: 'Signup/Signin Related Reviews',
 86: 'Positive Reviews',
 87: 'Font Size related reviews',
 88: 'Postive Reviews', # spelling excellent- exile
 89: 'Negative Reviews',
 90: 'Dark Mode/Theme related reviews',
 91: 'Different functions related reviews',
 92: 'Reviews comparing counter ticket vs app',
 93: 'Vague',
 94: 'AC related reviews',
 95: 'Vague',
 96: 'Print feature related reviews',
 97: 'automatic logout/activity related reviews',
 98: 'General Reviews'}
# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json
        # Assuming input is a list of features
        input_string = data['input_string']
        input_string_cleaned=clean_text(pd.Series(input_string))
        topic_num,proba=bert_model.transform(input_string_cleaned.to_list())
        sentiment=sentiment_analyzer(input_string)

        # Return the prediction as a JSON response
        return {'prediction': bert_mapping[topic_num[0]],'probability':proba[0],'sentiment':sentiment[0]['label'],'sentiment_score':sentiment[0]['score']}
    except Exception as e:
        return {'error': str(e)}

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
