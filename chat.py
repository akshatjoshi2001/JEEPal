import nltk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import random
import json
from flask import Flask
from flask import request

with open("intents.json") as file:
    data = json.load(file)
tags = []
responses = {}
words = []
docs_x = []
docs_y = []
for intent in data['intents']:
    tags.append(intent['tag'])
    responses[intent['tag']]=intent['responses'][0]
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
    
        docs_x.append(wrds)
        docs_y.append(intent['tag'])
        words.extend(wrds)
        

model = keras.models.load_model("chatbot.h5")
            
words = sorted(list(set([stemmer.stem(w.lower()) for w in words])))
tags = sorted(tags)
app = Flask(__name__)

def chat(sentence):
    
    inp = []
    wrd2 = nltk.word_tokenize(sentence)
    wrd2 = [stemmer.stem(w.lower()) for w in wrd2]
    for w in words:
        if w in wrd2:
            inp.append(1)
        else:
            inp.append(0)
    
    return responses[tags[model.predict([inp]).argmax()]]
@app.route('/chat')
def index():
    sent = request.args.get("q")
    return chat(sent)
app.run()
