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
from flask_cors import CORS
with open("intents.json") as file:
    data = json.load(file)
tags = []
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 

responses = {}
words = []
docs_x = []
docs_y = []
for intent in data['intents']:
    tags.append(intent['tag'])
    responses[intent['tag']]=intent['responses'][0]
    for pattern in intent['patterns']:
        
        wrds1 = nltk.word_tokenize(pattern)
        wrds = []
        for w in wrds1:
            if w not in stop_words:
                wrds.append(w)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])
        words.extend(wrds)
        
        
            
    

model = keras.models.load_model("chatbot.h5")
            
words = sorted(list(set([stemmer.stem(w.lower()) for w in words])))
tags = sorted(tags)
app = Flask(__name__)
CORS(app)

def chat(sentence):
    
    inp = []
    wrd2 = nltk.word_tokenize(sentence)
    wrd2 = [stemmer.stem(w.lower()) for w in wrd2]
    for w in words:
        if w in wrd2:
            inp.append(1)
        else:
            inp.append(0)
    cls = model.predict([inp]).argmax()
    prob = np.max(model.predict([inp]))
    if(prob>=0.5):
        return responses[tags[cls]]
    else:
        return "Sorry. I didn't understand, what you wanted. I really strive to improve daily. If you have a serious doubt, you can mail Akshat at akshatjoshi@smail.iitm.ac.in"
@app.route('/chat')
def index():
    sent = request.args.get("q")
    return chat(sent)

    
app.run(host='::',port = 80)
