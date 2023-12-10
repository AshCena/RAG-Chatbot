from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import json
import nltk
from nltk.corpus import stopwords
import re
import joblib
# nltk.download('stopwords')
# nltk.download('punkt')
app = Flask(__name__)
CORS(app)

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.json
        print('data : ', data)
        user_input = data['user_input']
        print('user_input : ', user_input)

        input_len = len(user_input)
        print('input_len : ', input_len)
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        response_ids = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        generated_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print('response id : ', response_ids)

        return jsonify({'generated_rerereresponse': generated_response[input_len:]})

    except Exception as e:
        return jsonify({'error': str(e)})
        
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    words_without_stop = [word for word in words if word not in stop_words]
    if len(words_without_stop)>0:
        return ' '.join(words_without_stop)
    return ' '.join(words)

@app.route('/chitchatclassifier', methods=['POST'])
def chitchat_classifier():
    try:
        data = request.json
        print('data : ', data)
        user_input = data['user_input']

        loaded_model = joblib.load('svm_model_new.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        user_input = preprocess_text(user_input)
        print('user_input : ', user_input)
        new_texts = [user_input, user_input]
        new_texts_tfidf = tfidf_vectorizer.transform(new_texts)
        predictions = loaded_model.predict(new_texts_tfidf)
        print(predictions)
        return jsonify({'isChitchat': str(predictions[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
