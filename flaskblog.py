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
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# nltk.download('stopwords')
# nltk.download('punkt')
app = Flask(__name__)
CORS(app)

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")

chromadb_mapping = {
    'novel-1.txt': './chroma/chroma_db_sherlock',
    'novel-2.txt': './chroma/chroma_db_alice',
    'novel-3.txt': './chroma/chroma_db_good',
    'novel-4.txt': './chroma/chroma_db_primitive',
    'novel-5.txt': './chroma/chroma_db_pigs_is_pigs',
    'novel-6.txt': './chroma/chroma_db_usher',
    'novel-7.txt': './chroma/chroma_db_magi',
    'novel-8.txt': './chroma/chroma_db_the_jungle_book',
    'novel-9.txt': './chroma/chroma_db_redroom',
    'novel-10.txt': './chroma/chroma_db_warrior'
}
try:
    with open("./open-ai-key.txt", "r") as file:
        open_ai_key = file.readline().strip()
        print('open_ai_key : ', open_ai_key)
except FileNotFoundError:
    open_ai_key = ""
llm = OpenAI(openai_api_key=open_ai_key)

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

@app.route('/novel', methods=['POST'])
def novel():
    try:
        data = request.json
        # print('data : ', data)
        query = data['user_input']

        book_name = ''
        embeddings_model = HuggingFaceEmbeddings()
        persistent_directory = './chroma/chroma_db'
        try:
            loaded_vectordb = Chroma(persist_directory=persistent_directory, embedding_function=embeddings_model)
        except FileNotFoundError:
            print("Chroma vector store not found. You may need to run the vectorization process first.")
            vectordb = None
        # query = "what is the story of Alice"
        docs = loaded_vectordb.similarity_search(query, k=5)
        for rank, doc in enumerate(docs):
            print(f"Rank {rank+1}:")
            # print(doc.page_content)
            print(doc.metadata)
            print("\n")
            book_name = doc.metadata['source']
            break
        book_name = book_name[2:]
        print('book name : ', book_name)

        persistent_directory = chromadb_mapping[book_name]
        try:
            final_vectordb = Chroma(persist_directory=persistent_directory, embedding_function=embeddings_model)
        except FileNotFoundError:
            print("Chroma vector store not found. You may need to run the vectorization process first.")
            vectordb = None
        docs = final_vectordb.similarity_search(query, k=5)
        for rank, doc in enumerate(docs):
            print(f"Rank {rank+1}:")
            print(doc.page_content)
            print(doc.metadata)
            print("\n")
            break

        #RAG--------------------------------
        new_line = '\n'
        template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
        qa_chain = RetrievalQA.from_chain_type(llm,
            retriever=final_vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
        result = qa_chain({"query": query})
        print('result  : ', result)
        return jsonify({'novel': result["result"].strip()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
