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
import matplotlib.pyplot as plt
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from collections import defaultdict

import pickle
import time

# nltk.download('stopwords')
# nltk.download('punkt')
app = Flask(__name__)
CORS(app)

response_times = []


model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")

novel_query_counter = defaultdict(int)
user_queries = []

global chitchat_counter, novel_counter
chitchat_counter = 0
novel_counter = 0
chromadb_mapping = {
    'novel-1.txt': './chroma/chroma_db_sherlock',
    'novel-2.txt': './chroma/chroma_db_alice',
    'novel-3.txt': './chroma/chroma_db_good',
    'novel-4.txt': './chroma/chroma_db_primitive',
    'novel-5.txt': './chroma/chroma_db_pigs_is_pigs',
    'novel-6.txt': './chroma/chroma_db_usher',
    'novel-7.txt': './chroma/chroma_db_magi',
    'novel-8.txt': './chroma/chroma_db_jungle',
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


def generate_pie_chart():
    novel_names_mapping = {
        'novel-1.txt': 'The Adventures of Sherlock Holmes',
        'novel-2.txt': 'Alice\'s Adventures in Wonderland',
        'novel-3.txt': 'And It Was Good',
        'novel-4.txt': 'Into the Primitive',
        'novel-5.txt': 'Pigs is Pigs',
        'novel-6.txt': 'The Fall of the House of Usher',
        'novel-7.txt': 'The Gift of the Magi',
        'novel-8.txt': 'The Jungle Book',
        'novel-9.txt': 'The Red Room',
        'novel-10.txt': 'Warrior of Two Worlds'
    }

    with open('novel_pie_count.pkl', 'rb') as file:
        novel_query_counter = pickle.load(file)

    novels = [novel_names_mapping[key] for key in novel_query_counter.keys()]
    query_distribution = list(novel_query_counter.values())

    plt.figure(figsize=(8, 8))
    plt.pie(query_distribution, labels=novels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('Distribution of User Queries Across Novels')
    plt.savefig('./chatbot/public/pie_chart.png')


def generate_bar_graph(queries):
    all_text = ' '.join(queries)
    processed_text = preprocess_text(all_text)
    words = nltk.word_tokenize(processed_text)
    stop_words = set(stopwords.words('english'))
    words_without_stop = [word for word in words if word not in stop_words]
    word_freq = nltk.FreqDist(words_without_stop)
    common_words = dict(word_freq.most_common(10))
    plt.figure(figsize=(10, 6))
    plt.bar(common_words.keys(), common_words.values(), color='blue')
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title('Most Common Words in User Queries')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./chatbot/public/bar_graph.png')

def generate_doughnut_chart():
    global chitchat_counter, novel_counter
    # Plotting a Doughnut Chart for Chit-Chat to Novel-Related Query Ratio
    with open('chitchat_count.pkl', 'rb') as file:
        chitchat_counter = pickle.load(file)

    with open('novel_count.pkl', 'rb') as file:
        novel_counter = pickle.load(file)

    plt.figure(figsize=(8, 8))
    plt.pie([chitchat_counter, novel_counter], labels=['Chit-Chat', 'Novel-Related'], autopct='%1.1f%%', startangle=90,
        colors=['#FF6347', '#3498db'])  # Hex color codes for 'rgb(255, 99, 132)' and 'rgb(54, 162, 235)'
    plt.title('Chit-Chat to Novel-Related Query Ratio')
    plt.savefig('./chatbot/public/doughnut_chart.png')


def line_chart():
    # Plot a line chart for response times

    with open('response_time.pkl', 'rb') as file:
        response_times = pickle.load(file)
    plt.figure(figsize=(10, 6))
    plt.plot(response_times, marker='o', linestyle='-', color='b', label='Query Response Time')

    # Calculate and plot average response time
    avg_response_time = sum(response_times) / len(response_times)
    plt.axhline(y=avg_response_time, linestyle='--', color='r', label='Average Response Time')

    plt.xlabel('Query Number')
    plt.ylabel('Response Time (seconds)')
    plt.title('Chatbot Response Time for Different Queries')
    plt.legend()
    plt.grid(True)
    plt.savefig('./chatbot/public/response_time_chart.png')

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.json
        print('data : ', data)
        user_input = data['user_input']
        print('user_input : ', user_input)

        user_queries.append(user_input)
        with open('user_queries.pkl', 'wb') as file:
            pickle.dump(user_queries, file)
        start_time = time.time()

        input_len = len(user_input)
        print('input_len : ', input_len)
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        response_ids = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        generated_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print('response id : ', response_ids)
        end_time = time.time()

        response_time = end_time - start_time
        response_times.append(response_time)

        with open('response_time.pkl', 'wb') as file:
            pickle.dump(response_times, file)

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
        global chitchat_counter, novel_counter
        data = request.json
        print('chitchatclassifier data : ', data)
        user_input = data['user_input']
        user_queries.append(user_input)

        with open('user_queries.pkl', 'wb') as file:
            pickle.dump(user_queries, file)
        loaded_model = joblib.load('svm_model_new.pkl')

        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        user_input = preprocess_text(user_input)
        print('user_input : ', user_input)
        new_texts = [user_input, user_input]
        new_texts_tfidf = tfidf_vectorizer.transform(new_texts)
        predictions = loaded_model.predict(new_texts_tfidf)
        print('predictions : ', predictions)
        if predictions[0] == 1:
            print(' in if ')
            chitchat_counter += 1
            with open('chitchat_count.pkl', 'wb') as file:
                pickle.dump(chitchat_counter, file)
        else:
            print(' in else ')
            novel_counter += 1
            with open('novel_count.pkl', 'wb') as file:
                pickle.dump(novel_counter, file)
        print(' if ended ')
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
        novel_query_counter[book_name] += 1
        with open('novel_pie_count.pkl', 'wb') as file:
            pickle.dump(novel_query_counter, file)
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
        return jsonify({'novel': result["result"].strip(), 'name' : persistent_directory[19:]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/visualization', methods=['GET'])
def generate_visualization():
    try:
        # data = request.json
        # user_queries = data.get('user_queries', [])
        with open('user_queries.pkl', 'rb') as file:
            loaded_array = pickle.load(file)
        generate_pie_chart()
        generate_bar_graph(loaded_array)
        generate_doughnut_chart()
        line_chart()
        return jsonify({'message': 'Visualization generated successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
