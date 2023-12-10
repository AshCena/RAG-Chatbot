from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import json

# Load and preprocess chit-chat dataset
with open('dataset.json', 'r') as file:
    chit_chat_data = json.load(file)
    
# Load and preprocess novel dataset
chit_chat_texts = []
for key, value in chit_chat_data.items():
    messages = value.get("messages", [])
    for message_list in messages:
        # print(message_list)
        for message in message_list:
            text = message.get("text", "")
            # print(text)
            chit_chat_texts.append(text)

novel_texts = []
for i in range(1, 11):
    with open(f'novel_1 ({i}).txt', 'r', encoding='utf-8') as file:
        novel_text = file.read()
        # Split novel text into paragraphs (you may need to adjust this based on the actual structure)
        paragraphs = novel_text.split('\n\n')  # Adjust the delimiter based on your text structure
        novel_texts.extend(paragraphs)

# Create labels for the datasets (1 for chit-chat, 0 for novel)

labels_chit_chat = [1] * len(chit_chat_texts)
labels_novel = [0] * len(novel_texts)
print('labels_chit_chat : ', len(labels_chit_chat))
print('labels_novel : ', len(labels_novel))
# Combine datasets
all_texts = chit_chat_texts + novel_texts
all_labels = labels_chit_chat + labels_novel

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
