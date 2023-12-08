from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)


model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")


conversation_history = []

@app.route('/chat', methods=['POST'])
def generate_response():
    try:
        global conversation_history

        data = request.json
        user_input = data['user_input']


        conversation_history.append(f"User: {user_input}")


        input_ids = tokenizer.encode('\n'.join(conversation_history) + tokenizer.eos_token, return_tensors='pt')
        response_ids = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        generated_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)


        conversation_history.append(f"Chatgenix: {generated_response}")


        ai_response = conversation_history[-1].split('AI: ')[1]

        return jsonify({'generated_response': ai_response})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
