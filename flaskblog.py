from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Load pre-trained DialoGPT model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.json
        print('data : ', data)
        user_input = data['user_input']
        print('user_input : ', user_input)

        # Tokenize and generate response
        input_len = len(user_input)
        print('input_len : ', input_len)
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        response_ids = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        generated_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print('response id : ', response_ids)

        return jsonify({'generated_rerereresponse': generated_response[input_len:]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
