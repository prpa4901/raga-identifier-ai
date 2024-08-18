from flask import Flask, request, jsonify
# from transformers import LlamaForCausalLM, LlamaTokenizer
# import torch
import requests

app = Flask(__name__)

'''
# Load the LLaMA model and tokenizer (replace 'model_name' with the actual model)
model_name = "huggingface/llama"  # Replace with the path to your LLaMA model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
'''

def llama_predict(prompt):
    # Tokenize the prompt
    '''
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate a response using the LLaMA model
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=100)
    
    # Decode the generated tokens into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    '''
    import json
    response = requests.post('http://ollama-service:11434/api/generate',
                             headers={"Content-Type": "application/json"},
                             data=json.dumps({"model": "llama3","prompt":prompt,"stream": False}))

    return response.json()['response']



@app.route('/api/identify-raga', methods=['POST'])
def identify_raga():
    # Check if an audio file is included in the request
    if request.json.get('file'):
        prompt = request.json.get('prompt', 'Identify the raga')

        # Save the audio file to the shared volume
        # audio_file_path = "/shared-data/output.wav"
        # audio_file.save(audio_file_path)

        # Step 1: Call the audio processing service to extract notes
        notes_response = requests.post("http://recording-analyzer:5001/api/process")
        notes = notes_response.json().get("notes", [])
        print(notes)

        # Step 2: Use the LLaMA model to identify the raga based on the notes
        raga_name = llama_predict(f"{prompt} Notes: {notes}")

        # Return the identified raga as a JSON response
        return jsonify({"raga": raga_name})
    
    else:
        # Handle text-only conversation
        prompt = request.form.get('prompt', 'Ask me anything...')
        response = llama_predict(prompt)
        return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000)
