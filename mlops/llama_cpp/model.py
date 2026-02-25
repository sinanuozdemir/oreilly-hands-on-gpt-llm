# import torch
from flask import Flask, request, jsonify, Response
from llama_cpp import Llama

# Create a Flask object
app = Flask("Llama server")

try:
    llm = Llama.from_pretrained(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="*Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        verbose=False,
        n_gpu_layers=-1
    )
except Exception as e:
    print(f"Error: {str(e)}")
    llm = None

qt_embedding_model = Llama.from_pretrained(  # roughly 25MB
    "second-state/All-MiniLM-L6-v2-Embedding-GGUF",
    filename="all-MiniLM-L6-v2-Q6_K.gguf",
    n_gpu_layers=-1, verbose=False, embedding=True
)


def stream_generate_response(system, user):
    try:
        for token in llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                stream=True
        ):
            content = token["choices"][0]["delta"].get("content", "")
            yield f"{content}"
    except Exception as e:
        yield f"Error: {str(e)}"


@app.route('/llama', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
        system = data.get("system", "")
        user = data.get("user", "")
        return Response(stream_generate_response(system, user), mimetype='text/plain')

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

@app.route('/embedding', methods=['POST'])
def generate_embedding():
    try:
        data = request.get_json()
        texts = data.get("texts", [])
        return jsonify(qt_embedding_model.embed(texts))

    except Exception as e:
        return jsonify({"Error": str(e)}), 500


# health check
@app.route('/', methods=['GET'])
def health_check():
    try:
        import torch
        device = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    except:
        device = 'NO TORCH'
    # check if cuda is available and get GPU type

    return jsonify({"status": "ok", "device": device})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False)
