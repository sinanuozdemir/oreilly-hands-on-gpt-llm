
# Llama 3 / Embedding Flask App with GPU

A simple and efficient Flask application that serves a language model and embedding model using the `llama_cpp` library. This server provides endpoints to generate text completions and text embeddings through HTTP requests.

---

## Features

- **Text Completion**: Generate text responses based on a system prompt and user input using the `Meta-Llama-3.1-8B-Instruct` model.
- **Text Embeddings**: Generate text embeddings using the `All-MiniLM-L6-v2` model.
- **Streaming Responses**: Supports streaming of generated text for efficient and responsive applications.
- **Health Check Endpoint**: Easily verify the status of the server and the device it’s running on.

---

## Building and Running with Docker

To simplify deployment and ensure consistency across different environments, you can build and run the Flask application using Docker. Below are the steps to build the Docker image and run the container.

### Dockerfile Overview

The Dockerfile provided uses the CUDA 12.2 base image to support GPU-accelerated inference with the Llama model. It includes steps to install necessary dependencies, build the `llama.cpp` library, and set up the Flask application.

### Building the Docker Image

1. **Ensure Docker is installed** on your machine. If not, you can download it from the [Docker website](https://www.docker.com/get-started).

2. **Navigate to the directory** where your `Dockerfile` and `model.py` are located.

3. **Build the Docker image** using the following command:

   ```bash
   docker build -t llama-flask-app .
   ```

   This command creates a Docker image named `llama-flask-app` based on the instructions in your `Dockerfile`. It might take about 10-15 minutes.

### Running the Docker Container

Once the image is built, you can run the container using the following command:

```bash
docker run --gpus all -p 5005:5005 llama-flask-app
```

- **`--gpus all`**: Ensures the container has access to all available GPUs.
- **`-p 5005:5005`**: Maps port 5005 on your local machine to port 5005 in the container, making the Flask app accessible via `http://localhost:5005`.

### Accessing the Application

After running the container, you can access the Flask application at `http://localhost:5005`.

- **API Endpoints**: Use the `/llama`, `/embedding`, and `/` endpoints as described earlier in the README.
- **Logs**: Docker will output logs to your terminal, allowing you to monitor the application's activity.

### Dockerfile Summary

Here’s a brief explanation of the key steps in the `Dockerfile`:

- **Base Image**: `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04` for CUDA support.
- **Dependencies**: Installs Python, pip, git, and build tools.
- **Llama.cpp**: Clones and builds the `llama.cpp` library.
- **Flask Setup**: Copies `model.py`, installs Python packages, and sets up the environment for GPU support.
- **Expose Port**: Exposes port 5005 for the Flask application.
- **Run Command**: The container runs `model.py` when started.

By using Docker, you ensure that your application runs consistently across different environments, with all dependencies and configurations encapsulated within the container.


## API Endpoints

### /llama

- **Method**: POST
- **Description**: Generates text completion based on the provided system and user prompts.
- **Payload**:
  ```json
  {
    "system": "string",
    "user": "string"
  }
  ```
- **Response**: Streamed text completion.

### /embedding

- **Method**: POST
- **Description**: Generates text embeddings for the provided list of texts.
- **Payload**:
  ```json
  {
    "texts": ["string1", "string2", ...]
  }
  ```
- **Response**: JSON array of embeddings.

### / (Health Check)

- **Method**: GET
- **Description**: Checks the health status of the server.
- **Response**: JSON status of the server and the device it's running on.

---

## Example Requests

### Generate Text Completion

```bash
curl -X POST http://localhost:5005/llama      -H "Content-Type: application/json"      -d '{"system": "You are a helpful assistant.", "user": "Tell me about Mount Everest in 5 words."}'
```

### Generate Embeddings

```bash
curl -X POST http://localhost:5005/embedding      -H "Content-Type: application/json"      -d '{"texts": ["You are a helpful assistant.", "Tell me about Mount Everest in 5 words."]}'
```

### Health Check Request

```bash
curl http://localhost:5005/
```

---

## Configuration

You can customize the Flask application and models by modifying the `app.py` file according to your needs.

---

## Deploying with Kubernetes
To deploy this Flask application using Kubernetes, follow these steps:

### 1. Prepare Your Kubernetes Cluster

Ensure you have a Kubernetes cluster up and running. You can use GKE, EKS, AKS, or Minikube for local testing.

### 2. Apply the Kubernetes YAML Configuration

The llama-k8s.yaml file contains the necessary configuration for deploying the application to a Kubernetes cluster.

Navigate to the directory where your llama-k8s.yaml file is located.

Apply the configuration using the following command:

```bash
kubectl apply -f llama-k8s.yaml
```

This command will create the necessary Kubernetes resources, including a Deployment and a Service.

### 3. Access the Application

Once the deployment is successful, you can access the application using the service's external IP or through an Ingress controller if configured.

Get the service details:

```bash
kubectl get services
Look for the external IP assigned to your service.
```

Access the application by navigating to http://<<-external-ip>>.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for the pre-trained models.
- [Llama-Cpp](https://github.com/ggerganov/llama.cpp) for the lightweight language model inference.