# Use the CUDA 12.2 base image
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install your application dependencies here, including the CUDA toolkit
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3 \
    git \
    build-essential \
    cmake && \
    ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory in the container
WORKDIR /app

# Clone the Llama.cpp repository
RUN git clone https://github.com/ggerganov/llama.cpp.git /app/llama.cpp

# Build the Llama.cpp library
RUN cd /app/llama.cpp && mkdir build && cd build && cmake .. && make

# Copy the necessary files into the container at /app
COPY ./model.py /app/model.py

# Set the C and C++ compiler environment variables
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
# Setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1

# Upgrade pip, setuptools, and wheel to the latest versions
RUN pip install --upgrade pip setuptools wheel Flask huggingface-hub

RUN CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install llama-cpp-python --verbose

# Expose port 5005 outside of the container
EXPOSE 5005

# Run model.py with Gunicorn when the container launches
CMD ["python", "/app/model.py"]
