# Use the CUDA 12.2 base image
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies including the CUDA toolkit
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

# Build the Llama.cpp library with correct CUDA arch (7.5 for T4)
RUN cd /app/llama.cpp && mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && make

# Copy your application code into the container
COPY ./model.py /app/model.py

# Set the C and C++ compiler environment variables
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

# Set build-related environment variables
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1

# Upgrade pip, setuptools, and wheel to the latest versions
RUN pip install --upgrade pip setuptools wheel Flask huggingface-hub

# Install llama-cpp-python with GPU support and correct CUDA arch flags
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" python -m pip install llama-cpp-python --verbose

# Expose port 5005
EXPOSE 5005

# Run model.py when the container launches
CMD ["python", "/app/model.py"]
