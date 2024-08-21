![oreilly-logo](images/oreilly.png)

# Deploying GPT & Large Language Models

This repository contains code for the [O'Reilly Live Online Training for Deploying GPT & LLMs](https://learning.oreilly.com/live-events/deploying-gpt-and-large-language-models/0636920087375/0636920087374)

In this training, you learn how to use GPT-4, ChatGPT, OpenAI embeddings, and other large language models to build applications for both experimenting and production. We cover the fundamentals of GPT and its applications and explore alternative generative models such as Cohere and GPT-J. You gain practical experience in building a variety of applications with these models, including text generation, summarization, question answering, and more. Learn how to leverage prompt engineering, context stuffing, and few-shot learning to get the most out of GPT-like models. Then the focus shifts to deploying these models in production, including best practices and debugging techniques. By the end of the training, you have a working knowledge of GPT and other large language models, as well as the skills to start building your own applications with them.

### Notebooks

- **Fine-Tuning LLMs**

	- [Finetuning app_reviews with OpenAI](notebooks/fine_tuned_classification_sentiment.ipynb)

	- [Fine-tuning BERT for app_reviews](notebooks/BERT%20vs%20GPT.ipynb)

	- [Model Freezing with BERT](notebooks/anime_category_classification_model_freezing.ipynb)

- **Prompt Engineering**
	- [Introduction to Prompt Engineering](notebooks/intro_prompt_engineering.ipynb)

	- [Advanced to Prompt Engineering](notebooks/adv_prompt_engineering.ipynb)

- **RAG**

	- [Semantic Search](notebooks/semantic_search.ipynb)

	- [A basic RAG Bot using GPT and Pinecone](notebooks/rag_bot.ipynb)

**LLM Distillation**

- [Distilling BERT models to optimize for speed/memory](https://colab.research.google.com/drive/1GO8w1gC2TRII9-aaRNaFN6mkCglm2pJa?usp=sharing)

**LLM Quantization**

- [Quantizing Llama-3 dynamically](https://colab.research.google.com/drive/12RTnrcaXCeAqyGQNbWsrvcqKyOdr0NSm?usp=sharing)

- [Working with GGUF (no GPU)](https://colab.research.google.com/drive/15IC5cI-aFbpND5GrvKjAMas1Hmc7M6Rg?usp=sharing)

- [Working with GGUF (with a GPU)](https://colab.research.google.com/drive/1D6k-BeuF8YRTR8BGi2YYJrSOAZ6cYX8Y?usp=sharing)

- See [this directory](./llama_cpp) for a K8s demo of using embedding models and Llama 3 with GGUF on a GPU


## Instructor

**Sinan Ozdemir** is the Founder and CTO of LoopGenius where he uses State of the art AI to help people create and run their businesses. Sinan is a former lecturer of Data Science at Johns Hopkins University and the author of multiple textbooks on data science and machine learning. Additionally, he is the founder of the recently acquired Kylie.ai, an enterprise-grade conversational AI platform with RPA capabilities. He holds a master’s degree in Pure Mathematics from Johns Hopkins University and is based in San Francisco, CA.

