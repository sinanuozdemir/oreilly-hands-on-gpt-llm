![oreilly-logo](images/oreilly.png)

# Deploying GPT & Large Language Models

This repository contains code for the [O'Reilly Live Online Training for Deploying GPT & LLMs](https://www.oreilly.com/live-events/deploying-gpt-large-language-models-llms/0642572012963)

This course is designed to equip software engineers, data scientists, and machine learning professionals with the skills and knowledge needed to deploy AI models effectively in production environments. As AI continues to revolutionize industries, the ability to deploy, manage, and optimize AI applications at scale is becoming increasingly crucial. This course covers the full spectrum of deployment considerations, from leveraging cutting-edge tools like Kubernetes, llama.cpp, and GGUF, to mastering cost management, compute optimization, and model quantization.

## Base Notebooks

**Introduction to LLMs and Prompting**

- [Introduction to 3rd Party Providers](notebooks/third_party_inference.ipynb) - Using Together.ai, HuggingFace, and Groq to run LLMs

- [Prompt Injection Examples](notebooks/prompt_injection.ipynb) - See how three kinds of prompt innjection attacks can attempt to jailbreak an LLM

**Cleaning Data and Monitoring Drift**

- [Cleaning Data using Deep Learning](https://colab.research.google.com/drive/1hPnU9sLsV9W50q9rd_oxUU1Bv7SUCVU5?usp=sharing) - Using AUM and Cosine Similarity to clean data

- [Combatting AI drift](https://colab.research.google.com/drive/14E6DMP_RGctUPqjI6VMa8EFlggXR7fat?usp=sharing) - Using Online Learning to combat drift

**Evaluating Agents**

- [Evaluating AI Agents: Task Automation and Tool Integration](https://ai-office-hours.beehiiv.com/p/evaluating-ai-agent-tool-selection) - A basic case study on tool selection accuracy
	- [Positional Bias on Agent Response Evaluation](https://github.com/sinanuozdemir/oreilly-ai-agents/blob/main/notebooks/Evaluating_LLMs_with_Rubrics.ipynb) - Identifying and evaluating positional bias on multiple LLMs

**Advanced Deployment Techniques**

- [Speculative Decoding](https://colab.research.google.com/drive/1QXqUjgMLUbAqXzGc8uBWJ5t4BEtJQbWh?usp=sharing) - Using an assistant model to aid token decoding

- [Prompt Caching Llama 3](https://colab.research.google.com/drive/1LlocxmN6adI-bFeT2dGGa4U2zkku77o7?usp=sharing) - Replicating prompt caching with HuggingFace tools

- [Distilling BERT](https://colab.research.google.com/drive/1GO8w1gC2TRII9-aaRNaFN6mkCglm2pJa?usp=sharing) - Distilling models to optimize for speed/memory

- [Quantizing Llama-3 dynamically](https://colab.research.google.com/drive/12RTnrcaXCeAqyGQNbWsrvcqKyOdr0NSm?usp=sharing) - Using bitsandbytes to quantize nearly any LLM on HuggingFace

- [Working with GGUF (no GPU)](https://colab.research.google.com/drive/15IC5cI-aFbpND5GrvKjAMas1Hmc7M6Rg?usp=sharing) - Using Llama.cpp to work with models

- [Working with GGUF (with a GPU)](https://colab.research.google.com/drive/1D6k-BeuF8YRTR8BGi2YYJrSOAZ6cYX8Y?usp=sharing) - Using Llama.cpp to work with models

- [DeepSeek Model on GGUF](https://colab.research.google.com/drive/1dHx_t_BSfqANBECcHXm2atpvqwxWmE3k?usp=sharing) - Running a DeepSeek Distilled Llama model using Llama.cpp

- See [this directory](./llama_cpp) for a K8s demo of using embedding models and Llama 3 with GGUF on a GPU


## More

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

## Instructor

**Sinan Ozdemir** is the Founder and CTO of LoopGenius where he uses State of the art AI to help people create and run their businesses. Sinan is a former lecturer of Data Science at Johns Hopkins University and the author of multiple textbooks on data science and machine learning. Additionally, he is the founder of the recently acquired Kylie.ai, an enterprise-grade conversational AI platform with RPA capabilities. He holds a master’s degree in Pure Mathematics from Johns Hopkins University and is based in San Francisco, CA.

