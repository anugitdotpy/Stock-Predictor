# Stock Predictor System

[Link to usage video](https://www.loom.com/share/fabe67b9fe274c52b18c7d97236c98a7?sid=ddf418f0-018e-448a-a2b0-81e249a218f4)

This repository contains a multi-service stock prediction framework that includes the following components:

1. **Stock-Prediction-Process-management:**  
   A FastAPI-based service that handles news article processing, natural language processing (NLP) tasks, and analysis. It leverages several ML models and pipelines for tasks such as summarization, translation, sentiment analysis, and zero-shot classification.

2. **Stock-Predictor-Hindi-TTS-Management:**  
   A FastAPI service built on top of a Text-to-Speech (TTS) framework. It synthesizes Hindi speech from provided text using a pre-trained TTS model.

3. **Streamlit UI:**  
   A simple user interface (`streamlit.py`) to interact with the services. Use it to input queries, view results, and download synthesized audio.

---

## Repository Structure

```
Stock-Predictor/
├── README.md
├── streamlit.py             # Front-end UI for interacting with the services
├── Stock-Prediction-Process-management/   # Process management service (FastAPI)
│   ├── Dockerfile
│   ├── README.md
│   ├── requirements.txt
│   ├── run.py               # Entry point; starts the FastAPI server on port 5060
│   ├── routes/
│   │   ├── __init__.py
│   │   └── router.py
│   └── src/                 # Additional source code
└── Stock-Predictor-Hindi-TTS-Management/    # TTS service (FastAPI)
    ├── Dockerfile
    ├── README.md
    ├── requirements.txt
    └── run.py               # Entry point; starts the FastAPI server on port 5050
```

---

## How to Build and Run the Docker Containers

### 1. Stock-Prediction-Process-management Service

**Build the Docker image:**

```bash
cd Stock-Prediction-Process-management
docker build -t stock-prediction-process .
```

**Run the container:**

Map the container's port **5060** to a port on your host:

```bash
docker run -d -p 5060:5060 stock-prediction-process
```

This service initializes multiple NLP pipelines and background services (such as pulling an Ollama model) to process news articles and perform analysis.

---

### 2. Stock-Predictor-Hindi-TTS-Management Service

**Build the Docker image:**

```bash
cd Stock-Predictor-Hindi-TTS-Management
docker build -t stock-predictor-hindi-tts .
```

**Run the container:**

Map the container's port **5050** to a port on your host:

```bash
docker run -d -p 5050:5050 stock-predictor-hindi-tts
```

This service loads a TTS model (e.g., `tts_models/de/thorsten/tacotron2-DDC`) and provides an endpoint (`/synthesize`) to convert Hindi text to a WAV audio file.

---

## Running the Streamlit UI


The `streamlit.py` file in the root directory provides a user interface to interact with these services. To run the UI locally, ensure you have Streamlit installed, then execute:

```bash
pip install streamlit fpdf
streamlit run streamlit.py
```

This will start the Streamlit UI in your browser, allowing you to:
- Scrape and process articles (handled by the Process Management service).
- Synthesize speech from the processed Hindi text (handled by the TTS service).

The `streamlit.py` is currently configured to call API's of instances hosted in Huggingface Spaces, if you want it to point to your local instances, you can modify the variables *PROCESS_URL* and *TTS_URL* present inside the `streamlit.py`

*Note: The spaces in huggingface might be on sleep due to inactivity, so the frst API call might take a higher time to get processed.*

---

## Overview of the Models and Pipelines

### Stock-Prediction-Process-management Service

#### FastAPI Framework with Uvicorn
- The service is built using FastAPI and runs via Uvicorn.
- It includes middleware (e.g., GZip) for performance optimization.

#### NLP Pipelines and Models

- **Ollama & LangChain:**  
  Uses an Ollama model (`deepseek-r1`) to load a conversation model via LangChain. A prompt template is defined for comparison analysis.

- **Translation Pipeline:**  
  Utilizes the Helsinki-NLP/opus-mt-en-hi model to translate English text to Hindi.

- **Summarization Pipeline:**  
  Uses Facebook BART (`facebook/bart-large-cnn`) to generate summaries of articles.

- **Sentiment and Classification Models:**
  - **FinBERT:**  
    The FinBERT model (`ProsusAI/finbert`) is loaded for sequence classification to assess sentiment.
  - **Zero-shot Classification:**  
    A zero-shot classifier using Facebook’s `bart-large-mnli` to determine relevant topics from a predefined list.

#### Additional Components

- **NLTK Initialization:**  
  Downloads necessary NLTK data (e.g., `punkt`) for tokenization.

- **Background Services:**  
  Starts the Ollama service in a separate thread to ensure the conversation model is ready before processing requests.

---

### Stock-Predictor-Hindi-TTS-Management Service

#### TTS Model
- Loads a pre-trained TTS model (`tts_models/de/thorsten/tacotron2-DDC`) that converts Hindi text into speech.

#### Endpoint Implementation
- The `/synthesize` endpoint receives Hindi text and outputs a WAV file.
- It uses background tasks to clean up temporary files after serving the response.

---

## Final Notes

- **Pre-requisites:**  
  Ensure Docker is installed on your system to build and run the containers.

- **Port Configuration:**  
  Adjust port mappings if needed to avoid conflicts on your host machine.

- **Logs and Debugging:**  
  Use Docker logs (`docker logs <container-id>`) to inspect service output if any issues arise.

Enjoy using and extending the Stock Predictor System!
EOF
