import threading
import subprocess
import time

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from routes import routes_router


def init_app() -> FastAPI:
    """Initialize the FastAPI application, add routes and middleware."""
    app = FastAPI()
    app.include_router(routes_router)
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    return app


def init_nltk(app: FastAPI):
    """Download required NLTK packages to a custom directory."""
    import os
    import nltk

    # Use NLTK_DATA from the environment if available; otherwise, fall back to ./nltk_data
    nltk_data_path = os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data"))
    os.makedirs(nltk_data_path, exist_ok=True)
    
    # Optionally, add the path to nltk's data search paths
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
    
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    app.state.NLTK = nltk


def start_ollama_service():
    """Start the Ollama service in a separate process."""
    subprocess.Popen(["ollama", "serve"])


def pull_ollama_model():
    """
    Pull the deepseek-r1 model using Ollama.
    This ensures that the model is cached and ready before it's used.
    """
    result = subprocess.run(["ollama", "pull", "deepseek-r1"], check=True, capture_output=True, text=True)
    print("Ollama pull output:", result.stdout)


def start_background_services():
    """Start background services, including Ollama."""
    thread = threading.Thread(target=start_ollama_service, daemon=True)
    thread.start()
    # Wait a few seconds to ensure Ollama service is up
    time.sleep(5)


def init_models(app: FastAPI):
    """Load and attach all models and pipelines to the app state."""
    # Conversation model using Ollama and LangChain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM

    # Now that deepseek-r1 is pulled, load the model
    ollama_model = OllamaLLM(model="deepseek-r1")
    template = (
        "Question: {question}\n"
        "Answer:The comparison analysis of these articles are"
    )
    prompt = ChatPromptTemplate.from_template(template)
    app.state.CONVERSER = prompt | ollama_model

    # Translator model (English to Hindi) using Helsinki-NLP
    from transformers import AutoModelForSeq2SeqLM, pipeline, AutoModelForSequenceClassification, AutoTokenizer

    translator_model_name = "Helsinki-NLP/opus-mt-en-hi"
    app.state.TRANSLATOR_TOKENIZER = AutoTokenizer.from_pretrained(translator_model_name)
    app.state.TRANSLATOR = AutoModelForSeq2SeqLM.from_pretrained(translator_model_name)
    # Summarizer pipeline using Facebook BART
    app.state.SUMMARIZER = pipeline("summarization", model="facebook/bart-large-cnn")

    # FinBERT for sequence classification
    finbert_model_name = "ProsusAI/finbert"
    app.state.FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(finbert_model_name)
    model_finbert = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
    model_finbert.to("cuda")
    app.state.FINBERT = model_finbert

    # Zero-shot classification pipeline using Facebook BART MNLI
    app.state.ZSCLASSIFIER = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli"
    )

    # Define a list of topics
    app.state.TOPICS = [
        "financial performance",
        "new product release",
        "mergers and acquisitions",
        "legal or regulatory issues",
        "union or labor issues",
        "executive leadership changes",
        "market competition or strategy",
        "technology and innovation",
        "partnerships or collaborations",
        "data privacy or security",
        "customer satisfaction or feedback",
        "environmental or sustainability",
        "corporate social responsibility",
        "research and development",
        "investments or funding",
        "bankruptcy or restructuring",
        "scandals or controversies",
        "stock price movement",
        "marketing or advertising",
        "product recall or safety concerns",
        "infrastructure or supply chain issues",
        "economic impact",
    ]
    app.state.LABEL_MAP = {
        0: "positive",
        1: "negative",
        2: "neutral"
    }


def main() -> FastAPI:
    """Main initialization function."""
    app = init_app()
    init_nltk(app)

    # Start background services, including the Ollama service
    start_background_services()

    # Pull the deepseek-r1 model before loading it into the conversation model
    pull_ollama_model()

    # Now initialize all ML models and pipelines
    init_models(app)
    return app


# Initialize the FastAPI app
app = main()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("run:app", host="0.0.0.0", port=5060, workers=1)
