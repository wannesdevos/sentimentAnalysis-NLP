# Sentiment Analysis - NLP

A sentiment analysis project that benchmarks multiple pre-trained models on the IMDB movie reviews dataset and provides a REST API for sentiment prediction.

## Overview

This project implements sentiment analysis using two pre-trained transformer models:
- **Model 1**: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- **Model 2**: RoBERTa Large (siebert/sentiment-roberta-large-english)

The project includes benchmarking capabilities and a FastAPI endpoint for real-time predictions.

## Setup

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd sentimentAnalysis-NLP
```

2. Install dependencies:
```bash
uv sync
```

This will create a virtual environment and install all required packages.

## Project Structure

```
.
├── data/                    # IMDB movie reviews dataset
├── output/                  # Benchmark results and predictions
├── src/
│   ├── models.py                   # Model definitions and prediction functions
│   ├── benchmark.py                # Benchmarking script
│   ├── sentiment_analysis.py       # FastAPI application
│   └── sentiment_analysis_gradio.py # Gradio web interface
├── pyproject.toml                  # Project dependencies
└── README.md
```

## Usage

### Running Benchmarks

To benchmark both models on the IMDB dataset:

```bash
uv run src/benchmark.py
```

This will:
- Process all reviews in the dataset
- Calculate accuracy, precision, recall, and F1-score
- Measure inference time
- Save results to `output/` directory
- Generate a comparison report

### Running the API

Start the FastAPI server:

```bash
uv run fastapi dev src/sentiment_analysis_fastapi.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**Model 1 - DistilBERT:**
```bash
POST http://localhost:8000/sentiment-analysis/model-1
Content-Type: application/json

{
  "text": "This movie was absolutely fantastic!"
}
```

**Model 2 - RoBERTa Large:**
```bash
POST http://localhost:8000/sentiment-analysis/model-2
Content-Type: application/json

{
  "text": "This movie was absolutely fantastic!"
}
```

**Response:**
```json
{
  "label": "positive",
  "score": 0.9998
}
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

### Running the Gradio Demo

For a user-friendly web interface to test the models:

```bash
uv run src/sentiment_analysis_gradio.py
```

This will launch a Gradio interface in your browser where you can:
- Enter text to analyze sentiment
- Choose between Model 1 (DistilBERT) or Model 2 (RoBERTa)
- See the predicted sentiment and confidence score
- Try pre-loaded examples

The interface will automatically open at `http://localhost:7860`

## Models

### Model 1: DistilBERT
- Lightweight and fast
- Good for production environments with limited resources
- Lower accuracy but faster inference

### Model 2: RoBERTa Large
- Higher accuracy
- Slower inference time
- Better for scenarios where accuracy is critical

## Benchmark Results

Benchmark results are saved in the `output/` directory:
- `benchmark_comparison.csv` - Side-by-side model comparison
- Individual result files for each model with predictions

## Dependencies

Key dependencies:
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch framework
- `fastapi` - Web framework for API
- `gradio` - Interactive web UI for demos
- `scikit-learn` - Metrics and evaluation
- `pandas` - Data processing

See `pyproject.toml` for complete list.

## Development

This project uses:
- `uv` for dependency management
- FastAPI for the REST API
- Pre-trained Hugging Face models
- Git for version control
