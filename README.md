# Sentiment Analysis Application Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture and Design](#architecture-and-design)
3. [Understanding the Hybrid Approach](#understanding-the-hybrid-approach)
4. [Code Structure and Components](#code-structure-and-components)
5. [How Sentiment Analysis Works](#how-sentiment-analysis-works)
6. [User Interface Design](#user-interface-design)
7. [Advanced Features and Edge Cases](#advanced-features-and-edge-cases)
8. [Installation and Usage](#installation-and-usage)
9. [Troubleshooting](#troubleshooting)
10. [Future Improvements](#future-improvements)

---

## Overview

This application is a **PyQt6-based sentiment analysis tool** designed for analyzing product reviews. It leverages a hybrid approach combining **HuggingFace Large Language Models (LLMs)** for high-accuracy core sentiment classification, and **NLTK's VADER** for aspect-based sentiment and keyword extraction. 

The tool now supports native, offline inference for both English (using `Qwen2.5-3B-Instruct`) and Hindi (using `twitter-xlm-roberta-base-sentiment`), natively handling complex nuances like sarcasm, negation, and context.

### Key Features

- **Offline Operation**: Works entirely offline after the initial model downloads.
- **State-of-the-Art LLMs**: Uses generative AI (`Qwen2.5-3B-Instruct`) for English and fine-tuned RoBERTa for Hindi.
- **Deep Contextual Understanding**: Naturally detects sarcasm, irony, and product-specific phrasing without rigid manual rules.
- **Multilingual Support**: Native inference for English and Hindi, with translation fallbacks for other languages.
- **Visual Feedback**: Real-time progress bars, emotion breakdown, and color-coded results via a PyQt6 interface.

---

## Architecture and Design

### System Architecture

```text
┌─────────────────────────────────────────┐
│            UI Layer (PyQt6)             │
│    ┌─────────────────────────────────┐  │
│    │  Input Text Area                │  │
│    │  LLM Download Prompts           │  │
│    │  Results Display & Progress     │  │
│    └─────────────────────────────────┘  │
├─────────────────────────────────────────┤
│         Sentiment Analyzer Class        │
│    ┌─────────────────────────────────┐  │
│    │  Language Detection/Fallback    │  │
│    │  HuggingFace Pipeline Router    │  │
│    │  VADER Aspect/Word Extraction   │  │
│    └─────────────────────────────────┘  │
├─────────────────────────────────────────┤
│             Inference Layer             │
│    ┌─────────────────────────────────┐  │
│    │ English: Qwen2.5-3B (Gen-LLM)   │  │
│    │ Hindi: XLM-RoBERTa (Classifier) │  │
│    │ Lexicon: NLTK VADER             │  │
│    └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## Understanding the Hybrid Approach

Previously, this application relied solely on rule-based NLP (VADER) and manual pattern matching for sarcasm. It now utilizes a **hybrid architecture**:

### 1. Generative LLM for English Sentiment
For English reviews, the app uses `Qwen/Qwen2.5-3B-Instruct`. By passing the review through a specific prompt (`"Analyze the sentiment... Consider sarcasm... Output ONLY valid JSON"`), the system bypasses rigid rules. The LLM naturally understands complex human phrasing, negations, and sarcasm, returning a structured JSON containing the sentiment and confidence score.

### 2. Fine-Tuned Transformer for Hindi
For Hindi, the app uses `cardiffnlp/twitter-xlm-roberta-base-sentiment`, a model specifically trained on multilingual text. This provides native sequence classification without relying on lossy translations.

### 3. VADER & Lexicons for Aspects/Emotions
While LLMs handle the core classification, NLTK's VADER and custom lexicons are still utilized for:
- **Aspect-Based Analysis**: Breaking down sentiment across "quality", "price", "performance".
- **Emotion Scoring**: Calculating normalized scores for Joy, Anger, Fear, and Sadness based on word frequencies.
- **Keyword Extraction**: Pulling positive and negative words from the text using VADER's lexicon.

---

## Code Structure and Components

### 1. SentimentAnalyzer Class (`sentiment_analyzer.py`)

Handles model loading, inference, and text processing.

- `__init__`: Initializes VADER and prepares placeholder pipelines for the HuggingFace models.
- `predict`: The main routing function. It checks the language, loads the appropriate HuggingFace model (if permitted), constructs the prompt (for English), parses the output, and aggregates aspect/emotion data.
- `analyze_aspects`: Uses sentence tokenization to find aspect-related keywords, scoring them using the Hindi model or VADER (for English).
- `analyze_emotions`: Uses dictionary-based keyword matching to gauge emotions.

### 2. UI Component (`sentiment_ui.py`)

Handles the user interface, utilizing PyQt6.

- Detects missing models and prompts the user to download them (e.g., the ~6GB English LLM).
- Displays color-coded results, confidence scores, aspect breakdown, and extracted keywords.

---

## How Sentiment Analysis Works

### Step-by-Step Execution (English)

1. **User Input**:
   `"Oh fantastic, it arrived broken. Just what I needed."`

2. **Model Loading & Prompt Construction**:
   The `SentimentAnalyzer` formats the input into a strict prompt:
   ```python
   prompt = f"Analyze the sentiment of the following product review. Consider sarcasm and context carefully. Review: '{processed_text}'. Output ONLY a valid JSON object..."
   ```

3. **LLM Inference**:
   The `Qwen2.5-3B-Instruct` model processes the prompt and outputs:
   ```json
   {
     "sentiment": "Negative",
     "confidence": 0.95
   }
   ```

4. **Parsing and UI Update**:
   The application cleans the output (removing markdown backticks), parses the JSON, updates the compound score, and updates the UI to reflect a highly confident Negative result.

---

## User Interface Design

- **Smart Model Prompts**: The UI actively checks for local HuggingFace models. If missing, it provides a styled `QMessageBox` explaining the download size and benefits before proceeding.
- **Color Coding**: Real-time coloring (Green for Positive, Red for Negative, Yellow for Neutral).
- **Progress Bars**: Visual representations of the compound sentiment and specific emotion intensities.

---

## Advanced Features and Edge Cases

### Sarcasm and Context
Because the system now relies on a 3-billion parameter generative model for English, hardcoded sarcasm rules (e.g., checking for "great but") are no longer necessary. The LLM's vast training data allows it to intuitively recognize sarcastic structures, idioms, and product-specific malfunction phrasing.

### Fallback Translation
If a language other than English or Hindi is selected, the application attempts to use `deep-translator` (or a similar translation module) to translate the text into English, which is then fed into the highly accurate English LLM.

---

## Installation and Usage

### Prerequisites

```bash
pip install PyQt6 nltk transformers torch accelerate
```
*(Note: `torch` and `accelerate` are required for running HuggingFace pipelines efficiently).*

### Running the Application

**GUI Version:**
```bash
python sentiment_ui.py
```

**Command Line Version:**
```bash
python sentiment_analyzer.py
```

### First Time Setup
1. **NLTK Resources**: VADER lexicons download automatically.
2. **LLM Downloads**: Upon first analysis of an English or Hindi text, the UI will prompt to download the respective model from HuggingFace. The English model is approximately 6GB and will be stored in your HuggingFace cache directory. Subsequent runs will be completely offline.

---

## Troubleshooting

### 1. Model Download Hangs or Fails
**Symptom**: App freezes during model download or throws `Out of Memory` errors.
**Solution**: Ensure you have a stable internet connection for the initial 6GB download. If memory issues occur, ensure your machine has at least 8GB of RAM.

### 2. JSON Parsing Errors from LLM
**Symptom**: The analyzer defaults to "Neutral" with 0.0 confidence despite strong wording.
**Solution**: The LLM may have generated output outside the strict JSON format. Check the console output (`Failed to parse LLM output: ...`) to see the raw generation. 

### 3. Missing `transformers` or `torch`
**Symptom**: `ImportError` on startup.
**Solution**: Run `pip install transformers torch accelerate`.

---

## Future Improvements

1. **Quantization**: Implement 4-bit or 8-bit quantization for the English model to reduce the memory footprint from ~6GB to ~2GB.
2. **LoRA Fine-tuning**: Fine-tune the Qwen model specifically on product reviews to guarantee JSON output structure without prompting hacks.
3. **Expand Native Languages**: Add native HuggingFace pipelines for Spanish, French, etc., reducing reliance on the translation fallback.
