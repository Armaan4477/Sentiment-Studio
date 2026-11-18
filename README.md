# Sentiment Analysis Studio & Multilingual NLP Toolkit

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![Hugging Face Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP%20Lexicon-154f5b)](https://www.nltk.org/)
[![Meta NLLB-200](https://img.shields.io/badge/Meta-NLLB--200-0668E1?logo=meta&logoColor=white)](https://huggingface.co/facebook/nllb-200-distilled-600M)

A modern, production-grade Natural Language Processing (NLP) suite designed for high-accuracy sentiment analysis, aspect extraction, emotion classification, and multilingual translation of customer product reviews. 

The application pairs state-of-the-art Generative Large Language Models (**Qwen 2.5-3B-Instruct**) and Multilingual Sequence Transformers (**CardiffNLP Twitter XLM-RoBERTa**) with lexical heuristics (**NLTK VADER**) and Neural Machine Translation (**Meta NLLB-200**), wrapped in an asynchronous, modern **PyQt6 desktop studio**.

---

## Table of Contents

1. [Overview & Highlights](#overview--highlights)
2. [Key Capabilities](#key-capabilities)
3. [System Architecture](#system-architecture)
4. [AI & NLP Models](#ai--nlp-models)
   - [English Engine: Qwen 2.5-3B-Instruct LLM](#1-english-sentiment-engine-qwen-25-3b-instruct-llm)
   - [Hindi Engine: CardiffNLP Twitter XLM-RoBERTa](#2-hindi-sentiment-engine-cardiffnlp-twitter-xlm-roberta)
   - [Multilingual Translation Engine: Meta NLLB-200](#3-multilingual-translation-engine-meta-nllb-200)
   - [Aspect-Based Sentiment Analysis (ABSA)](#4-aspect-based-sentiment-analysis-absa)
   - [Multi-Class Emotion Detection](#5-multi-class-emotion-detection)
   - [Explainability & Keyword Extraction](#6-explainability--keyword-extraction)
   - [Lexical & Tokenization Baseline: NLTK VADER](#7-lexical--tokenization-baseline-nltk-vader)
5. [Contextual Sarcasm Detection](#contextual-sarcasm-detection)
6. [Desktop Application: Sentiment Analysis Studio](#desktop-application-sentiment-analysis-studio)
7. [Repository Structure](#repository-structure)
8. [Installation & Setup](#installation--setup)
9. [Running the Application](#running-the-application)
   - [1. PyQt6 Studio GUI](#1-pyqt6-sentiment-analysis-studio)
   - [2. Interactive & Batch CLI](#2-command-line-interface-cli)
   - [3. Standalone Multilingual Translator GUI](#3-standalone-multilingual-translator-gui)
10. [Programmatic Python API](#programmatic-python-api)
11. [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Overview & Highlights

Modern customer feedback rarely adheres to simple polarity rules. Customers frequently express complex, mixed opinions, subtle sarcasm (*"Oh fantastic, it arrived completely broken. Just what I needed."*), and localized expressions across languages. 

Traditional lexicon-based tools (such as standalone VADER) misclassify sarcastic or nuanced reviews because they rely on bag-of-words keyword lookup without semantic context. 

This project solves this challenge by implementing a **hybrid multi-tier architecture**:

- **Reasoning-Driven Sarcasm Detection**: Leverages **Qwen 2.5-3B-Instruct** with 16-bit half-precision weights (`torch.bfloat16`) to deeply analyze reviewer pragmatics, irony, negation scope, and sentiment confidence.
- **Native Hindi Deep Learning**: Implements **CardiffNLP XLM-RoBERTa** natively trained on multilingual social sentiment to analyze Devanagari text directly without loss in translation.
- **Universal Translation Fallback**: Integrates **Meta's NLLB-200** model (200+ languages, 600M distilled parameters) to dynamically translate non-native inputs before feeding them to downstream engines.
- **Granular Aspect-Based Scoring**: Evaluates sentiment across 6 distinct commercial dimensions (Quality, Performance, Price, Design, Customer Service, Ease of Use).
- **Emotion Profiling**: Measures Joy, Anger, Fear, and Sadness with normalized intensity percentages.
- **Asynchronous Desktop GUI**: Engineered with PyQt6, featuring non-blocking background workers (`QThread`), card-based responsive design, drop shadows, and automatic on-demand model weight downloading.

---

## Key Capabilities

| Feature | Implementation | Description |
| :--- | :--- | :--- |
| **English Sentiment & Sarcasm** | `Qwen/Qwen2.5-3B-Instruct` | 3B-parameter Causal LLM producing structured JSON (`sentiment`, `confidence`) with contextual sarcasm reasoning |
| **Native Hindi Sentiment** | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 3-class Multilingual RoBERTa Sequence Classifier trained on social media text |
| **Multilingual NMT** | `facebook/nllb-200-distilled-600M` | Seq2Seq translation across 11+ languages with caching and standalone UI |
| **Aspect-Based Analysis (ABSA)** | Sentence Tokenization + Keyword Routing | Detects and scores 6 product aspects: Quality, Performance, Price, Design, Support, Ease of Use |
| **Emotion Analysis** | Multi-category Lexical Normalizer | Quantifies Joy, Anger, Fear, and Sadness on a 0–100% normalized scale |
| **Explainable Keywords** | VADER Lexicon + Hindi Sentiment Lexicon | Highlights specific positive and negative words driving classification |
| **Desktop UI Studio** | PyQt6 (`QGraphicsDropShadow`, `QScrollArea`, Fusion) | Slate-themed GUI with dynamic meters, cards, color-coded badges, and thread isolation |
| **Offline Capability** | Pre-bundled NLTK assets + Local Hugging Face cache | Works completely offline once model weights are stored in local cache |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               PyQt6 Presentation Layer (Sentiment Analysis Studio)     │
│  ┌────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │ Review Input & Language│  │ Primary Sentiment & Compound Gauge   │  │
│  │ Dropdown (EN / HI)     │  │ Aspects Breakdown (Progress Bars)    │  │
│  │ "Analyze Sentiment"    │  │ Emotion Profiling (Joy/Anger/Fear...)│  │
│  │ Threaded Dispatcher    │  │ Explainability / Keyword Tags        │  │
│  └───────────┬────────────┘  └──────────────────▲───────────────────┘  │
└──────────────┼──────────────────────────────────┼──────────────────────┘
               │ (Non-blocking QThread)           │ (Analysis Signals)
┌──────────────▼──────────────────────────────────┴──────────────────────┐
│                    SentimentAnalyzer Engine Core                       │
│                                                                        │
│   Language Router & Preprocessing (Whitespace, Tokenization, Case)     │
│        │                                     │                         │
│        ├──────────────────────┐              │                         │
│        ▼                      ▼              ▼                         │
│   [English Track]       [Hindi Track]  [Fallback NMT Track]            │
│        │                      │              │                         │
│        │                      │         Meta NLLB-200                  │
│        │                      │         Translation                    │
│        │                      │              └────────────┐            │
│        ▼                      ▼                           ▼            │
│  Qwen 2.5-3B-Instruct   CardiffNLP RoBERTa       Qwen 2.5-3B Engine    │
│  Generative LLM         Multilingual CLS         (Translated Text)     │
│  (Structured JSON)      (Softmax Scores)                               │
│        │                      │                           │            │
│        └──────────────────────┼───────────────────────────┘            │
│                               ▼                                        │
│          Aspect-Based Sentiment Analysis (ABSA) Engine                 │
│          Multi-Class Emotion Detection Engine                          │
│          Lexicon Explainability & Keyword Extraction                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## AI & NLP Models

### 1. English Sentiment Engine: Qwen 2.5-3B-Instruct LLM
- **Hugging Face ID**: `Qwen/Qwen2.5-3B-Instruct`
- **Architecture**: Causal autoregressive transformer with Grouped Query Attention (GQA) and SwiGLU activations.
- **Precision**: `torch.bfloat16` with dynamic device mapping (`device_map="auto"` for Apple Silicon MPS, NVIDIA CUDA, or optimized multi-core CPU).
- **Inference Mode**: Runs zero-shot classification guided by a strict system persona:
  ```json
  {"role": "system", "content": "You are a highly accurate sentiment analysis API. You only respond in valid JSON format."}
  ```
- **Prompt Formulation**:
  > *"Analyze the sentiment of the following product review. Consider sarcasm and context carefully. Review: '{processed_text}'. Output ONLY a valid JSON object with two keys: 'sentiment' (must be exactly Positive, Negative, or Neutral) and 'confidence' (a float between 0.0 and 1.0)."*
- **Output Parsing**: Strips formatting tags, validates JSON keys, maps polarity to compound score intervals, and derives confidence bounds.

### 2. Hindi Sentiment Engine: CardiffNLP Twitter XLM-RoBERTa
- **Hugging Face ID**: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- **Architecture**: Cross-lingual RoBERTa (XLM-R) fine-tuned on sentiment-labeled social datasets.
- **Classes**: `negative` (mapped to id 0), `neutral` (mapped to id 1), `positive` (mapped to id 2).
- **Execution**: Evaluates Devanagari text directly without syntactic loss or intermediate translation bottlenecks.

### 3. Multilingual Translation Engine: Meta NLLB-200
- **Hugging Face ID**: `facebook/nllb-200-distilled-600M`
- **Architecture**: Sequence-to-Sequence (Seq2Seq) encoder-decoder trained on Meta's No Language Left Behind dataset.
- **Languages Supported**: Over 200 languages (including English `eng_Latn`, Hindi `hin_Deva`, Marathi `mar_Deva`, German `deu_Latn`, French `fra_Latn`, Spanish `spa_Latn`, Russian `rus_Cyrl`, Chinese `zho_Hans`, Arabic `arb_Arab`, Portuguese `por_Latn`, Japanese `jpn_Jpan`).
- **Integration**:
  - Implemented in `translation/translator.py` with caching (`_translation_cache`).
  - Provides seamless fallback when users test foreign reviews without local native transformer pipelines.
  - Can be run as a standalone multi-language desktop translator.

### 4. Aspect-Based Sentiment Analysis (ABSA)
Rather than producing a single blanket score, the engine segments the review using sentence boundary detection (`nltk.sent_tokenize`) and maps clauses to 6 core commercial dimensions:

1. **Quality** (`build`, `material`, `sturdy`, `cheap`, `durable`, `broke`, `गुणवत्ता`, `क्वालिटी`, `मजबूत`, `खराब`)
2. **Performance** (`performance`, `speed`, `fast`, `slow`, `battery`, `lag`, `काम`, `धीमा`, `तेज`, `परफॉरमेंस`)
3. **Price** (`price`, `cost`, `expensive`, `cheap`, `value`, `money`, `worth`, `कीमत`, `महंगा`, `सस्ता`)
4. **Design** (`design`, `look`, `color`, `ugly`, `beautiful`, `style`, `डिज़ाइन`, `सुंदर`, `दिखावट`)
5. **Customer Service** (`service`, `support`, `help`, `agent`, `rude`, `friendly`, `सर्विस`, `मदद`)
6. **Ease of Use** (`easy`, `hard`, `difficult`, `simple`, `intuitive`, `setup`, `उपयोग`, `आसान`, `कठिन`)

Each detected aspect clause is scored individually to indicate whether that specific product feature delighted or frustrated the user.

### 5. Multi-Class Emotion Detection
The analyzer quantifies the emotional resonance of the review across 4 psychological vectors:
- **Joy** (`happy`, `delighted`, `thrilled`, `love`, `amazing`, `खुश`, `आनंद`, `पसंद`)
- **Anger** (`angry`, `furious`, `irritated`, `annoyed`, `hate`, `गुस्सा`, `नाराज`, `क्रोध`)
- **Fear** (`scared`, `afraid`, `worried`, `nervous`, `anxious`, `डर`, `भय`, `चिंता`)
- **Sadness** (`sad`, `disappointed`, `upset`, `depressed`, `regret`, `दुखी`, `निराश`, `बर्बाद`)

Scores are normalized between 0% and 100% based on emotion density relative to token volume.

### 6. Explainability & Keyword Extraction
Provides transparency by extracting specific words influencing the score:
- **English**: Powered by VADER's validated sentiment lexicon (`self.sid.lexicon`).
- **Hindi**: Evaluated against a curated Devanagari sentiment keyword lexicon.
- Results are surfaced as positive and negative badges in the UI.

### 7. Lexical & Tokenization Baseline: NLTK VADER
- Uses NLTK's **VADER** (`SentimentIntensityAnalyzer`) and **Punkt** sentence/word tokenizers.
- Assets are pre-packaged locally in `nltk_data/` (`tokenizers/punkt` and `sentiment/vader_lexicon.zip`), ensuring no network lookup is needed for lexical operations.

---

## Contextual Sarcasm Detection

Sarcasm is one of the hardest NLP tasks in customer review analysis. Consider these real benchmark cases:

| Input Review | Standalone Lexical (VADER) | Qwen 2.5-3B-Instruct (Our Engine) | Explanation |
| :--- | :---: | :---: | :--- |
| *"Oh fantastic, it arrived completely broken. Just what I needed."* | **Positive (+0.75)** ❌ | **Negative (-0.95)** ✅ | Lexicon sees "fantastic" and "needed". Qwen detects that receiving broken goods is unacceptable. |
| *"Yeah right, like I'm going to buy this again."* | **Neutral (0.00)** ❌ | **Negative (-0.85)** ✅ | Lexicon detects no strong polarity words. Qwen identifies dismissive sarcasm and future refusal. |
| *"Great product if you enjoy wasting your time and money."* | **Positive (+0.62)** ❌ | **Negative (-0.90)** ✅ | Lexicon is misled by "Great product". Qwen analyzes the conditional caveat ("wasting time and money"). |

The transition from fragile hardcoded regex rules to instruction-tuned Large Language Models enables deep semantic understanding without fragile phrase dictionaries.

---

## Desktop Application: Sentiment Analysis Studio

The graphical interface is built with **PyQt6** and offers a desktop workspace:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Sentiment Analysis Studio                       │
│      Discover the hidden emotions and detailed sentiments inside       │
│                                                                        │
│ ┌─ [Card] Analyze a Review ──────────────────── Language: [English ▼] ┐│
│ │                                                                     ││
│ │  Paste or type your product review here...                          ││
│ │                                                                     ││
│ │                                         [ Analyze Sentiment ] (btn) ││
│ └─────────────────────────────────────────────────────────────────────┘│
│                                                                        │
│ ┌─ [Card] Analysis Results ────────────────────── Overall Score: 0.92 ─┐│
│ │                                                                     ││
│ │                   ┌───────────────────────────────┐                 ││
│ │                   │         Positive 😃           │                 ││
│ │                   └───────────────────────────────┘                 ││
│ │                                                                     ││
│ │  Aspects (Left Column)               Emotions (Right Column)        ││
│ │  Quality: Positive [========  ]      Joy:     85% [============= ]  ││
│ │  Price:   Neutral  [====      ]      Anger:    0% [              ]  ││
│ │  Design:  Positive [==========]      Sadness:  0% [              ]  ││
│ │                                                                     ││
│ │  ─── Analysis Details ────────────────────────────────────────────  ││
│ │  Positive Keywords: amazing, sturdy, recommended                    ││
│ │  Negative Keywords: none                                            ││
│ └─────────────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────────┘
```

### UI Features:
1. **Asynchronous Architecture (`AnalysisThread`)**: Model inferences run on worker threads, keeping the UI responsive even during heavy LLM text generation.
2. **On-Demand Model Download Dialogs**: If large weights (~6GB for Qwen LLM, ~500MB for Hindi RoBERTa) are not detected locally, a styled `QMessageBox` asks the user for permission before downloading.
3. **Card-Based Visual Hierarchy**: Rounded cards with drop shadows (`QGraphicsDropShadowEffect`) and custom scroll containers.
4. **Dynamic Visual Indicators**: Emojis, color-coded badges (Emerald `#10b981`, Amber `#f59e0b`, Red `#ef4444`), and progress bars.
5. **Flash Visual Feedback**: Results card flashes upon completion to signal new results.

---

## Repository Structure

```
NLP-Project/
├── sentiment_analyzer.py          # Core NLP engine: Qwen-2.5, RoBERTa, ABSA, Emotions, VADER
├── sentiment_ui.py                # Modern PyQt6 desktop studio GUI application
├── nltk_data/                     # Pre-packaged offline NLTK resources
│   ├── tokenizers/punkt/          # NLTK punkt sentence tokenizer
│   └── sentiment/                 # VADER sentiment lexicon archive
├── translation/                   # Multilingual translation subsystem
│   ├── translator.py              # Standalone NLLB-200 translation GUI & API module
│   ├── Translation.md             # NLLB translation module documentation
│   └── requirementtranslate.txt   # Translation dependency specifications
├── README.md                      # Comprehensive project documentation
└── .gitignore                     # Git ignore rules
```

---

## Installation & Setup

### 1. Prerequisites
- **Python**: Version `3.10` or higher
- **RAM**: Minimum 8GB (16GB recommended for local Qwen LLM inference)
- **Disk Space**: ~8GB free space for Hugging Face transformer caches

### 2. Environment Setup
It is recommended to run the project inside a virtual environment:

```bash
# Clone the repository
git clone https://github.com/Armaan4477/NLP-Project.git
cd NLP-Project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
Install all required packages:

```bash
pip install --upgrade pip
pip install PyQt6 torch transformers nltk sentencepiece protobuf QtAwesome accelerate deep-translator
```

> **Hardware Acceleration Tip**:
> - **Apple Silicon (Mac M1/M2/M3/M4)**: PyTorch automatically leverages Metal Performance Shaders (`mps`).
> - **NVIDIA GPUs**: Ensure CUDA-enabled PyTorch is installed:
>   ```bash
>   pip install torch --index-url https://download.pytorch.org/whl/cu121
>   ```

---

## Running the Application

### 1. PyQt6 Sentiment Analysis Studio
Launch the desktop application:

```bash
python sentiment_ui.py
```
1. Select your target language (**English** or **Hindi**).
2. Enter or paste a product review into the text area.
3. Click **Analyze Sentiment**.
4. If running for the first time, confirm the model download prompt.

### 2. Command Line Interface (CLI)
Run the command-line interface for interactive testing or sample review analysis:

```bash
python sentiment_analyzer.py
```
This runs built-in test suites covering positive, negative, and sarcastic reviews, then enters interactive mode:
```text
Enter a product review to analyze (or 'quit' to exit):
> Great phone if you enjoy 1-hour battery life.
Review: "Great phone if you enjoy 1-hour battery life."
Sentiment: Negative (compound score: -0.90)
Detailed scores: Positive=0.00, Neutral=0.00, Negative=0.90
```

### 3. Standalone Multilingual Translator GUI
Launch the dedicated NLLB-200 translation interface:

```bash
python translation/translator.py
```
Supports real-time bidirectional translation between 11+ languages with integrated caching.

---

## Programmatic Python API

The `SentimentAnalyzer` class provides a clean programmatic interface:

```python
from sentiment_analyzer import SentimentAnalyzer

# 1. Initialize analyzer
analyzer = SentimentAnalyzer()

# 2. Analyze English review (allow_download=True enables automatic LLM download if absent)
review = "I was skeptical at first, but this exceeded all my expectations! Super fast charging."
result = analyzer.predict(review, language='English', allow_download=True)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence (Compound): {result['scores']['compound']}")
print(f"Aspects: {result['aspects'].keys()}")
print(f"Emotions: {result['emotions']}")
print(f"Positive Keywords: {result['positive_words']}")

# 3. Analyze Hindi review natively
hindi_review = "यह फोन बहुत ही बेहतरीन है, बैटरी बैकअप शानदार है लेकिन कीमत थोड़ी ज्यादा है।"
hindi_result = analyzer.predict(hindi_review, language='Hindi', allow_download=True)

print(f"Hindi Sentiment: {hindi_result['sentiment']}")
print(f"Detected Aspects: {hindi_result['aspects']}")
```

### Batch Analysis & File Export

```python
reviews = [
    "Amazing product, totally worth the money.",
    "Broke after two days of use. Avoid at all costs.",
    "Decent, but shipping took three weeks."
]

# Analyze multiple reviews at once
batch_results = analyzer.predict_batch(reviews, language='English')

# Analyze and dump results directly to JSON
analyzer.analyze_to_file(reviews, output_file="output/analysis_results.json", language='English')
```

### Direct Multilingual Translation API

```python
from translation.translator import translate_text

# Translate from Hindi to English
translated = translate_text("यह उत्पाद बहुत खराब है", source_lang="Hindi", target_lang="English")
print(translated)
# Output: "This product is very bad"
```

---

## Output Data Structure

The `predict()` method returns a dictionary:

```json
{
  "text": "Great phone if you enjoy 1-hour battery life. Screen is gorgeous though.",
  "translated_text": null,
  "scores": {
    "compound": -0.85,
    "pos": 0.0,
    "neg": 0.85,
    "neu": 0.0
  },
  "sentiment": "Negative",
  "sentiment_id": 0,
  "aspects": {
    "performance": {
      "sentiment": "Negative",
      "scores": {"compound": -0.85},
      "text": "Great phone if you enjoy 1-hour battery life."
    },
    "design": {
      "sentiment": "Positive",
      "scores": {"compound": 0.60},
      "text": "Screen is gorgeous though."
    }
  },
  "emotions": {
    "Joy": 0,
    "Anger": 50,
    "Fear": 0,
    "Sadness": 50
  },
  "positive_words": ["great", "gorgeous"],
  "negative_words": []
}
```

---

## Troubleshooting & FAQs

### 1. `ValueError: English model not downloaded and allow_download is False`
- **Cause**: The ~6GB `Qwen/Qwen2.5-3B-Instruct` model weights are not in the local cache, and permission was not granted to download them.
- **Solution**: Pass `allow_download=True` in programmatic calls or click **Yes** on the dialog in the GUI.

### 2. Slow Inference on CPU
- **Cause**: Causal LLM generation on CPU without hardware acceleration can take several seconds per review.
- **Solution**:
  - On macOS, PyTorch uses Apple Silicon MPS automatically.
  - On Linux/Windows, ensure CUDA is available via `torch.cuda.is_available()`.
  - Alternatively, pass smaller inputs or enable translation fallback mode for lightweight inference.

### 3. Transformers `device_map="auto"` Requires `accelerate`
- **Error**: `ImportError: Using device_map='auto' requires accelerate`.
- **Solution**: Run `pip install accelerate`.

### 4. NLTK Data Lookup Errors
- **Error**: `LookupError: Resource punkt not found`.
- **Solution**: The application points `nltk.data.path` to the local `nltk_data/` folder automatically. If running outside the repository root, verify `NLTK_DATA_PATH` in `sentiment_analyzer.py`.

---

## Authors & Contributors

- **Armaan Nakhuda** ([@Armaan4477](https://github.com/Armaan4477))
- **Vrishab Shenvi** ([@vrishab0105](https://github.com/vrishab0105))
