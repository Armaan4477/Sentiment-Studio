"""
Offline Sentiment Analysis using NLTK's VADER.
This module provides a simple interface for sentiment analysis of product reviews
without requiring internet connectivity after initial setup.
"""

import os
import re
import string
import nltk
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import sys

# Import translation modules
try:
    from translation.translator import translate_text, LANGUAGE_CODES
except ImportError:
    # Use deep-translator as fallback if transformers/torch is missing
    try:
        from deep_translator import MyMemoryTranslator
        def translate_text(text, source_lang, target_lang):
            return MyMemoryTranslator(source='hi-IN', target='en-GB').translate(text)
        LANGUAGE_CODES = {'English': 'en', 'Hindi': 'hi'}
    except ImportError:
        translate_text = None
        LANGUAGE_CODES = {}

# Path for storing NLTK data
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

class SentimentAnalyzer:
    """Class for sentiment analysis of product reviews using VADER."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self._download_nltk_resources()
        self.sid = SentimentIntensityAnalyzer()
        self.hindi_pipeline = None
        self.HINDI_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.english_pipeline = None
        self.ENGLISH_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    
    def has_hindi_model(self):
        """Check if the native Hindi model is downloaded locally."""
        if self.hindi_pipeline is not None:
            return True
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.HINDI_MODEL, local_files_only=True)
            return True
        except Exception:
            return False

    def has_english_model(self):
        """Check if the native English model is downloaded locally."""
        if self.english_pipeline is not None:
            return True
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.ENGLISH_MODEL, local_files_only=True)
            return True
        except Exception:
            return False
    
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        try:
            # Check if resources are already downloaded
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            # Download required resources
            nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)
            nltk.download('vader_lexicon', download_dir=NLTK_DATA_PATH, quiet=True)
    
    def preprocess_text(self, text):
        """
        Basic preprocessing of text.
        
        Args:
            text (str): The text to preprocess.
            
        Returns:
            str: Preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def predict(self, text, language='English', fallback=False, allow_download=False):
        """
        Predict the sentiment of a single text.
        
        Args:
            text (str): The text to analyze.
            language (str): The language of the text. Default is English.
            fallback (bool): If True, use translation fallback when Hindi ML model is missing.
            allow_download (bool): If True, allows downloading the Hindi ML model.
            
        Returns:
            dict: Sentiment scores and classification.
        """
        original_text = text
        translated_text = None
        
        if language == 'Hindi' and not fallback:
            # Use native Hindi model
            if not self.has_hindi_model() and not allow_download:
                raise ValueError("Hindi model not downloaded and allow_download is False.")
                
            if self.hindi_pipeline is None:
                from transformers import pipeline
                self.hindi_pipeline = pipeline("sentiment-analysis", model=self.HINDI_MODEL, tokenizer=self.HINDI_MODEL)
            
            # Predict using HuggingFace
            result = self.hindi_pipeline(text)[0]
            label = result['label']
            score = result['score']
            
            # Map label to Positive/Negative/Neutral
            # cardiffnlp uses: 'negative', 'neutral', 'positive'
            if label == 'positive':
                sentiment = 'Positive'
                sentiment_id = 2
                compound = score
            elif label == 'negative':
                sentiment = 'Negative'
                sentiment_id = 0
                compound = -score
            else:
                sentiment = 'Neutral'
                sentiment_id = 1
                compound = 0.0
                
            scores = {'compound': compound, 'pos': score if sentiment == 'Positive' else 0, 'neg': score if sentiment == 'Negative' else 0, 'neu': score if sentiment == 'Neutral' else 0}
            
            aspects = self.analyze_aspects(text, language)
            emotions = self.analyze_emotions(text, language)
            positive_words, negative_words = self.extract_sentiment_words(text, language)
            
            return {
                'text': text,
                'translated_text': None,
                'scores': scores,
                'sentiment': sentiment,
                'sentiment_id': sentiment_id,
                'aspects': aspects,
                'emotions': emotions,
                'positive_words': list(positive_words),
                'negative_words': list(negative_words)
            }
            
        # English or fallback logic
        if language != 'English' and translate_text is not None and language in LANGUAGE_CODES:
            try:
                translated_text = translate_text(text, LANGUAGE_CODES[language], LANGUAGE_CODES['English'])
                text_to_analyze = translated_text
            except Exception as e:
                print(f"Translation failed: {e}")
                text_to_analyze = text
        else:
            text_to_analyze = text

        processed_text = self.preprocess_text(text_to_analyze)
        
        # Use English Transformer model
        if not self.has_english_model() and not allow_download:
            raise ValueError("English model not downloaded and allow_download is False.")
            
        if self.english_pipeline is None:
            from transformers import pipeline
            import torch
            self.english_pipeline = pipeline(
                "text-generation", 
                model=self.ENGLISH_MODEL, 
                tokenizer=self.ENGLISH_MODEL, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            
        # Predict using HuggingFace Generative LLM
        prompt = f"Analyze the sentiment of the following product review. Consider sarcasm and context carefully. Review: '{processed_text[:1024]}'. Output ONLY a valid JSON object with two keys: 'sentiment' (must be exactly Positive, Negative, or Neutral) and 'confidence' (a float between 0.0 and 1.0). Do not output any other text, markdown formatting, or explanation."
        
        messages = [
            {"role": "system", "content": "You are a highly accurate sentiment analysis API. You only respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.english_pipeline(messages, max_new_tokens=50, return_full_text=False)[0]
        generated_text = result['generated_text'].strip()
        
        # Clean up any potential markdown formatting the LLM might have added
        if generated_text.startswith("```json"):
            generated_text = generated_text[7:]
        elif generated_text.startswith("```"):
            generated_text = generated_text[3:]
        if generated_text.endswith("```"):
            generated_text = generated_text[:-3]
        generated_text = generated_text.strip()
            
        import json
        try:
            parsed = json.loads(generated_text)
            sentiment_str = parsed.get('sentiment', 'Neutral')
            score = float(parsed.get('confidence', 0.5))
        except Exception as e:
            print(f"Failed to parse LLM output: {generated_text}, error: {e}")
            sentiment_str = 'Neutral'
            score = 0.0
            
        if sentiment_str.lower() == 'positive':
            sentiment = 'Positive'
            sentiment_id = 2
            compound = score
        elif sentiment_str.lower() == 'negative':
            sentiment = 'Negative'
            sentiment_id = 0
            compound = -score
        else:
            sentiment = 'Neutral'
            sentiment_id = 1
            compound = 0.0
            
        scores = {'compound': compound, 'pos': score if sentiment == 'Positive' else 0, 'neg': score if sentiment == 'Negative' else 0, 'neu': score if sentiment == 'Neutral' else 0}
        
        aspects = self.analyze_aspects(text_to_analyze, language='English')
        emotions = self.analyze_emotions(text_to_analyze, language='English')
        positive_words, negative_words = self.extract_sentiment_words(text_to_analyze, language='English')
        
        return {
            'text': original_text,
            'translated_text': translated_text,
            'scores': scores,
            'sentiment': sentiment,
            'sentiment_id': sentiment_id,
            'aspects': aspects,
            'emotions': emotions,
            'positive_words': list(positive_words),
            'negative_words': list(negative_words)
        }
    
    def analyze_aspects(self, text, language='English'):
        """Analyze text for sentiment on specific product aspects."""
        if language == 'Hindi':
            aspect_keywords = {
                'quality': ['गुणवत्ता', 'अच्छा', 'क्वालिटी', 'बेहतरीन', 'टिकाऊ', 'मजबूत', 'खराब'],
                'performance': ['प्रदर्शन', 'गति', 'परफॉरमेंस', 'काम', 'धीमा', 'तेज'],
                'price': ['कीमत', 'महंगा', 'सस्ता', 'पैसा', 'मूल्य'],
                'design': ['डिज़ाइन', 'सुंदर', 'दिखावट', 'रंग'],
                'customer_service': ['ग्राहक सेवा', 'मदद', 'सर्विस', 'सपोर्ट'],
                'ease_of_use': ['उपयोग', 'आसान', 'कठिन', 'सरल']
            }
        else:
            aspect_keywords = {
                'quality': ['quality', 'build', 'material', 'sturdy', 'cheap', 'durable', 'broke', 'broken'],
                'performance': ['performance', 'speed', 'fast', 'slow', 'battery', 'lag', 'works', 'working'],
                'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth'],
                'design': ['design', 'look', 'color', 'ugly', 'beautiful', 'style', 'appearance'],
                'customer_service': ['service', 'support', 'help', 'agent', 'representative', 'call', 'rude', 'friendly'],
                'ease_of_use': ['easy', 'hard', 'difficult', 'simple', 'intuitive', 'setup', 'instructions', 'use']
            }
        
        aspect_sentiments = {}
        sentences = nltk.sent_tokenize(text)
        
        for aspect_name, keywords in aspect_keywords.items():
            relevant_sentences = []
            
            # Find sentences mentioning the aspect
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    relevant_sentences.append(sentence)
            
            # If aspect is mentioned, calculate sentiment for those specific sentences
            if relevant_sentences:
                combined_text = " ".join(relevant_sentences)
                if language == 'Hindi':
                    # Fallback to general ML model score logic for aspect sentence if native
                    if self.hindi_pipeline is None:
                        from transformers import pipeline
                        self.hindi_pipeline = pipeline("sentiment-analysis", model=self.HINDI_MODEL, tokenizer=self.HINDI_MODEL)
                    res = self.hindi_pipeline(combined_text[:512])[0]
                    label = res['label']
                    sentiment = 'Positive' if label == 'positive' else ('Negative' if label == 'negative' else 'Neutral')
                    comp = res['score'] if sentiment == 'Positive' else (-res['score'] if sentiment == 'Negative' else 0.0)
                    scores = {'compound': comp}
                else:
                    scores = self.sid.polarity_scores(combined_text)
                    if scores['compound'] >= 0.05:
                        sentiment = "Positive"
                    elif scores['compound'] <= -0.05:
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"
                    
                aspect_sentiments[aspect_name] = {
                    'sentiment': sentiment,
                    'scores': scores,
                    'text': combined_text
                }
        
        return aspect_sentiments
        
    def analyze_emotions(self, text, language='English'):
        """Analyze text for underlying emotions like joy, anger, fear, and sadness."""
        if language == 'Hindi':
            emotion_lexicon = {
                'Joy': ['खुश', 'आनंद', 'पसंद', 'अच्छा', 'सुंदर', 'मज़ा', 'प्यार'],
                'Anger': ['गुस्सा', 'नाराज', 'क्रोध', 'खराब', 'बकवास', 'नफरत'],
                'Fear': ['डर', 'भय', 'घबराहट', 'चिंता'],
                'Sadness': ['दुखी', 'निराश', 'खराब', 'अफसोस', 'रोता', 'बर्बाद']
            }
        else:
            emotion_lexicon = {
                'Joy': ['happy', 'excited', 'delighted', 'thrilled', 'joy', 'glad', 'love', 'amazing', 'perfect', 'great'],
                'Anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated', 'hate', 'terrible', 'worst'],
                'Fear': ['scared', 'afraid', 'worried', 'nervous', 'terrified', 'anxious', 'fear', 'panic'],
                'Sadness': ['sad', 'disappointed', 'upset', 'depressed', 'sorrow', 'unhappy', 'crying', 'regret', 'broke', 'failed']
            }
        
        words = word_tokenize(text.lower())
        emotions = {emotion: 0 for emotion in emotion_lexicon}
        total_words = len(words)
        if total_words == 0:
            return emotions
            
        for emotion, keywords in emotion_lexicon.items():
            count = sum(1 for word in words if word in keywords)
            # Calculate a normalized score (0 to 100) based on matches
            # To make it meaningful, assume 1-3 emotion words in a typical review is 100%
            normalized = min((count / 2) * 100, 100) 
            emotions[emotion] = int(normalized)
            
        return emotions
        
    def extract_sentiment_words(self, text, language='English'):
        """Extract positive and negative keywords from the text."""
        words = word_tokenize(text.lower())
        positive_words = set()
        negative_words = set()
        
        if language == 'Hindi':
            hindi_pos = ['अच्छा', 'बेहतरीन', 'सुंदर', 'मज़ा', 'प्यार', 'खुश', 'आनंद', 'टिकाऊ', 'मजबूत', 'तेज', 'पसंद']
            hindi_neg = ['खराब', 'धीमा', 'बकवास', 'महंगा', 'गुस्सा', 'दुखी', 'निराश', 'नफरत', 'बर्बाद', 'कठिन']
            for word in words:
                word = word.strip(string.punctuation)
                if word in hindi_pos:
                    positive_words.add(word)
                elif word in hindi_neg:
                    negative_words.add(word)
            return positive_words, negative_words

        # English logic using VADER
        for word in words:
            # Strip punctuation
            word = word.strip(string.punctuation)
            if not word:
                continue
                
            score = self.sid.lexicon.get(word, 0)
            if score > 0:
                positive_words.add(word)
            elif score < 0:
                negative_words.add(word)
                
        return positive_words, negative_words
    
    def predict_batch(self, texts, language='English'):
        """
        Predict the sentiment of multiple texts.
        
        Args:
            texts (list): List of texts to analyze.
            
        Returns:
            list: List of sentiment predictions.
        """
        return [self.predict(text, language) for text in texts]
    
    def analyze_to_file(self, texts, output_file, language='English'):
        """
        Analyze texts and save results to a file.
        
        Args:
            texts (list): List of texts to analyze.
            output_file (str): Path to save the results.
        """
        results = self.predict_batch(texts, language)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save results to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        return results


def analyze_review(text, language='English'):
    """
    Analyze a single review and print the result.
    
    Args:
        text (str): The review text to analyze.
        language (str): Language of the review.
    """
    analyzer = SentimentAnalyzer()
    result = analyzer.predict(text, language, allow_download=True)
    
    print(f"Review: \"{text}\"")
    if result.get('translated_text'):
        print(f"Translated to: \"{result['translated_text']}\"")
    print(f"Sentiment: {result['sentiment']} (compound score: {result['scores']['compound']:.2f})")
    print(f"Detailed scores: Positive={result['scores']['pos']:.2f}, Neutral={result['scores']['neu']:.2f}, Negative={result['scores']['neg']:.2f}")
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Offline Sentiment Analysis using NLTK VADER")
    print("=" * 60)
    print("This tool analyzes product reviews and classifies them as positive, negative, or neutral.")
    print("=" * 60)
    
    # Test on sample reviews
    test_texts = [
        "This product is amazing! I love it so much.",
        "The quality is decent but not exceptional.",
        "Terrible product. Broke after one use. Don't waste your money.",
        "Average performance, nothing special but it works.",
        "Best purchase I've made all year! Highly recommended!",
        # Basic sentences
        "okay",
        "not bad",
        "terrible",
        # Sarcastic sentences
        "Yeah right, like I'm going to buy this again.",
        "Oh fantastic, it arrived broken. Just what I needed.",
        "Great product if you enjoy wasting your time."
    ]
    
    print("Analyzing sample reviews:")
    for text in test_texts:
        analyze_review(text)
        print()
    
    # Interactive mode
    print("\nEnter a product review to analyze (or 'quit' to exit):")
    while True:
        user_input = input("> ")
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        analyze_review(user_input)
        print()