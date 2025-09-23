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
    
    def predict(self, text):
        """
        Predict the sentiment of a single text.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            dict: Sentiment scores and classification.
        """
        processed_text = self.preprocess_text(text)
        
        # Get sentiment scores
        scores = self.sid.polarity_scores(processed_text)
        
        # Additional handling for common negative phrases that VADER might misclassify
        lower_text = text.lower()
        
        # 🔹 Phrases that should always be negative but might be misclassified
        negative_phrases = [
            "don't think", "not worth", "not good enough", "not recommend", 
            "not worth the money", "not worth the price", "not satisfied",
            "never buy again", "would not buy", "not impressed", "not acceptable",
            "not reliable", "not happy", "not useful", "not helpful", "not great",
            "not recommended", "waste of time", "waste of money", "big disappointment",
            "poor quality", "very poor", "substandard", "cheaply made", "not durable",
            "not trustworthy", "not enjoyable", "not comfortable",
            "great but", "amazing but", "excellent but", "perfect but",
            "broke after", "stopped after", "failed after", "works until", "great until",
            "amazing until", "fantastic until", "perfect until", "good until it wasn’t",
            "works perfectly...", "great...", "excellent...", "perfect...", "sure...",
            "love it...", "amazing...", "fantastic...", "best...", "yeah right...",
            "waste money", "waste your money", "throw money", "throwing money", 
            "enjoy disappointment", "enjoy frustration", "enjoy problems",
            "hate quality", "garbage", "regret", "junk",
            "totally worth the regret", "love wasting money", "dream product if you hate sleep",
            "amazing deal... for the trash", "best garbage", "fantastic junk"
        ]

        # 🔹 Product malfunction phrases that should be strongly negative
        malfunction_phrases = [
            "does not work", "doesn't work", "not work", "not working",
            "does not function", "doesn't function", "not functioning",
            "not operating", "doesn't operate", "does not operate",
            "broken", "defective", "malfunctioning", "fails to work",
            "stopped working", "no longer works", "failed", "doesn't perform",
            "keeps crashing", "keeps freezing", "shuts down randomly", "won't turn on",
            "does not start", "does not load", "keeps restarting", "boot loop",
            "faulty", "dead on arrival", "glitchy", "unusable", "buggy",
            "stops responding", "overheats and shuts down", "screen went black",
            "does not connect", "loses signal", "battery won't charge",
            "doesn't hold charge", "fell apart", "stopped charging"
        ]

        # 🔹 Sarcasm indicators - phrases that indicate sarcasm
        sarcasm_patterns = [
            # Conditional sarcasm
            "great if you", "perfect if you", "excellent if you", "amazing if you", 
            "wonderful if you", "fantastic if you", "best if you", "ideal if you",
            "awesome if you", "brilliant if you", "superb if you",
            
            # Contrasting clauses
            "great but", "amazing but", "excellent but", "perfect but",
            "broke after", "stopped after", "failed after", "works until", "great until",
            "amazing until", "fantastic until", "perfect until", "good until it wasn’t",
            
            # Ellipsis followed by contradiction
            "works perfectly...", "great...", "excellent...", "perfect...", "sure...",
            "love it...", "amazing...", "fantastic...", "best...", "yeah right...",
            
            # Negative contexts following positive terms
            "waste money", "waste your money", "throw money", "throwing money", 
            "enjoy disappointment", "enjoy frustration", "enjoy problems",
            "hate quality", "garbage", "regret", "junk",
            "totally worth the regret", "love wasting money", "dream product if you hate sleep",
            "amazing deal... for the trash", "best garbage", "fantastic junk"
        ]

        # Check for negative phrases that might be misclassified
        phrase_found = any(phrase in lower_text for phrase in negative_phrases)

        # Check for malfunction phrases
        malfunction_found = any(phrase in lower_text for phrase in malfunction_phrases)

        # Check for sarcasm patterns
        sarcasm_found = any(pattern in lower_text for pattern in sarcasm_patterns)

        # Also check for specific sentence structures indicating sarcasm:
        # 1. Positive sentiment words followed by negative context
        positive_words = ["great", "amazing", "excellent", "perfect", "wonderful", "fantastic", "best", "brilliant"]
        negative_contexts = ["waste", "broke", "fail", "disappoint", "regret", "throw", "garbage", "junk", "terrible", "awful", "frustrat"]

        # Split the text into words
        words = lower_text.split()

        # Check if a positive word is followed by a negative context in the sentence
        has_positive_followed_by_negative = False
        for i, word in enumerate(words):
            if any(pos_word in word for pos_word in positive_words):
                # Check if any negative context appears after this positive word
                remaining_text = " ".join(words[i:])
                if any(neg_context in remaining_text for neg_context in negative_contexts):
                    has_positive_followed_by_negative = True
                    break
        
        # Override score if needed for edge cases
        if phrase_found and scores['compound'] > -0.1:  # More aggressive threshold
            # Adjust the compound score to be negative
            scores['compound'] = -0.2  # Set a mild negative score
            # Also adjust component scores
            scores['neg'] = max(scores['neg'], 0.4)
            scores['pos'] = min(scores['pos'], 0.1)
        
        # Stronger override for malfunction phrases
        if malfunction_found:
            # Always make malfunction phrases strongly negative
            scores['compound'] = min(scores['compound'], -0.4)  # Set at least a moderate negative score
            scores['neg'] = max(scores['neg'], 0.6)  # Increase negative component
            scores['neu'] = min(scores['neu'], 0.4)  # Decrease neutral component
            scores['pos'] = 0.0  # Remove positive component
        
        # For cases where phrases contain both "very" and negative sentiment indicators
        if "very bad" in lower_text or "very poor" in lower_text or "very terrible" in lower_text:
            scores['compound'] = min(scores['compound'], -0.7)  # Even stronger negative
            scores['neg'] = max(scores['neg'], 0.8)
        
        # Handle sarcasm - if detected, reverse positive sentiment to negative
        if (sarcasm_found or has_positive_followed_by_negative) and scores['compound'] > 0:
            # Reverse the sentiment - make positive scores negative
            original_compound = scores['compound']
            scores['compound'] = -original_compound  # Invert the sentiment
            
            # Swap positive and negative scores
            original_pos = scores['pos']
            scores['pos'] = 0.0
            scores['neg'] = max(scores['neg'], original_pos)
            
            # Ensure the compound score is sufficiently negative
            scores['compound'] = min(scores['compound'], -0.3)
        
        # Determine sentiment category based on compound score
        if scores['compound'] >= 0.05:
            sentiment = "Positive"
            sentiment_id = 2
        elif scores['compound'] <= -0.05:
            sentiment = "Negative"
            sentiment_id = 0
        else:
            sentiment = "Neutral"
            sentiment_id = 1
        
        return {
            'text': text,
            'scores': scores,
            'sentiment': sentiment,
            'sentiment_id': sentiment_id
        }
    
    def predict_batch(self, texts):
        """
        Predict the sentiment of multiple texts.
        
        Args:
            texts (list): List of texts to analyze.
            
        Returns:
            list: List of sentiment predictions.
        """
        return [self.predict(text) for text in texts]
    
    def analyze_to_file(self, texts, output_file):
        """
        Analyze texts and save results to a file.
        
        Args:
            texts (list): List of texts to analyze.
            output_file (str): Path to save the results.
        """
        results = self.predict_batch(texts)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save results to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        return results


def analyze_review(text):
    """
    Analyze a single review and print the result.
    
    Args:
        text (str): The review text to analyze.
    """
    analyzer = SentimentAnalyzer()
    result = analyzer.predict(text)
    
    print(f"Review: \"{text}\"")
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
        "Best purchase I've made all year! Highly recommended!"
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