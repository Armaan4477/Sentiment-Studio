import sys
import os
from pathlib import Path
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QTextEdit, QFrame, QSplitter,
    QStatusBar, QGridLayout, QSpacerItem, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor, QTextCursor
import qtawesome as qta

# Import translation modules from transformers
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers package not fully available. Install with: pip install transformers")

# Define paths
MODELS_DIR = Path("./models")
OPTIMIZED_DIR = Path("./optimized_models")

# Use the small NLLB model instead of Helsinki models
TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"

# Language codes for NLLB model (flores200 codes)
LANGUAGE_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
    "German": "deu_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "Russian": "rus_Cyrl",
    "Chinese": "zho_Hans",
    "Arabic": "arb_Arab",
    "Portuguese": "por_Latn",
    "Japanese": "jpn_Jpan"
}

# Translation cache
_translation_cache = {}

# Main window styling
WINDOW_STYLE = """
QMainWindow {
    background-color: #f0f0f0;
}
QTextEdit {
    border: 1px solid #bbb;
    border-radius: 4px;
    padding: 8px;
    background-color: white;
    font-size: 14px;
    color: #333333;
}
QPushButton {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton:pressed {
    background-color: #1e40af;
}
QComboBox {
    border: 1px solid #bbb;
    border-radius: 4px;
    padding: 5px;
    background-color: white;
    min-height: 30px;
    color: #333333;
}
QLabel {
    font-size: 14px;
    color: #333333;
}
QStatusBar {
    background-color: #f8f8f8;
    color: #444;
}
"""

class TranslationThread(QThread):
    """Thread for running translation in the background to keep UI responsive"""
    translation_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, text, source_lang, target_lang):
        super().__init__()
        self.text = text
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def run(self):
        try:
            result = translate_text(self.text, self.source_lang, self.target_lang)
            self.translation_complete.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

def clean_for_tts(text):
    """Clean text to make it more suitable for processing."""
    import re
    
    # Remove Markdown-style formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'__(.*?)__', r'\1', text)      # Bold
    text = re.sub(r'_(.*?)_', r'\1', text)        # Italic
    text = re.sub(r'`(.*?)`', r'\1', text)        # Code
    text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
    
    # Remove Markdown headers
    text = re.sub(r'^#{1,6}\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
    
    # Convert bullet points to simple text
    text = re.sub(r'^\s*[-*]\s+(.*?)$', r'• \1', text, flags=re.MULTILINE)
    
    # Fix common symbols
    text = text.replace('&', 'and')
    text = text.replace('>', '')
    text = text.replace('<', '')
    text = text.replace('#', '')
    
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def load_translation_model():
    """Load a single multilingual translation model that handles all language pairs"""
    global _translation_cache
    
    # If model is already loaded, return it
    if "model" in _translation_cache and "tokenizer" in _translation_cache:
        return _translation_cache["model"], _translation_cache["tokenizer"]
    
    # Ensure directories exist
    MODELS_DIR.mkdir(exist_ok=True)
    OPTIMIZED_DIR.mkdir(exist_ok=True)
    
    try:
        # Load model and tokenizer
        print(f"Loading model: {TRANSLATION_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL_ID, 
            cache_dir=str(MODELS_DIR),
            src_lang="eng_Latn"
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL_ID, 
            cache_dir=str(MODELS_DIR),
            low_cpu_mem_usage=True
        )
        
        model.eval()
        
        # Store in cache
        _translation_cache["model"] = model
        _translation_cache["tokenizer"] = tokenizer
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading translation model: {e}")
        raise

def translate_text(text, source_lang, target_lang):
    """Translate text using the NLLB model"""
    # Clean the input text
    text = clean_for_tts(text)
    
    # If text is empty, return empty string
    if not text.strip():
        return ""
    
    # Load the model and tokenizer if not already loaded
    model, tokenizer = load_translation_model()
    
    try:
        # Create translation pipeline on-the-fly
        translator = pipeline(
            "translation", 
            model=model, 
            tokenizer=tokenizer, 
            src_lang=source_lang,
            tgt_lang=target_lang,
            max_length=512
        )
        
        # Break text into manageable chunks to avoid token limits
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # If adding this sentence would make the chunk too long, save current chunk and start a new one
            if len(current_chunk) + len(sentence) > 200:  # Using smaller chunks for NLLB
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        # If no chunks were created, translate the whole text
        if not chunks:
            chunks = [text]
        
        # Translate each chunk
        translated_chunks = []
        for chunk in chunks:
            # Use the pipeline for translation
            result = translator(chunk)[0]['translation_text']
            translated_chunks.append(result)
            
        # Join the translated chunks
        result = " ".join(translated_chunks)
        # Clean up the result
        result = clean_for_tts(result)
        return result
        
    except Exception as e:
        print(f"Translation failed: {e}")
        raise

class TranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multilingual Translator")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(WINDOW_STYLE)
        self.current_thread = None
        
        # Configure icons
        self.translate_icon = qta.icon('fa5s.language', color='white')
        self.swap_icon = qta.icon('fa5s.exchange-alt', color='white')
        self.copy_icon = qta.icon('fa5s.copy', color='white')
        self.clear_icon = qta.icon('fa5s.trash-alt', color='white')
        
        # Setup UI components
        self.setup_ui()
        
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Application title
        title_label = QLabel("Compact Multilingual Translator")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2563eb; margin: 10px; background-color: #f0f0f0;")
        main_layout.addWidget(title_label)
        
        # Language selection
        lang_frame = QFrame()
        lang_frame.setFrameShape(QFrame.Shape.StyledPanel)
        lang_frame.setStyleSheet("background-color: #e0e0e0; border-radius: 8px; padding: 10px;")
        lang_layout = QHBoxLayout()
        lang_frame.setLayout(lang_layout)
        
        # Source language selection
        source_label = QLabel("From:")
        source_label.setStyleSheet("color: #333333; font-weight: bold;")
        lang_layout.addWidget(source_label)
        
        self.source_combo = QComboBox()
        self.source_combo.setStyleSheet("color: #333333; background-color: white;")
        for lang in LANGUAGE_CODES.keys():
            self.source_combo.addItem(lang)
        # Set default to English
        self.source_combo.setCurrentText("English")
        lang_layout.addWidget(self.source_combo)
        
        # Target language selection
        target_label = QLabel("To:")
        target_label.setStyleSheet("color: #333333; font-weight: bold;")
        lang_layout.addWidget(target_label)
        
        self.target_combo = QComboBox()
        self.target_combo.setStyleSheet("color: #333333; background-color: white;")
        for lang in LANGUAGE_CODES.keys():
            self.target_combo.addItem(lang)
        # Set default to Hindi
        self.target_combo.setCurrentText("Hindi")
        lang_layout.addWidget(self.target_combo)
        
        # Add translation and other buttons
        button_layout = QHBoxLayout()
        
        # Translate button
        self.translate_btn = QPushButton("Translate")
        self.translate_btn.setIcon(self.translate_icon)
        self.translate_btn.setIconSize(QSize(20, 20))
        self.translate_btn.clicked.connect(self.translate)
        button_layout.addWidget(self.translate_btn)
        
        # Copy button
        self.copy_btn = QPushButton("Copy Result")
        self.copy_btn.setIcon(self.copy_icon)
        self.copy_btn.setIconSize(QSize(18, 18))
        self.copy_btn.clicked.connect(self.copy_translation)
        button_layout.addWidget(self.copy_btn)
        
        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setIcon(self.clear_icon)
        self.clear_btn.setIconSize(QSize(18, 18))
        self.clear_btn.clicked.connect(self.clear_text)
        button_layout.addWidget(self.clear_btn)
        
        # Add language and button layouts to main layout
        main_layout.addWidget(lang_frame)
        main_layout.addLayout(button_layout)
        
        # Create text areas
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Input text area
        input_frame = QFrame()
        input_layout = QVBoxLayout()
        input_frame.setLayout(input_layout)
        
        input_label = QLabel("Input Text:")
        input_label.setStyleSheet("font-weight: bold; color: #333333; background-color: transparent;")
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter text to translate...")
        self.input_text.setStyleSheet("color: #333333; background-color: white;")
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_text)
        
        # Output text area
        output_frame = QFrame()
        output_layout = QVBoxLayout()
        output_frame.setLayout(output_layout)
        
        output_label = QLabel("Translation:")
        output_label.setStyleSheet("font-weight: bold; color: #333333; background-color: transparent;")
        self.output_text = QTextEdit()
        self.output_text.setPlaceholderText("Translation will appear here...")
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("color: #333333; background-color: white;")
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_text)
        
        # Add frames to splitter
        splitter.addWidget(input_frame)
        splitter.addWidget(output_frame)
        splitter.setSizes([300, 300])  # Equal sizing
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("background-color: #e0e0e0; color: #333333; font-weight: bold;")
        self.status_bar.showMessage("Ready")
        
    def translate(self):
        """Handle translation button click"""
        input_text = self.input_text.toPlainText()
        if not input_text.strip():
            self.status_bar.showMessage("Please enter text to translate")
            return
            
        # Get selected languages
        source_lang_name = self.source_combo.currentText()
        target_lang_name = self.target_combo.currentText()
        
        # Get language codes
        source_lang_code = LANGUAGE_CODES[source_lang_name]
        target_lang_code = LANGUAGE_CODES[target_lang_name]
        
        # Clear previous output
        self.output_text.clear()
        self.status_bar.showMessage(f"Translating from {source_lang_name} to {target_lang_name}...")
        self.translate_btn.setEnabled(False)
        
        # Create and start translation thread
        self.current_thread = TranslationThread(input_text, source_lang_code, target_lang_code)
        self.current_thread.translation_complete.connect(self.handle_translation_complete)
        self.current_thread.error_occurred.connect(self.handle_translation_error)
        self.current_thread.start()
    
    def handle_translation_complete(self, result):
        """Handle completed translation"""
        self.output_text.setText(result)
        self.status_bar.showMessage("Translation complete")
        self.translate_btn.setEnabled(True)
        
    def handle_translation_error(self, error_msg):
        """Handle translation error"""
        self.status_bar.showMessage(f"Error: {error_msg}")
        self.translate_btn.setEnabled(True)
        
    def copy_translation(self):
        """Copy translation to clipboard"""
        text = self.output_text.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.status_bar.showMessage("Translation copied to clipboard")
        else:
            self.status_bar.showMessage("No translation to copy")
            
    def clear_text(self):
        """Clear input and output text areas"""
        self.input_text.clear()
        self.output_text.clear()
        self.status_bar.showMessage("Cleared all text")

def main():
    # Set environment variable to force CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Check if required packages are available
    if not TRANSFORMERS_AVAILABLE:
        app = QApplication(sys.argv)
        QMessageBox.critical(
            None, 
            "Missing Dependencies",
            "The required transformers package is not fully installed.\n\n"
            "Please install it with:\n"
            "pip install transformers torch"
        )
        return
    
    # Create application
    app = QApplication(sys.argv)
    window = TranslatorApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()