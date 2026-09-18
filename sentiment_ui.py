"""
Sentiment Analysis UI Application
This module provides a user-friendly GUI for the sentiment analyzer.
"""

import sys
import os
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                            QFrame, QSpacerItem, QSizePolicy, QProgressBar, QComboBox)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap

class AnalysisThread(QThread):
    """Thread for running sentiment analysis asynchronously."""
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, analyzer, text, language, fallback=False, allow_download=False):
        super().__init__()
        self.analyzer = analyzer
        self.text = text
        self.language = language
        self.fallback = fallback
        self.allow_download = allow_download
        
    def run(self):
        try:
            result = self.analyzer.predict(self.text, self.language, self.fallback, self.allow_download)
            self.analysis_complete.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

# Import the sentiment analyzer
from sentiment_analyzer import SentimentAnalyzer

class SentimentAnalysisApp(QMainWindow):
    """Main window for the Sentiment Analysis Application."""
    
    def __init__(self):
        """Initialize the application window."""
        super().__init__()
        
        # Initialize the sentiment analyzer
        self.analyzer = SentimentAnalyzer()
        
        # Set up the UI
        self.init_ui()
    
    def init_ui(self):
        """Set up the user interface."""
        # Set window properties
        self.setWindowTitle('Sentiment Analysis Tool')
        self.setMinimumSize(700, 500)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Add header
        header_label = QLabel('Product Review Sentiment Analysis')
        header_label.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_label)
        
        # Add description
        description = QLabel('Enter a product review to analyze its sentiment. The tool will classify the review as positive, negative, or neutral.')
        description.setFont(QFont('Arial', 10))
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(description)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)
        
        # Add input section
        input_header_layout = QHBoxLayout()
        input_label = QLabel('Enter Product Review:')
        input_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        input_header_layout.addWidget(input_label)
        
        input_header_layout.addStretch(1)
        
        lang_label = QLabel('Language:')
        lang_label.setFont(QFont('Arial', 10))
        input_header_layout.addWidget(lang_label)
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(['English', 'Hindi'])
        input_header_layout.addWidget(self.lang_combo)
        
        main_layout.addLayout(input_header_layout)
        
        # Text input area
        self.review_input = QTextEdit()
        self.review_input.setFont(QFont('Arial', 11))
        self.review_input.setPlaceholderText('Type your product review here...')
        self.review_input.setMinimumHeight(100)
        main_layout.addWidget(self.review_input)
        
        # Add analyze button
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add spacer to push button to center
        button_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        
        self.analyze_button = QPushButton('Analyze Sentiment')
        self.analyze_button.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        self.analyze_button.setMinimumSize(200, 40)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        self.analyze_button.clicked.connect(self.analyze_review)
        button_layout.addWidget(self.analyze_button)
        
        # Add spacer to push button to center
        button_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        
        main_layout.addLayout(button_layout)
        
        # Add separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator2)
        
        # Add results section
        results_label = QLabel('Analysis Results:')
        results_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        main_layout.addWidget(results_label)
        
        # Results display
        self.results_frame = QFrame()
        self.results_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.results_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.results_frame.setLineWidth(1)
        self.results_frame.setStyleSheet('background-color: #f5f5f5; border-radius: 5px;')
        self.results_frame.setMinimumHeight(150)
        
        results_layout = QVBoxLayout(self.results_frame)
        results_layout.setContentsMargins(15, 15, 15, 15)
        results_layout.setSpacing(10)
        
        # Sentiment result
        self.sentiment_label = QLabel('No review analyzed yet')
        self.sentiment_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        self.sentiment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.sentiment_label)
        
        # Confidence scores section
        scores_layout = QHBoxLayout()
        scores_layout.setContentsMargins(0, 10, 0, 0)
        scores_layout.setSpacing(15)
        
        results_layout.addLayout(scores_layout)
        
        # Compound score
        compound_layout = QHBoxLayout()
        compound_layout.setContentsMargins(0, 10, 0, 0)
        
        compound_label = QLabel('Overall Sentiment Score:')
        compound_label.setFont(QFont('Arial', 10))
        compound_layout.addWidget(compound_label)
        
        self.compound_value = QLabel('0.00')
        self.compound_value.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        compound_layout.addWidget(self.compound_value)
        
        # Add spacer to push the elements to the left
        compound_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        
        results_layout.addLayout(compound_layout)
        
        # Aspects container
        self.aspects_container = QWidget()
        self.aspects_layout = QVBoxLayout(self.aspects_container)
        self.aspects_layout.setContentsMargins(0, 15, 0, 0)
        self.aspects_layout.setSpacing(5)
        results_layout.addWidget(self.aspects_container)
        
        self.aspect_title = QLabel('Aspect Sentiments:')
        self.aspect_title.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        self.aspects_layout.addWidget(self.aspect_title)
        
        self.aspect_widgets = []
        
        # Emotions container
        self.emotions_container = QWidget()
        self.emotions_layout = QVBoxLayout(self.emotions_container)
        self.emotions_layout.setContentsMargins(0, 15, 0, 0)
        self.emotions_layout.setSpacing(5)
        results_layout.addWidget(self.emotions_container)
        
        self.emotion_title = QLabel('Emotions detected:')
        self.emotion_title.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        self.emotions_layout.addWidget(self.emotion_title)
        
        self.emotion_widgets = []
        
        # Explanation container
        self.explanation_container = QWidget()
        self.explanation_layout = QVBoxLayout(self.explanation_container)
        self.explanation_layout.setContentsMargins(0, 15, 0, 0)
        self.explanation_layout.setSpacing(5)
        results_layout.addWidget(self.explanation_container)
        
        self.explanation_title = QLabel('Sentiment Explanation:')
        self.explanation_title.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        self.explanation_layout.addWidget(self.explanation_title)
        
        self.explanation_widgets = []
        
        main_layout.addWidget(self.results_frame)
        
        # Add a stretcher to push everything up
        main_layout.addStretch(1)
        
        # Add footer
        footer_label = QLabel('Powered by NLTK VADER Sentiment Analyzer')
        footer_label.setFont(QFont('Arial', 8))
        footer_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(footer_label)
        
        # Apply some styling to the whole window
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QLabel {
                color: #333;
            }
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        # Show the window
        self.show()
    
    def analyze_review(self):
        """Analyze the review text asynchronously."""
        # Get the review text
        review_text = self.review_input.toPlainText().strip()
        language = self.lang_combo.currentText()
        
        if not review_text:
            self.sentiment_label.setText('Please enter a review to analyze')
            self.sentiment_label.setStyleSheet('color: #F44336;')
            return
            
        fallback = False
        allow_download = False
        
        if language == 'Hindi':
            if not self.analyzer.has_hindi_model():
                from PyQt6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self, 'Download Hindi Model?',
                    'The native Hindi analysis model (~500MB) is not installed.\n\n'
                    'Would you like to download it now for more accurate native analysis?\n'
                    'If you choose No, the tool will fallback to translation.',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.Yes:
                    allow_download = True
                    self.sentiment_label.setText('Downloading Hindi ML model (this will take a few minutes)...')
                else:
                    fallback = True
                    self.sentiment_label.setText('Analyzing (Warning: using translation fallback for Hindi)...')
            else:
                self.sentiment_label.setText('Analyzing natively in Hindi...')
        else:
            self.sentiment_label.setText('Analyzing (this may take a moment if translating)...')
            
        self.sentiment_label.setStyleSheet('color: #2196F3;')
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText('Analyzing...')
        
        # Start analysis thread
        self.thread = AnalysisThread(self.analyzer, review_text, language, fallback, allow_download)
        self.thread.analysis_complete.connect(self.on_analysis_complete)
        self.thread.error_occurred.connect(self.on_analysis_error)
        self.thread.start()
        
    def on_analysis_error(self, error_msg):
        """Handle errors during analysis."""
        self.sentiment_label.setText(f'Error: {error_msg}')
        self.sentiment_label.setStyleSheet('color: #F44336;')
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText('Analyze Sentiment')

    def on_analysis_complete(self, result):
        """Update the UI with results."""
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText('Analyze Sentiment')
        
        sentiment = result['sentiment']
        scores = result['scores']
        
        # Set sentiment label with appropriate color
        if sentiment == 'Positive':
            self.sentiment_label.setText(f'Sentiment: {sentiment} 😃')
            self.sentiment_label.setStyleSheet('color: #4CAF50;')
        elif sentiment == 'Negative':
            self.sentiment_label.setText(f'Sentiment: {sentiment} 😞')
            self.sentiment_label.setStyleSheet('color: #F44336;')
        else:
            self.sentiment_label.setText(f'Sentiment: {sentiment} 😐')
            self.sentiment_label.setStyleSheet('color: #FFC107;')
        
        # Update compound score
        self.compound_value.setText(f"{scores['compound']:.2f}")
        
        # Update aspects, emotions, and explanation
        self.update_aspects(result.get('aspects', {}))
        self.update_emotions(result.get('emotions', {}))
        self.update_explanation(result)
        
        # Show warning if translated text is present (fallback was used)
        if result.get('translated_text'):
            self.sentiment_label.setText(self.sentiment_label.text() + " (Translation Fallback)")
        
        # Flash effect for results frame
        original_style = self.results_frame.styleSheet()
        self.results_frame.setStyleSheet('background-color: #e3f2fd; border-radius: 5px;')
        QTimer.singleShot(300, lambda: self.results_frame.setStyleSheet(original_style))

    def update_aspects(self, aspects):
        """Update the UI with aspect sentiments."""
        # Clear existing aspect widgets
        for widget in self.aspect_widgets:
            self.aspects_layout.removeWidget(widget)
            widget.deleteLater()
        self.aspect_widgets.clear()
        
        if not aspects:
            no_aspects = QLabel("No specific aspects detected in this review.")
            font = QFont('Arial', 10)
            font.setItalic(True)
            no_aspects.setFont(font)
            no_aspects.setStyleSheet('color: #777;')
            self.aspects_layout.addWidget(no_aspects)
            self.aspect_widgets.append(no_aspects)
            return
            
        for aspect, data in aspects.items():
            aspect_row = QWidget()
            row_layout = QHBoxLayout(aspect_row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            
            # Aspect name
            display_name = aspect.replace('_', ' ').title()
            name_label = QLabel(display_name + ":")
            name_label.setMinimumWidth(130)
            name_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
            row_layout.addWidget(name_label)
            
            # Progress bar based on compound score
            compound = data['scores']['compound']
            
            # Normalize compound (-1 to 1) to progress bar (0 to 100)
            progress_val = int((compound + 1.0) / 2.0 * 100)
            
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(progress_val)
            progress.setTextVisible(False)
            progress.setMinimumWidth(150)
            progress.setFixedHeight(12)
            
            # Color based on sentiment
            if data['sentiment'] == 'Positive':
                color = '#4CAF50'
                emoji = '😃'
            elif data['sentiment'] == 'Negative':
                color = '#F44336'
                emoji = '😞'
            else:
                color = '#FFC107'
                emoji = '😐'
                
            progress.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    background-color: #f0f0f0;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 5px;
                }}
            """)
            row_layout.addWidget(progress)
            
            # Sentiment label
            sent_label = QLabel(f"{data['sentiment']} {emoji}")
            sent_label.setFont(QFont('Arial', 10))
            sent_label.setStyleSheet(f'color: {color};')
            sent_label.setMinimumWidth(100)
            row_layout.addWidget(sent_label)
            
            self.aspects_layout.addWidget(aspect_row)
            self.aspect_widgets.append(aspect_row)

    def update_emotions(self, emotions):
        """Update the UI with emotion detection results."""
        for widget in self.emotion_widgets:
            self.emotions_layout.removeWidget(widget)
            widget.deleteLater()
        self.emotion_widgets.clear()
        
        if not emotions or all(v == 0 for v in emotions.values()):
            no_emo = QLabel("No strong emotions detected.")
            font = QFont('Arial', 10)
            font.setItalic(True)
            no_emo.setFont(font)
            no_emo.setStyleSheet('color: #777;')
            self.emotions_layout.addWidget(no_emo)
            self.emotion_widgets.append(no_emo)
            return
            
        for emotion, score in emotions.items():
            if score == 0:
                continue
                
            emo_row = QWidget()
            row_layout = QHBoxLayout(emo_row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            
            name_label = QLabel(f"{emotion}:")
            name_label.setMinimumWidth(100)
            name_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
            row_layout.addWidget(name_label)
            
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(score)
            progress.setTextVisible(False)
            progress.setMinimumWidth(150)
            progress.setFixedHeight(12)
            
            # Colors for emotions
            colors = {
                'Joy': '#4CAF50',     # Green
                'Anger': '#F44336',   # Red
                'Fear': '#9C27B0',    # Purple
                'Sadness': '#2196F3'  # Blue
            }
            color = colors.get(emotion, '#999999')
            
            progress.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    background-color: #f0f0f0;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 5px;
                }}
            """)
            row_layout.addWidget(progress)
            
            score_label = QLabel(f"{score}%")
            score_label.setMinimumWidth(40)
            row_layout.addWidget(score_label)
            
            self.emotions_layout.addWidget(emo_row)
            self.emotion_widgets.append(emo_row)

    def update_explanation(self, result):
        """Update the UI with explanation (positive/negative keywords and translation)."""
        for widget in self.explanation_widgets:
            self.explanation_layout.removeWidget(widget)
            widget.deleteLater()
        self.explanation_widgets.clear()
        
        translated_text = result.get('translated_text')
        if translated_text:
            trans_label = QLabel(f"<b>Translated Text:</b> <i>{translated_text}</i>")
            trans_label.setWordWrap(True)
            trans_label.setFont(QFont('Arial', 10))
            self.explanation_layout.addWidget(trans_label)
            self.explanation_widgets.append(trans_label)
            
        pos_words = result.get('positive_words', [])
        neg_words = result.get('negative_words', [])
        
        if pos_words:
            pos_label = QLabel(f"<b>Positive Keywords:</b> <span style='color: #4CAF50;'>{', '.join(pos_words)}</span>")
            pos_label.setWordWrap(True)
            pos_label.setFont(QFont('Arial', 10))
            self.explanation_layout.addWidget(pos_label)
            self.explanation_widgets.append(pos_label)
            
        if neg_words:
            neg_label = QLabel(f"<b>Negative Keywords:</b> <span style='color: #F44336;'>{', '.join(neg_words)}</span>")
            neg_label.setWordWrap(True)
            neg_label.setFont(QFont('Arial', 10))
            self.explanation_layout.addWidget(neg_label)
            self.explanation_widgets.append(neg_label)
            
        if not pos_words and not neg_words:
            no_words = QLabel("No specific sentiment-bearing keywords found.")
            font = QFont('Arial', 10)
            font.setItalic(True)
            no_words.setFont(font)
            no_words.setStyleSheet('color: #777;')
            self.explanation_layout.addWidget(no_words)
            self.explanation_widgets.append(no_words)


# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a modern look
    
    # Create and show the main window
    main_window = SentimentAnalysisApp()
    
    sys.exit(app.exec())