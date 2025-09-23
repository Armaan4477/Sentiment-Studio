"""
Sentiment Analysis UI Application
This module provides a user-friendly GUI for the sentiment analyzer.
"""

import sys
import os
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                            QFrame, QSpacerItem, QSizePolicy, QProgressBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap

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
        input_label = QLabel('Enter Product Review:')
        input_label.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        main_layout.addWidget(input_label)
        
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
        """Analyze the review text and update the UI with results."""
        # Get the review text
        review_text = self.review_input.toPlainText().strip()
        
        if not review_text:
            self.sentiment_label.setText('Please enter a review to analyze')
            self.sentiment_label.setStyleSheet('color: #F44336;')
            return
        
        # Analyze the review
        result = self.analyzer.predict(review_text)
        
        # Update the UI with the results
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
        
        # Flash effect for results frame
        original_style = self.results_frame.styleSheet()
        self.results_frame.setStyleSheet('background-color: #e3f2fd; border-radius: 5px;')
        QTimer.singleShot(300, lambda: self.results_frame.setStyleSheet(original_style))


# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a modern look
    
    # Create and show the main window
    main_window = SentimentAnalysisApp()
    
    sys.exit(app.exec())