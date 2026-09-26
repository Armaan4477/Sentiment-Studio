"""
Sentiment Analysis UI Application
This module provides a user-friendly GUI for the sentiment analyzer.
"""

import sys
import os
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                            QFrame, QSpacerItem, QSizePolicy, QProgressBar, QComboBox,
                            QGraphicsDropShadowEffect, QScrollArea)
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
    
    # Define Theme Colors
    BG_COLOR = "#f8fafc"          # slate-50
    CARD_BG = "#ffffff"           # white
    TEXT_PRIMARY = "#0f172a"      # slate-900
    TEXT_SECONDARY = "#475569"    # slate-600
    ACCENT_COLOR = "#2563eb"      # blue-600
    ACCENT_HOVER = "#1d4ed8"      # blue-700
    SUCCESS_COLOR = "#10b981"     # emerald-500
    WARNING_COLOR = "#f59e0b"     # amber-500
    DANGER_COLOR = "#ef4444"      # red-500
    BORDER_COLOR = "#e2e8f0"      # slate-200
    FONT_FAMILY = "Helvetica Neue"
    
    def __init__(self):
        """Initialize the application window."""
        super().__init__()
        
        # Initialize the sentiment analyzer
        self.analyzer = SentimentAnalyzer()
        
        # Set up the UI
        self.init_ui()
    
    def create_shadow(self):
        """Create a standard drop shadow effect."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 4)
        return shadow

    def create_card(self):
        """Create a styled card frame."""
        card = QFrame()
        card.setObjectName("Card")
        card.setGraphicsEffect(self.create_shadow())
        card.setStyleSheet(f"""
            QFrame#Card {{
                background-color: {self.CARD_BG};
                border-radius: 12px;
                border: 1px solid {self.BORDER_COLOR};
            }}
        """)
        return card

    def init_ui(self):
        """Set up the user interface."""
        # Set window properties
        self.setWindowTitle('Sentiment Analysis Studio')
        self.setMinimumSize(850, 700)
        
        # Main Stylesheet
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {self.BG_COLOR};
            }}
            QWidget {{
                color: {self.TEXT_PRIMARY};
                font-family: '{self.FONT_FAMILY}', 'Inter', sans-serif;
            }}
            QLabel {{
                color: {self.TEXT_PRIMARY};
            }}
            QTextEdit {{
                background-color: {self.BG_COLOR};
                border: 1px solid {self.BORDER_COLOR};
                border-radius: 8px;
                padding: 12px;
                color: {self.TEXT_PRIMARY};
                selection-background-color: {self.ACCENT_COLOR};
            }}
            QTextEdit:focus {{
                border: 1px solid {self.ACCENT_COLOR};
            }}
            /* QComboBox uses native macOS styling, but we explicitly style the dropdown menu 
               so it doesn't inherit the OS dark mode background with our forced dark text */
            QComboBox QAbstractItemView {{
                background-color: {self.CARD_BG};
                color: {self.TEXT_PRIMARY};
                selection-background-color: {self.ACCENT_COLOR};
                selection-color: white;
            }}
            QScrollBar:vertical {{
                border: none;
                background: {self.CARD_BG};
                width: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {self.BORDER_COLOR};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {self.TEXT_SECONDARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
        """)
        
        # Central Widget & Scroll Area
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("background-color: transparent;")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(30, 30, 30, 30)
        scroll_layout.setSpacing(20)
        
        # --- Header Section ---
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)
        
        header_label = QLabel('Sentiment Analysis Studio')
        header_label.setFont(QFont(self.FONT_FAMILY, 24, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet(f"color: {self.TEXT_PRIMARY};")
        header_layout.addWidget(header_label)
        
        description = QLabel('Discover the hidden emotions and detailed sentiments inside your product reviews.')
        description.setFont(QFont(self.FONT_FAMILY, 12))
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet(f"color: {self.TEXT_SECONDARY};")
        header_layout.addWidget(description)
        
        scroll_layout.addLayout(header_layout)
        scroll_layout.addSpacing(10)
        
        # --- Input Card ---
        input_card = self.create_card()
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(20, 20, 20, 20)
        input_layout.setSpacing(15)
        
        input_header_layout = QHBoxLayout()
        input_label = QLabel('Analyze a Review')
        input_label.setFont(QFont(self.FONT_FAMILY, 14, QFont.Weight.Bold))
        input_header_layout.addWidget(input_label)
        
        input_header_layout.addStretch(1)
        
        lang_label = QLabel('Language:')
        lang_label.setFont(QFont(self.FONT_FAMILY, 11))
        lang_label.setStyleSheet(f"color: {self.TEXT_SECONDARY};")
        input_header_layout.addWidget(lang_label)
        
        self.lang_combo = QComboBox()
        self.lang_combo.setFont(QFont(self.FONT_FAMILY, 11))
        self.lang_combo.addItems(['English', 'Hindi'])
        input_header_layout.addWidget(self.lang_combo)
        
        input_layout.addLayout(input_header_layout)
        
        # Text input area
        self.review_input = QTextEdit()
        self.review_input.setFont(QFont(self.FONT_FAMILY, 12))
        self.review_input.setPlaceholderText('Paste or type your product review here...')
        self.review_input.setMinimumHeight(120)
        input_layout.addWidget(self.review_input)
        
        # Add analyze button
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 5, 0, 0)
        
        self.analyze_button = QPushButton('Analyze Sentiment')
        self.analyze_button.setFont(QFont(self.FONT_FAMILY, 12, QFont.Weight.Bold))
        self.analyze_button.setMinimumSize(220, 45)
        self.analyze_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.ACCENT_COLOR};
                color: white;
                border-radius: 8px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self.ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: #1d4ed8;
            }}
            QPushButton:disabled {{
                background-color: {self.BORDER_COLOR};
                color: {self.TEXT_SECONDARY};
            }}
        """)
        self.analyze_button.clicked.connect(self.analyze_review)
        
        button_layout.addStretch()
        button_layout.addWidget(self.analyze_button)
        input_layout.addLayout(button_layout)
        
        scroll_layout.addWidget(input_card)
        
        # --- Results Card ---
        self.results_card = self.create_card()
        results_layout = QVBoxLayout(self.results_card)
        results_layout.setContentsMargins(25, 25, 25, 25)
        results_layout.setSpacing(20)
        
        # Top Header of Results
        results_header = QHBoxLayout()
        results_label = QLabel('Analysis Results')
        results_label.setFont(QFont(self.FONT_FAMILY, 16, QFont.Weight.Bold))
        results_header.addWidget(results_label)
        results_header.addStretch()
        
        # Compound score display (top right)
        self.compound_frame = QFrame()
        self.compound_frame.setStyleSheet(f"background-color: {self.BG_COLOR}; border-radius: 8px; padding: 5px;")
        compound_layout = QHBoxLayout(self.compound_frame)
        compound_layout.setContentsMargins(10, 5, 10, 5)
        compound_label = QLabel('Overall Score:')
        compound_label.setFont(QFont(self.FONT_FAMILY, 10))
        compound_label.setStyleSheet(f"color: {self.TEXT_SECONDARY};")
        self.compound_value = QLabel('--')
        self.compound_value.setFont(QFont(self.FONT_FAMILY, 12, QFont.Weight.Bold))
        compound_layout.addWidget(compound_label)
        compound_layout.addWidget(self.compound_value)
        results_header.addWidget(self.compound_frame)
        self.compound_frame.hide()
        
        results_layout.addLayout(results_header)
        
        # Primary Sentiment Banner
        self.sentiment_banner = QFrame()
        self.sentiment_banner.setStyleSheet(f"background-color: {self.BG_COLOR}; border-radius: 12px;")
        banner_layout = QVBoxLayout(self.sentiment_banner)
        banner_layout.setContentsMargins(15, 20, 15, 20)
        
        self.sentiment_label = QLabel('Ready to analyze')
        self.sentiment_label.setFont(QFont(self.FONT_FAMILY, 18, QFont.Weight.Bold))
        self.sentiment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sentiment_label.setStyleSheet(f"color: {self.TEXT_SECONDARY};")
        banner_layout.addWidget(self.sentiment_label)
        results_layout.addWidget(self.sentiment_banner)
        
        # Details Area (Two Columns)
        details_layout = QHBoxLayout()
        details_layout.setSpacing(20)
        
        # Left Column: Aspects
        self.aspects_container = QWidget()
        self.aspects_layout = QVBoxLayout(self.aspects_container)
        self.aspects_layout.setContentsMargins(0, 0, 0, 0)
        self.aspects_layout.setSpacing(12)
        
        self.aspect_title = QLabel('Aspects')
        self.aspect_title.setFont(QFont(self.FONT_FAMILY, 13, QFont.Weight.Bold))
        self.aspects_layout.addWidget(self.aspect_title)
        
        self.aspect_widgets = []
        details_layout.addWidget(self.aspects_container, 1)
        
        # Right Column: Emotions
        self.emotions_container = QWidget()
        self.emotions_layout = QVBoxLayout(self.emotions_container)
        self.emotions_layout.setContentsMargins(0, 0, 0, 0)
        self.emotions_layout.setSpacing(12)
        
        self.emotion_title = QLabel('Emotions')
        self.emotion_title.setFont(QFont(self.FONT_FAMILY, 13, QFont.Weight.Bold))
        self.emotions_layout.addWidget(self.emotion_title)
        
        self.emotion_widgets = []
        details_layout.addWidget(self.emotions_container, 1)
        
        results_layout.addLayout(details_layout)
        
        # Explanation Section
        self.explanation_container = QWidget()
        self.explanation_layout = QVBoxLayout(self.explanation_container)
        self.explanation_layout.setContentsMargins(0, 15, 0, 0)
        self.explanation_layout.setSpacing(8)
        
        # Add separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: {self.BORDER_COLOR};")
        self.explanation_layout.addWidget(sep)
        
        self.explanation_title = QLabel('Analysis Details')
        self.explanation_title.setFont(QFont(self.FONT_FAMILY, 13, QFont.Weight.Bold))
        self.explanation_title.setContentsMargins(0, 10, 0, 0)
        self.explanation_layout.addWidget(self.explanation_title)
        
        self.explanation_widgets = []
        results_layout.addWidget(self.explanation_container)
        
        # Hide sections initially
        self.aspects_container.hide()
        self.emotions_container.hide()
        self.explanation_container.hide()
        
        scroll_layout.addWidget(self.results_card)
        
        # Add a stretcher to push everything up
        scroll_layout.addStretch(1)
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Add footer
        footer_label = QLabel('Powered by NLP Models')
        footer_label.setFont(QFont(self.FONT_FAMILY, 9))
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_label.setStyleSheet(f"color: {self.TEXT_SECONDARY}; padding: 10px;")
        main_layout.addWidget(footer_label)
        
        # Show the window
        self.show()
    
    def analyze_review(self):
        """Analyze the review text asynchronously."""
        review_text = self.review_input.toPlainText().strip()
        language = self.lang_combo.currentText()
        
        if not review_text:
            self.sentiment_label.setText('Please enter a review to analyze')
            self.sentiment_label.setStyleSheet(f'color: {self.WARNING_COLOR};')
            return
            
        fallback = False
        allow_download = False
        
        if language == 'Hindi':
            if not self.analyzer.has_hindi_model():
                from PyQt6.QtWidgets import QMessageBox
                
                # Style the message box to match theme
                msg = QMessageBox(self)
                msg.setWindowTitle('Download Hindi Model?')
                msg.setText('The native Hindi analysis model (~500MB) is not installed.')
                msg.setInformativeText('Would you like to download it now for more accurate native analysis?\\nIf you choose No, the tool will fallback to translation.')
                msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg.setDefaultButton(QMessageBox.StandardButton.Yes)
                msg.setStyleSheet(self.styleSheet() + " QMessageBox { background-color: " + self.CARD_BG + "; }")
                
                reply = msg.exec()
                
                if reply == QMessageBox.StandardButton.Yes:
                    allow_download = True
                    self.sentiment_label.setText('Downloading Hindi ML model (this may take a few minutes)...')
                else:
                    fallback = True
                    self.sentiment_label.setText('Analyzing (using translation fallback for Hindi)...')
            else:
                self.sentiment_label.setText('Analyzing natively in Hindi...')
        elif language == 'English':
            if not self.analyzer.has_english_model():
                from PyQt6.QtWidgets import QMessageBox
                
                # Style the message box to match theme
                msg = QMessageBox(self)
                msg.setWindowTitle('Download English LLM?')
                msg.setText('The English Large Language Model (~6GB) is not installed.')
                msg.setInformativeText('Would you like to download it now for highly accurate contextual sarcasm detection?')
                msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg.setDefaultButton(QMessageBox.StandardButton.Yes)
                msg.setStyleSheet(self.styleSheet() + " QMessageBox { background-color: " + self.CARD_BG + "; }")
                
                reply = msg.exec()
                
                if reply == QMessageBox.StandardButton.Yes:
                    allow_download = True
                    self.sentiment_label.setText('Downloading English ML model (this may take a few minutes)...')
                else:
                    self.sentiment_label.setText('Please download the model to continue.')
                    self.sentiment_label.setStyleSheet(f'color: {self.WARNING_COLOR};')
                    return
            else:
                self.sentiment_label.setText('Analyzing natively in English...')
        else:
            self.sentiment_label.setText('Analyzing...')
        self.sentiment_label.setStyleSheet(f'color: {self.ACCENT_COLOR};')
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText('Analyzing...')
        
        # Reset UI
        self.compound_frame.hide()
        self.aspects_container.hide()
        self.emotions_container.hide()
        self.explanation_container.hide()
        
        # Start analysis thread
        self.thread = AnalysisThread(self.analyzer, review_text, language, fallback, allow_download)
        self.thread.analysis_complete.connect(self.on_analysis_complete)
        self.thread.error_occurred.connect(self.on_analysis_error)
        self.thread.start()
        
    def on_analysis_error(self, error_msg):
        """Handle errors during analysis."""
        self.sentiment_label.setText(f'Error: {error_msg}')
        self.sentiment_label.setStyleSheet(f'color: {self.DANGER_COLOR}; font-size: 14px;')
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText('Analyze Sentiment')

    def on_analysis_complete(self, result):
        """Update the UI with results."""
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText('Analyze Sentiment')
        
        self.aspects_container.show()
        self.emotions_container.show()
        self.explanation_container.show()
        self.compound_frame.show()
        
        sentiment = result['sentiment']
        scores = result['scores']
        
        # Set sentiment label with appropriate color
        if sentiment == 'Positive':
            self.sentiment_label.setText(f'{sentiment} 😃')
            self.sentiment_label.setStyleSheet(f'color: {self.SUCCESS_COLOR};')
        elif sentiment == 'Negative':
            self.sentiment_label.setText(f'{sentiment} 😞')
            self.sentiment_label.setStyleSheet(f'color: {self.DANGER_COLOR};')
        else:
            self.sentiment_label.setText(f'{sentiment} 😐')
            self.sentiment_label.setStyleSheet(f'color: {self.WARNING_COLOR};')
        
        # Update compound score
        self.compound_value.setText(f"{scores['compound']:.2f}")
        val = scores['compound']
        if val > 0.05:
            self.compound_value.setStyleSheet(f"color: {self.SUCCESS_COLOR};")
        elif val < -0.05:
            self.compound_value.setStyleSheet(f"color: {self.DANGER_COLOR};")
        else:
            self.compound_value.setStyleSheet(f"color: {self.WARNING_COLOR};")
            
        # Update aspects, emotions, and explanation
        self.update_aspects(result.get('aspects', {}))
        self.update_emotions(result.get('emotions', {}))
        self.update_explanation(result)
        
        # Show warning if translated text is present
        if result.get('translated_text'):
            self.sentiment_label.setText(self.sentiment_label.text() + "\\n(Translated)")
            font = self.sentiment_label.font()
            self.sentiment_label.setFont(QFont(self.FONT_FAMILY, 14, QFont.Weight.Bold))
        else:
            self.sentiment_label.setFont(QFont(self.FONT_FAMILY, 18, QFont.Weight.Bold))
            
        # Flash effect for results frame
        original_style = self.results_card.styleSheet()
        self.results_card.setStyleSheet(f"""
            QFrame#Card {{
                background-color: #e2e8f0; 
                border-radius: 12px;
                border: 1px solid {self.ACCENT_COLOR};
            }}
        """)
        QTimer.singleShot(400, lambda: self.results_card.setStyleSheet(original_style))

    def update_aspects(self, aspects):
        """Update the UI with aspect sentiments."""
        # Clear existing
        for widget in self.aspect_widgets:
            self.aspects_layout.removeWidget(widget)
            widget.deleteLater()
        self.aspect_widgets.clear()
        
        if not aspects:
            no_aspects = QLabel("No specific aspects detected.")
            no_aspects.setFont(QFont(self.FONT_FAMILY, 11, QFont.Weight.Light))
            no_aspects.setStyleSheet(f'color: {self.TEXT_SECONDARY}; font-style: italic;')
            self.aspects_layout.addWidget(no_aspects)
            self.aspect_widgets.append(no_aspects)
            return
            
        for aspect, data in aspects.items():
            aspect_row = QWidget()
            row_layout = QVBoxLayout(aspect_row)
            row_layout.setContentsMargins(0, 5, 0, 5)
            row_layout.setSpacing(4)
            
            # Label container
            label_container = QHBoxLayout()
            display_name = aspect.replace('_', ' ').title()
            name_label = QLabel(display_name)
            name_label.setFont(QFont(self.FONT_FAMILY, 11, QFont.Weight.Medium))
            label_container.addWidget(name_label)
            
            label_container.addStretch()
            
            # Sentiment label
            emoji = '😃' if data['sentiment'] == 'Positive' else ('😞' if data['sentiment'] == 'Negative' else '😐')
            sent_label = QLabel(f"{data['sentiment']} {emoji}")
            sent_label.setFont(QFont(self.FONT_FAMILY, 10))
            
            color = self.SUCCESS_COLOR if data['sentiment'] == 'Positive' else (self.DANGER_COLOR if data['sentiment'] == 'Negative' else self.WARNING_COLOR)
            sent_label.setStyleSheet(f'color: {color};')
            label_container.addWidget(sent_label)
            
            row_layout.addLayout(label_container)
            
            # Progress bar
            compound = data['scores']['compound']
            progress_val = int((compound + 1.0) / 2.0 * 100)
            
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(progress_val)
            progress.setTextVisible(False)
            progress.setFixedHeight(8)
            
            progress.setStyleSheet(f"""
                QProgressBar {{
                    border: none;
                    border-radius: 4px;
                    background-color: {self.BG_COLOR};
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 4px;
                }}
            """)
            row_layout.addWidget(progress)
            
            self.aspects_layout.addWidget(aspect_row)
            self.aspect_widgets.append(aspect_row)
            
        self.aspects_layout.addStretch()

    def update_emotions(self, emotions):
        """Update the UI with emotion detection results."""
        for widget in self.emotion_widgets:
            self.emotions_layout.removeWidget(widget)
            widget.deleteLater()
        self.emotion_widgets.clear()
        
        if not emotions or all(v == 0 for v in emotions.values()):
            no_emo = QLabel("No strong emotions detected.")
            no_emo.setFont(QFont(self.FONT_FAMILY, 11, QFont.Weight.Light))
            no_emo.setStyleSheet(f'color: {self.TEXT_SECONDARY}; font-style: italic;')
            self.emotions_layout.addWidget(no_emo)
            self.emotion_widgets.append(no_emo)
            return
            
        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            if score == 0:
                continue
                
            emo_row = QWidget()
            row_layout = QVBoxLayout(emo_row)
            row_layout.setContentsMargins(0, 5, 0, 5)
            row_layout.setSpacing(4)
            
            # Label container
            label_container = QHBoxLayout()
            name_label = QLabel(emotion)
            name_label.setFont(QFont(self.FONT_FAMILY, 11, QFont.Weight.Medium))
            label_container.addWidget(name_label)
            
            label_container.addStretch()
            
            score_label = QLabel(f"{score}%")
            score_label.setFont(QFont(self.FONT_FAMILY, 10))
            score_label.setStyleSheet(f'color: {self.TEXT_SECONDARY};')
            label_container.addWidget(score_label)
            
            row_layout.addLayout(label_container)
            
            # Progress bar
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(score)
            progress.setTextVisible(False)
            progress.setFixedHeight(8)
            
            colors = {
                'Joy': '#10b981',       # emerald-500
                'Anger': '#ef4444',     # red-500
                'Fear': '#a855f7',      # purple-500
                'Sadness': '#3b82f6',   # blue-500
                'Surprise': '#f59e0b'   # amber-500
            }
            color = colors.get(emotion, self.TEXT_SECONDARY)
            
            progress.setStyleSheet(f"""
                QProgressBar {{
                    border: none;
                    border-radius: 4px;
                    background-color: {self.BG_COLOR};
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 4px;
                }}
            """)
            row_layout.addWidget(progress)
            
            self.emotions_layout.addWidget(emo_row)
            self.emotion_widgets.append(emo_row)
            
        self.emotions_layout.addStretch()

    def update_explanation(self, result):
        """Update the UI with explanation details."""
        for widget in self.explanation_widgets:
            self.explanation_layout.removeWidget(widget)
            widget.deleteLater()
        self.explanation_widgets.clear()
        
        translated_text = result.get('translated_text')
        if translated_text:
            trans_container = QWidget()
            trans_layout = QVBoxLayout(trans_container)
            trans_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl = QLabel("Translated Text")
            lbl.setFont(QFont(self.FONT_FAMILY, 10, QFont.Weight.Bold))
            lbl.setStyleSheet(f"color: {self.TEXT_SECONDARY};")
            trans_layout.addWidget(lbl)
            
            val = QLabel(translated_text)
            val.setWordWrap(True)
            val.setFont(QFont(self.FONT_FAMILY, 11, QFont.Weight.Light))
            val.setStyleSheet(f"color: {self.TEXT_PRIMARY}; font-style: italic;")
            trans_layout.addWidget(val)
            
            self.explanation_layout.addWidget(trans_container)
            self.explanation_widgets.append(trans_container)
            
        pos_words = result.get('positive_words', [])
        neg_words = result.get('negative_words', [])
        
        if pos_words or neg_words:
            words_container = QWidget()
            words_layout = QHBoxLayout(words_container)
            words_layout.setContentsMargins(0, 5, 0, 5)
            
            if pos_words:
                pos_layout = QVBoxLayout()
                lbl = QLabel("Positive Keywords")
                lbl.setFont(QFont(self.FONT_FAMILY, 10, QFont.Weight.Bold))
                lbl.setStyleSheet(f"color: {self.TEXT_SECONDARY};")
                pos_layout.addWidget(lbl)
                
                val = QLabel(', '.join(pos_words))
                val.setWordWrap(True)
                val.setFont(QFont(self.FONT_FAMILY, 11))
                val.setStyleSheet(f"color: {self.SUCCESS_COLOR};")
                pos_layout.addWidget(val)
                pos_layout.addStretch()
                words_layout.addLayout(pos_layout)
                
            if neg_words:
                neg_layout = QVBoxLayout()
                lbl = QLabel("Negative Keywords")
                lbl.setFont(QFont(self.FONT_FAMILY, 10, QFont.Weight.Bold))
                lbl.setStyleSheet(f"color: {self.TEXT_SECONDARY};")
                neg_layout.addWidget(lbl)
                
                val = QLabel(', '.join(neg_words))
                val.setWordWrap(True)
                val.setFont(QFont(self.FONT_FAMILY, 11))
                val.setStyleSheet(f"color: {self.DANGER_COLOR};")
                neg_layout.addWidget(val)
                neg_layout.addStretch()
                words_layout.addLayout(neg_layout)
                
            self.explanation_layout.addWidget(words_container)
            self.explanation_widgets.append(words_container)
            
        if not pos_words and not neg_words and not translated_text:
            no_details = QLabel("No specific sentiment-bearing keywords detected.")
            no_details.setFont(QFont(self.FONT_FAMILY, 11, QFont.Weight.Light))
            no_details.setStyleSheet(f'color: {self.TEXT_SECONDARY}; font-style: italic;')
            self.explanation_layout.addWidget(no_details)
            self.explanation_widgets.append(no_details)


# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Try to set a modern style
    app.setStyle('Fusion')
    
    # Enable high DPI scaling
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        
    main_window = SentimentAnalysisApp()
    sys.exit(app.exec())