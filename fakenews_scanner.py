"""
Fake News Classifier 
25F AI Infrastructure and Arch. - 01
#4 Sentiment Analysis // Azure [10%]
Author: Alexander Sanchez
Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Usage:

python fakenews_scanner.py

Description:

This CLI app generates the following reports
1.classification_report.txt - Detailed metrics report
2.confusion_matrix.png - Visual confusion matrix
3.fake_news_model.pkl - Saved model file
4.batch_predictions.txt - Batch prediction results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import sys
import time
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("Installing colorama for colored output...")
    os.system('pip install colorama')
    from colorama import init, Fore, Style
    init(autoreset=True)

try:
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Visualization libraries not available. Install matplotlib and wordcloud for full features.")


class SpinnerProgress:
    """Animated spinner for progress indication"""
    def __init__(self, message="Processing"):
        self.message = message
        self.spinning = False
        self.spinner_chars = ['|', '/', '-', '\\']
        
    def spin(self, duration=0.1):
        """Show spinner animation"""
        for char in self.spinner_chars:
            if not self.spinning:
                break
            sys.stdout.write(f'\r{self.message} {char}')
            sys.stdout.flush()
            time.sleep(duration)
    
    def start(self):
        """Start spinning"""
        self.spinning = True
        
    def stop(self, final_message=None):
        """Stop spinning"""
        self.spinning = False
        if final_message:
            sys.stdout.write(f'\r{final_message}\n')
        else:
            sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()


class FakeNewsClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_trained = False
        self.training_accuracy = 0
        self.test_accuracy = 0
        self.classification_rep = ""
        self.conf_matrix = None
        
    def print_colored(self, text, color="white"):
        """Print colored text"""
        if not COLORS_AVAILABLE:
            print(text)
            return
            
        color_map = {
            "red": Fore.RED,
            "green": Fore.GREEN,
            "yellow": Fore.YELLOW,
            "blue": Fore.BLUE,
            "cyan": Fore.CYAN,
            "magenta": Fore.MAGENTA,
            "white": Fore.WHITE
        }
        print(color_map.get(color.lower(), Fore.WHITE) + text + Style.RESET_ALL)
    
    def print_banner(self):
        """Print application banner"""
        banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║          FAKE NEWS DETECTION SYSTEM v1.0                  ║
║          25F AI Infrastructure and Arch. - 01             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
        """
        self.print_colored(banner, "cyan")
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z\s]', '', str(text))
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def load_and_prepare_data(self, train_path='train.csv', test_path='test.csv'):
        """Load and prepare dataset"""
        try:
            self.print_colored("\n[1/5] Loading dataset...", "yellow")
            spinner = SpinnerProgress("Loading data")
            spinner.start()
            
            # Try different common dataset formats
            try:
                # Format 1: Separate train and test files
                if os.path.exists(train_path):
                    df_train = pd.read_csv(train_path)
                    df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None
                else:
                    # Format 2: Single file with Fake and True folders
                    fake_path = 'Fake.csv'
                    true_path = 'True.csv'
                    if os.path.exists(fake_path) and os.path.exists(true_path):
                        df_fake = pd.read_csv(fake_path)
                        df_fake['label'] = 0
                        df_true = pd.read_csv(true_path)
                        df_true['label'] = 1
                        df = pd.concat([df_fake, df_true], ignore_index=True)
                        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
                    else:
                        raise FileNotFoundError("Dataset files not found")
            except Exception as e:
                spinner.stop()
                self.print_colored(f"\n Error loading dataset: {e}", "red")
                self.print_colored("\nPlease ensure you have one of these dataset formats:", "yellow")
                self.print_colored("  1. train.csv and test.csv", "white")
                self.print_colored("  2. Fake.csv and True.csv", "white")
                self.print_colored("\nDownload from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset", "cyan")
                return None, None, None, None
            
            spinner.stop(f" Dataset loaded successfully")
            
            # Identify text and label columns
            text_column = None
            label_column = None
            
            # Common column names for text
            for col in ['text', 'title', 'content', 'article', 'news']:
                if col in df_train.columns:
                    text_column = col
                    break
            
            # Common column names for labels
            for col in ['label', 'class', 'target', 'category']:
                if col in df_train.columns:
                    label_column = col
                    break
            
            if not text_column:
                # Try to combine title and text if both exist
                if 'title' in df_train.columns and 'text' in df_train.columns:
                    df_train['combined_text'] = df_train['title'] + " " + df_train['text']
                    if df_test is not None:
                        df_test['combined_text'] = df_test['title'] + " " + df_test['text']
                    text_column = 'combined_text'
                else:
                    text_column = df_train.columns[0]  # Use first column as fallback
            
            if not label_column:
                label_column = df_train.columns[-1]  # Use last column as fallback
            
            self.print_colored(f"[2/5] Preprocessing data (using '{text_column}' column)...", "yellow")
            spinner = SpinnerProgress("Cleaning text")
            spinner.start()
            
            # Clean and prepare data
            df_train[text_column] = df_train[text_column].apply(self.clean_text)
            if df_test is not None:
                df_test[text_column] = df_test[text_column].apply(self.clean_text)
            
            # Remove empty rows
            df_train = df_train[df_train[text_column].str.len() > 10]
            if df_test is not None:
                df_test = df_test[df_test[text_column].str.len() > 10]
            
            spinner.stop(" Data preprocessed")
            
            X_train = df_train[text_column].values
            y_train = df_train[label_column].values
            
            if df_test is not None:
                X_test = df_test[text_column].values
                y_test = df_test[label_column].values
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
            
            self.print_colored(f"Training samples: {len(X_train)}", "green")
            self.print_colored(f"Testing samples: {len(X_test)}", "green")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.print_colored(f"\nError in data preparation: {str(e)}", "red")
            return None, None, None, None
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train the classification model"""
        try:
            self.print_colored("\n[3/5] Vectorizing text data...", "yellow")
            spinner = SpinnerProgress("Creating TF-IDF features")
            spinner.start()
            
            # TF-IDF Vectorization
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            spinner.stop(" Vectorization complete")
            
            self.print_colored("[4/5] Training Logistic Regression model...", "yellow")
            spinner = SpinnerProgress("Training model")
            spinner.start()
            
            # Train model
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X_train_tfidf, y_train)
            
            spinner.stop(" Model trained successfully")
            
            self.print_colored("[5/5] Evaluating model performance...", "yellow")
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_tfidf)
            y_test_pred = self.model.predict(X_test_tfidf)
            
            # Calculate metrics
            self.training_accuracy = accuracy_score(y_train, y_train_pred)
            self.test_accuracy = accuracy_score(y_test, y_test_pred)
            self.classification_rep = classification_report(y_test, y_test_pred, 
                                                           target_names=['FAKE', 'REAL'])
            self.conf_matrix = confusion_matrix(y_test, y_test_pred)
            
            self.model_trained = True
            
            self.print_colored(f"\n Training Accuracy: {self.training_accuracy*100:.2f}%", "green")
            self.print_colored(f" Testing Accuracy: {self.test_accuracy*100:.2f}%", "green")
            
            return True
            
        except Exception as e:
            self.print_colored(f"\n Error in model training: {str(e)}", "red")
            return False
    
    def predict_news(self, text, show_confidence=True):
        """Predict if news is fake or real"""
        if not self.model_trained:
            self.print_colored(" Model not trained yet!", "red")
            return None
        
        try:
            cleaned_text = self.clean_text(text)
            text_tfidf = self.vectorizer.transform([cleaned_text])
            prediction = self.model.predict(text_tfidf)[0]
            confidence = self.model.predict_proba(text_tfidf)[0]
            
            label = "REAL" if prediction == 1 else "FAKE"
            conf_score = confidence[prediction] * 100
            
            if show_confidence:
                color = "green" if label == "REAL" else "red"
                self.print_colored(f"\n{'='*60}", "cyan")
                self.print_colored(f"Prediction: {label} NEWS", color)
                self.print_colored(f"Confidence: {conf_score:.2f}%", "yellow")
                self.print_colored(f"{'='*60}", "cyan")
            
            return label, conf_score
            
        except Exception as e:
            self.print_colored(f" Error in prediction: {str(e)}", "red")
            return None
    
    def save_model(self, filename='fake_news_model.pkl'):
        """Save trained model and vectorizer"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'training_accuracy': self.training_accuracy,
                    'test_accuracy': self.test_accuracy
                }, f)
            self.print_colored(f"✓ Model saved to {filename}", "green")
            return True
        except Exception as e:
            self.print_colored(f" Error saving model: {str(e)}", "red")
            return False
    
    def load_model(self, filename='fake_news_model.pkl'):
        """Load trained model and vectorizer"""
        try:
            if not os.path.exists(filename):
                return False
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.training_accuracy = data.get('training_accuracy', 0)
                self.test_accuracy = data.get('test_accuracy', 0)
                self.model_trained = True
            
            self.print_colored(f" Model loaded from {filename}", "green")
            return True
        except Exception as e:
            self.print_colored(f" Error loading model: {str(e)}", "red")
            return False
    
    def generate_report(self, filename='classification_report.txt'):
        """Generate detailed text report"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report = f"""
{'='*70}
                    FAKE NEWS CLASSIFICATION REPORT
{'='*70}

Generated: {timestamp}

MODEL PERFORMANCE METRICS
{'='*70}

Training Accuracy: {self.training_accuracy*100:.2f}%
Testing Accuracy:  {self.test_accuracy*100:.2f}%

CLASSIFICATION REPORT
{'='*70}
{self.classification_rep}

CONFUSION MATRIX
{'='*70}
                Predicted FAKE    Predicted REAL
Actual FAKE     {self.conf_matrix[0][0]:>15}    {self.conf_matrix[0][1]:>15}
Actual REAL     {self.conf_matrix[1][0]:>15}    {self.conf_matrix[1][1]:>15}

INTERPRETATION
{'='*70}
True Negatives (TN):  {self.conf_matrix[0][0]} - Correctly identified FAKE news
False Positives (FP): {self.conf_matrix[0][1]} - FAKE news incorrectly labeled as REAL
False Negatives (FN): {self.conf_matrix[1][0]} - REAL news incorrectly labeled as FAKE
True Positives (TP):  {self.conf_matrix[1][1]} - Correctly identified REAL news

MODEL DETAILS
{'='*70}
Algorithm: Logistic Regression
Vectorization: TF-IDF (max_features=5000)
Features: Bag of words with term frequency-inverse document frequency

{'='*70}
            """
            
            with open(filename, 'w') as f:
                f.write(report)
            
            self.print_colored(f"\n Report saved to {filename}", "green")
            return True
            
        except Exception as e:
            self.print_colored(f" Error generating report: {str(e)}", "red")
            return False
    
    def generate_visualizations(self):
        """Generate word clouds and confusion matrix visualization"""
        if not VISUALIZATION_AVAILABLE:
            self.print_colored(" Visualization libraries not available", "yellow")
            return
        
        try:
            self.print_colored("\n Generating visualizations...", "cyan")
            
            # Confusion Matrix Plot
            plt.figure(figsize=(8, 6))
            plt.imshow(self.conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['FAKE', 'REAL'])
            plt.yticks(tick_marks, ['FAKE', 'REAL'])
            
            # Add text annotations
            thresh = self.conf_matrix.max() / 2.
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, format(self.conf_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if self.conf_matrix[i, j] > thresh else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.print_colored(" Confusion matrix saved to confusion_matrix.png", "green")
            
        except Exception as e:
            self.print_colored(f" Error generating visualizations: {str(e)}", "red")


def main():
    classifier = FakeNewsClassifier()
    classifier.print_banner()
    
    while True:
        print("\n" + "="*60)
        classifier.print_colored("MAIN MENU", "cyan")
        print("="*60)
        print("1. Train New Model")
        print("2. Load Existing Model")
        print("3. Test with Custom News Article")
        print("4. Batch Prediction from File")
        print("5. View Model Performance")
        print("6. Generate Reports")
        print("7. Generate Visualizations")
        print("8. Save Current Model")
        print("9. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            X_train, X_test, y_train, y_test = classifier.load_and_prepare_data()
            if X_train is not None:
                if classifier.train_model(X_train, X_test, y_train, y_test):
                    classifier.print_colored("\n Model training completed successfully!", "green")
                    classifier.save_model()
        
        elif choice == '2':
            if classifier.load_model():
                classifier.print_colored(f"Model loaded with {classifier.test_accuracy*100:.2f}% test accuracy", "green")
            else:
                classifier.print_colored("No saved model found. Please train a new model first.", "yellow")
        
        elif choice == '3':
            if not classifier.model_trained:
                classifier.print_colored(" Please train or load a model first!", "yellow")
                continue
            
            print("\n" + "="*60)
            classifier.print_colored("CUSTOM NEWS ARTICLE TESTING", "cyan")
            print("="*60)
            print("Enter news article (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            text = " ".join(lines)
            if text.strip():
                classifier.predict_news(text)
            else:
                classifier.print_colored("No text entered!", "red")
        
        elif choice == '4':
            if not classifier.model_trained:
                classifier.print_colored(" Please train or load a model first!", "yellow")
                continue
            
            filename = input("\nEnter filename with news articles (one per line): ").strip()
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    articles = f.readlines()
                
                results = []
                classifier.print_colored(f"\nProcessing {len(articles)} articles...", "cyan")
                
                for idx, article in enumerate(articles, 1):
                    if article.strip():
                        label, conf = classifier.predict_news(article, show_confidence=False)
                        results.append(f"Article {idx}: {label} ({conf:.2f}% confidence)")
                        print(f"  {idx}. {label} ({conf:.2f}%)")
                
                # Save batch results
                output_file = 'batch_predictions.txt'
                with open(output_file, 'w') as f:
                    f.write("\n".join(results))
                
                classifier.print_colored(f"\n Results saved to {output_file}", "green")
                
            except FileNotFoundError:
                classifier.print_colored(f" File '{filename}' not found!", "red")
            except Exception as e:
                classifier.print_colored(f" Error: {str(e)}", "red")
        
        elif choice == '5':
            if not classifier.model_trained:
                classifier.print_colored(" Please train or load a model first!", "yellow")
                continue
            
            print("\n" + "="*60)
            classifier.print_colored("MODEL PERFORMANCE METRICS", "cyan")
            print("="*60)
            classifier.print_colored(f"Training Accuracy: {classifier.training_accuracy*100:.2f}%", "green")
            classifier.print_colored(f"Testing Accuracy: {classifier.test_accuracy*100:.2f}%", "green")
            
            if classifier.classification_rep:
                print("\nDetailed Classification Report:")
                print(classifier.classification_rep)
        
        elif choice == '6':
            if not classifier.model_trained:
                classifier.print_colored(" Please train or load a model first!", "yellow")
                continue
            
            classifier.generate_report()
        
        elif choice == '7':
            if not classifier.model_trained:
                classifier.print_colored(" Please train or load a model first!", "yellow")
                continue
            
            classifier.generate_visualizations()
        
        elif choice == '8':
            if not classifier.model_trained:
                classifier.print_colored(" No model to save!", "yellow")
                continue
            
            filename = input("\nEnter filename to save model (default: fake_news_model.pkl): ").strip()
            if not filename:
                filename = 'fake_news_model.pkl'
            classifier.save_model(filename)
        
        elif choice == '9':
            classifier.print_colored("\nThank you for using Fake News Detection System!", "cyan")
            classifier.print_colored("Stay informed, stay safe! ", "green")
            break
        
        else:
            classifier.print_colored(" Invalid choice! Please enter 1-9.", "red")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n Fatal error: {str(e)}")
        print("Please report this issue if it persists.")