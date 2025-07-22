"""
Text preprocessing module for document clustering and topic modeling.
"""

import re
import string
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Comprehensive text preprocessing for NLP tasks."""
    
    def __init__(self, language='english', use_spacy=True):
        self.language = language
        self.use_spacy = use_spacy
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Initialize spaCy if available
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
                self.nlp = None
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Count vectorizer for LDA
        self.count_vectorizer = CountVectorizer(
            max_features=10000,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
    
    def clean_text(self, text):
        """Basic text cleaning."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text using spaCy or NLTK."""
        if self.use_spacy and self.nlp:
            # Use spaCy for tokenization and lemmatization
            doc = self.nlp(text)
            tokens = [
                token.lemma_ for token in doc 
                if not token.is_stop 
                and not token.is_punct 
                and not token.is_space
                and len(token.text) > 2
            ]
        else:
            # Use NLTK for tokenization and lemmatization
            tokens = nltk.word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words 
                and token not in string.punctuation
                and len(token) > 2
            ]
        
        return tokens
    
    def preprocess_document(self, text):
        """Complete preprocessing pipeline for a single document."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        
        return tokens
    
    def preprocess_documents(self, documents):
        """Preprocess a list of documents."""
        print(f"Preprocessing {len(documents)} documents...")
        
        processed_docs = []
        for i, doc in enumerate(documents):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(documents)} documents")
            
            processed_doc = self.preprocess_document(doc)
            processed_docs.append(processed_doc)
        
        print("Preprocessing completed!")
        return processed_docs
    
    def create_tfidf_matrix(self, documents):
        """Create TF-IDF matrix from documents."""
        # Clean documents for TF-IDF
        cleaned_docs = [self.clean_text(doc) for doc in documents]
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_docs)
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Feature names sample: {self.tfidf_vectorizer.get_feature_names_out()[:10]}")
        
        return tfidf_matrix
    
    def create_bow_matrix(self, processed_documents):
        """Create Bag-of-Words matrix and dictionary for Gensim LDA."""
        # Create dictionary
        dictionary = corpora.Dictionary(processed_documents)
        
        # Filter extremes
        dictionary.filter_extremes(no_below=2, no_above=0.95)
        
        # Create corpus (bag-of-words representation)
        corpus = [dictionary.doc2bow(doc) for doc in processed_documents]
        
        print(f"Dictionary size: {len(dictionary)}")
        print(f"Corpus size: {len(corpus)}")
        
        return corpus, dictionary
    
    def get_feature_names(self):
        """Get TF-IDF feature names."""
        return self.tfidf_vectorizer.get_feature_names_out()
    
    def transform_new_documents(self, documents):
        """Transform new documents using fitted vectorizers."""
        # Clean documents
        cleaned_docs = [self.clean_text(doc) for doc in documents]
        
        # Transform with TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.transform(cleaned_docs)
        
        # Process for LDA
        processed_docs = [self.preprocess_document(doc) for doc in documents]
        
        return tfidf_matrix, processed_docs
    
    def get_preprocessing_stats(self, original_docs, processed_docs):
        """Get preprocessing statistics."""
        original_lengths = [len(doc.split()) for doc in original_docs]
        processed_lengths = [len(doc) for doc in processed_docs]
        
        stats = {
            'original_avg_length': sum(original_lengths) / len(original_lengths),
            'processed_avg_length': sum(processed_lengths) / len(processed_lengths),
            'vocabulary_reduction': 1 - (sum(processed_lengths) / sum(original_lengths)),
            'total_documents': len(original_docs)
        }
        
        return stats

def preprocess_custom_dataset(file_path, text_column=None):
    """Utility function to preprocess custom datasets."""
    preprocessor = TextPreprocessor()
    
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        if text_column:
            documents = df[text_column].tolist()
        else:
            documents = df.iloc[:, 0].tolist()  # First column
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = f.readlines()
    else:
        raise ValueError("Unsupported file format")
    
    # Preprocess
    processed_docs = preprocessor.preprocess_documents(documents)
    tfidf_matrix = preprocessor.create_tfidf_matrix(documents)
    bow_corpus, dictionary = preprocessor.create_bow_matrix(processed_docs)
    
    return {
        'original_documents': documents,
        'processed_documents': processed_docs,
        'tfidf_matrix': tfidf_matrix,
        'bow_corpus': bow_corpus,
        'dictionary': dictionary,
        'preprocessor': preprocessor
    }