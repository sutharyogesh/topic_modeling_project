"""
Comprehensive Document Clustering and Topic Modeling Project
Using 20 Newsgroups Dataset with K-means and LDA
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import nltk
import spacy
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from src.preprocessing import TextPreprocessor
from src.clustering import DocumentClusterer
from src.topic_modeling import TopicModeler
from src.visualization import Visualizer
from src.evaluation import ModelEvaluator

class DocumentAnalysisPipeline:
    """Main pipeline for document clustering and topic modeling analysis."""
    
    def __init__(self, n_clusters=8, n_topics=8, random_state=42):
        self.n_clusters = n_clusters
        self.n_topics = n_topics
        self.random_state = random_state
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.clusterer = DocumentClusterer(n_clusters=n_clusters, random_state=random_state)
        self.topic_modeler = TopicModeler(n_topics=n_topics, random_state=random_state)
        self.visualizer = Visualizer()
        self.evaluator = ModelEvaluator()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.tfidf_matrix = None
        self.bow_matrix = None
        self.cluster_labels = None
        self.lda_model = None
        self.dictionary = None
        self.labels = None
        self.label_names = None

    def load_data(self, dataset_name: str, custom_path: str = None) -> None:
        """Load raw text data into the pipeline."""
        if dataset_name.lower() == "20newsgroups":
            print("Loading 20newsgroups dataset...")
            
            # Define where the data should live
            data_home = Path(r"D:\scikit_learn_data") if not custom_path else Path(custom_path)
            
            try:
                # Try to load from local cache with download disabled
                newsgroups_train = fetch_20newsgroups(
                    subset="train",
                    data_home=str(data_home),
                    remove=("headers", "footers", "quotes"),
                    download_if_missing=False  # This prevents the 403 error
                )
                
                # Populate pipeline fields
                self.raw_data = {
                    'documents': newsgroups_train.data,
                    'targets': newsgroups_train.target,
                    'target_names': newsgroups_train.target_names
                }
                self.labels = newsgroups_train.target
                self.label_names = newsgroups_train.target_names
                
                print(f"✅ Loaded {len(self.raw_data['documents'])} documents from local cache")
                return
                
            except (FileNotFoundError, OSError) as e:
                # Try to download if missing
                try:
                    print("Local cache not found, attempting to download...")
                    newsgroups_train = fetch_20newsgroups(
                        subset="train",
                        data_home=str(data_home),
                        remove=("headers", "footers", "quotes"),
                        download_if_missing=True
                    )
                    
                    # Populate pipeline fields
                    self.raw_data = {
                        'documents': newsgroups_train.data,
                        'targets': newsgroups_train.target,
                        'target_names': newsgroups_train.target_names
                    }
                    self.labels = newsgroups_train.target
                    self.label_names = newsgroups_train.target_names
                    
                    print(f"✅ Downloaded and loaded {len(self.raw_data['documents'])} documents")
                    return
                    
                except Exception as download_error:
                    # Provide clear instructions if download fails
                    raise RuntimeError(
                        f"20Newsgroups dataset not found in {data_home} and download failed. "
                        "Please download '20news-bydate.tar.gz' manually and place it in "
                        f"{data_home}/20news_home/ or set a custom path."
                    ) from download_error
        
        # For other datasets, use custom loader
        elif custom_path:
            self.raw_data = self._load_custom_dataset(custom_path)
            print(f"Loaded {len(self.raw_data['documents'])} documents from custom dataset: ✅")
        else:
            raise ValueError("Unsupported dataset or missing custom_path.")

    def _load_custom_dataset(self, file_path):
        """Load custom dataset from file."""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Assume first column contains text
            documents = df.iloc[:, 0].tolist()
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = f.readlines()
        else:
            raise ValueError("Unsupported file format. Use .csv or .txt")

        return {
            'documents': documents,
            'targets': None,
            'target_names': None
        }
    
    def preprocess_data(self):
        """Preprocess text data."""
        print("Preprocessing documents...")
        
        # Clean and preprocess text
        self.processed_data = self.preprocessor.preprocess_documents(
            self.raw_data['documents']
        )
        
        # Create TF-IDF matrix
        print("Creating TF-IDF representation...")
        self.tfidf_matrix = self.preprocessor.create_tfidf_matrix(
            self.raw_data['documents']
        )
        
        # Create Bag-of-Words matrix for LDA
        print("Creating Bag-of-Words representation...")
        self.bow_matrix, self.dictionary = self.preprocessor.create_bow_matrix(
            self.processed_data
        )
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.dictionary)}")
        
    def perform_clustering(self):
        """Perform K-means clustering."""
        print(f"Performing K-means clustering with {self.n_clusters} clusters...")
        
        # Fit K-means
        self.cluster_labels = self.clusterer.fit_predict(self.tfidf_matrix)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.tfidf_matrix, self.cluster_labels)
        print(f"Average silhouette score: {silhouette_avg:.3f}")
        
        return self.cluster_labels, silhouette_avg
    
    def perform_topic_modeling(self):
        """Perform LDA topic modeling."""
        print(f"Performing LDA topic modeling with {self.n_topics} topics...")
        
        # Fit LDA model
        self.lda_model = self.topic_modeler.fit_lda(
            self.bow_matrix, 
            self.dictionary,
            num_topics=self.n_topics
        )
        
        # Calculate coherence score
        coherence_score = self.topic_modeler.calculate_coherence(
            self.lda_model, 
            self.processed_data, 
            self.dictionary
        )
        print(f"Coherence score: {coherence_score:.3f}")
        
        return self.lda_model, coherence_score
    
    def generate_visualizations(self, output_dir='visualizations'):
        """Generate all visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating visualizations in {output_dir}/...")
        
        # 1. Cluster visualization with t-SNE
        print("Creating t-SNE cluster visualization...")
        self.visualizer.plot_tsne_clusters(
            self.tfidf_matrix, 
            self.cluster_labels,
            save_path=f"{output_dir}/tsne_clusters.html"
        )
        
        # 2. PCA visualization
        print("Creating PCA visualization...")
        self.visualizer.plot_pca_clusters(
            self.tfidf_matrix,
            self.cluster_labels,
            save_path=f"{output_dir}/pca_clusters.html"
        )
        
        # 3. Topic word clouds
        print("Creating topic word clouds...")
        self.visualizer.create_topic_wordclouds(
            self.lda_model,
            save_dir=f"{output_dir}/wordclouds"
        )
        
        # 4. Topic distribution
        print("Creating topic distribution plot...")
        self.visualizer.plot_topic_distribution(
            self.lda_model,
            self.bow_matrix,
            save_path=f"{output_dir}/topic_distribution.html"
        )
        
        # 5. Interactive LDA visualization
        print("Creating interactive LDA visualization...")
        self.visualizer.create_pyldavis(
            self.lda_model,
            self.bow_matrix,
            self.dictionary,
            save_path=f"{output_dir}/lda_interactive.html"
        )
        
        # 6. Model comparison
        print("Creating model comparison...")
        self.visualizer.plot_model_comparison(
            self.cluster_labels,
            self.lda_model,
            self.bow_matrix,
            save_path=f"{output_dir}/model_comparison.html"
        )
    
    def evaluate_models(self):
        """Evaluate both clustering and topic modeling."""
        print("Evaluating models...")
        
        # Clustering evaluation
        clustering_metrics = self.evaluator.evaluate_clustering(
            self.tfidf_matrix,
            self.cluster_labels,
            self.raw_data.get('targets')
        )
        
        # Topic modeling evaluation
        topic_metrics = self.evaluator.evaluate_topic_model(
            self.lda_model,
            self.processed_data,
            self.dictionary,
            self.bow_matrix
        )
        
        # Print results
        print("\n" + "="*50)
        print("CLUSTERING EVALUATION")
        print("="*50)
        for metric, value in clustering_metrics.items():
            print(f"{metric}: {value:.3f}")
        
        print("\n" + "="*50)
        print("TOPIC MODELING EVALUATION")
        print("="*50)
        for metric, value in topic_metrics.items():
            print(f"{metric}: {value:.3f}")
        
        return clustering_metrics, topic_metrics
    
    def save_models(self, output_dir='models'):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save K-means model
        with open(f"{output_dir}/kmeans_model.pkl", 'wb') as f:
            pickle.dump(self.clusterer.model, f)
        
        # Save LDA model
        self.lda_model.save(f"{output_dir}/lda_model")
        
        # Save dictionary
        self.dictionary.save(f"{output_dir}/dictionary.dict")
        
        # Save preprocessed data
        with open(f"{output_dir}/processed_data.pkl", 'wb') as f:
            pickle.dump({
                'tfidf_matrix': self.tfidf_matrix,
                'bow_matrix': self.bow_matrix,
                'cluster_labels': self.cluster_labels,
                'processed_documents': self.processed_data
            }, f)
        
        print(f"Models saved to {output_dir}/")
    
    def run_full_pipeline(self, dataset_name='20newsgroups', custom_path=None):
        """Run the complete analysis pipeline."""
        print("Starting Document Analysis Pipeline...")
        print("="*60)
        
        try:
            # 1. Load data
            self.load_data(dataset_name, custom_path)
            
            # 2. Preprocess
            self.preprocess_data()
            
            # 3. Clustering
            cluster_labels, silhouette_score = self.perform_clustering()
            
            # 4. Topic modeling
            lda_model, coherence_score = self.perform_topic_modeling()
            
            # 5. Evaluation
            clustering_metrics, topic_metrics = self.evaluate_models()
            
            # 6. Visualizations
            self.generate_visualizations()
            
            # 7. Save models
            self.save_models()
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Clustering Silhouette Score: {silhouette_score:.3f}")
            print(f"Topic Coherence Score: {coherence_score:.3f}")
            print("Check 'visualizations/' for plots and 'models/' for saved models")
            
            return {
                'clustering_metrics': clustering_metrics,
                'topic_metrics': topic_metrics,
                'silhouette_score': silhouette_score,
                'coherence_score': coherence_score
            }
            
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            raise

def main():
    """Main execution function."""
    # Initialize pipeline
    pipeline = DocumentAnalysisPipeline(
        n_clusters=8,
        n_topics=8,
        random_state=42
    )
    
    # Run full analysis
    results = pipeline.run_full_pipeline()
    
    # Optional: Run with different parameters
    print("\n" + "="*60)
    print("RUNNING PARAMETER COMPARISON...")
    print("="*60)
    
    # Test different numbers of clusters/topics
    for n in [5, 10, 15]:
        print(f"\nTesting with {n} clusters/topics...")
        test_pipeline = DocumentAnalysisPipeline(
            n_clusters=n,
            n_topics=n,
            random_state=42
        )
        test_pipeline.raw_data = pipeline.raw_data
        test_pipeline.processed_data = pipeline.processed_data
        test_pipeline.tfidf_matrix = pipeline.tfidf_matrix
        test_pipeline.bow_matrix = pipeline.bow_matrix
        test_pipeline.dictionary = pipeline.dictionary
        
        # Quick evaluation
        cluster_labels, sil_score = test_pipeline.perform_clustering()
        lda_model, coh_score = test_pipeline.perform_topic_modeling()
        
        print(f"  Silhouette Score: {sil_score:.3f}")
        print(f"  Coherence Score: {coh_score:.3f}")

if __name__ == "__main__":
    main()