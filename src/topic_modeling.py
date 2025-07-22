"""
Topic modeling module using LDA and other topic modeling algorithms.
"""

import numpy as np
import pandas as pd
from gensim import models
from gensim.models import LdaModel, LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import HdpModel
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

class TopicModeler:
    """Topic modeling using LDA and other algorithms."""
    
    def __init__(self, n_topics=10, algorithm='lda', random_state=42):
        self.n_topics = n_topics
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.dictionary = None
        self.corpus = None
    
    def fit_lda(self, corpus, dictionary, num_topics=None, **kwargs):
        """Fit LDA model to corpus."""
        if num_topics is None:
            num_topics = self.n_topics
        
        print(f"Training LDA model with {num_topics} topics...")
        
        # Default parameters
        default_params = {
            'num_topics': num_topics,
            'id2word': dictionary,
            'corpus': corpus,
            'random_state': self.random_state,
            'chunksize': 100,
            'passes': 10,
            'alpha': 'auto',
            'per_word_topics': True
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        
        # Train model
        self.model = LdaModel(**default_params)
        self.dictionary = dictionary
        self.corpus = corpus
        
        print("LDA training completed!")
        return self.model
    
    def fit_lda_multicore(self, corpus, dictionary, num_topics=None, workers=4, **kwargs):
        """Fit LDA model using multiple cores."""
        if num_topics is None:
            num_topics = self.n_topics
        
        print(f"Training LDA model with {num_topics} topics using {workers} cores...")
        
        default_params = {
            'corpus': corpus,
            'id2word': dictionary,
            'num_topics': num_topics,
            'random_state': self.random_state,
            'chunksize': 100,
            'passes': 10,
            'alpha': 'auto',
            'per_word_topics': True,
            'workers': workers
        }
        
        default_params.update(kwargs)
        
        self.model = LdaMulticore(**default_params)
        self.dictionary = dictionary
        self.corpus = corpus
        
        print("LDA training completed!")
        return self.model
    
    def fit_hdp(self, corpus, dictionary, **kwargs):
        """Fit Hierarchical Dirichlet Process model."""
        print("Training HDP model...")
        
        default_params = {
            'corpus': corpus,
            'id2word': dictionary,
            'random_state': self.random_state
        }
        
        default_params.update(kwargs)
        
        self.model = HdpModel(**default_params)
        self.dictionary = dictionary
        self.corpus = corpus
        
        print("HDP training completed!")
        return self.model
    
    def get_topics(self, num_words=10):
        """Get topics with top words."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        topics = []
        for topic_id in range(self.model.num_topics):
            topic_words = self.model.show_topic(topic_id, topn=num_words)
            topics.append({
                'topic_id': topic_id,
                'words': topic_words,
                'words_only': [word for word, _ in topic_words],
                'weights': [weight for _, weight in topic_words]
            })
        
        return topics
    
    def get_document_topics(self, corpus=None):
        """Get topic distribution for documents."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        if corpus is None:
            corpus = self.corpus
        
        doc_topics = []
        for doc in corpus:
            topic_dist = self.model.get_document_topics(doc, minimum_probability=0.01)
            doc_topics.append(topic_dist)
        
        return doc_topics
    
    def calculate_coherence(self, model=None, texts=None, dictionary=None, coherence='c_v'):
        """Calculate topic coherence score."""
        if model is None:
            model = self.model
        if dictionary is None:
            dictionary = self.dictionary
        
        if model is None or dictionary is None:
            raise ValueError("Model and dictionary must be provided.")
        
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence=coherence
        )
        
        return coherence_model.get_coherence()
    
    def calculate_perplexity(self, corpus=None):
        """Calculate model perplexity."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        if corpus is None:
            corpus = self.corpus
        
        return self.model.log_perplexity(corpus)
    
    def get_topic_terms_matrix(self, num_words=20):
        """Get topic-term matrix."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        topics = self.get_topics(num_words)
        
        # Create matrix
        all_words = set()
        for topic in topics:
            all_words.update(topic['words_only'])
        
        word_list = sorted(list(all_words))
        matrix = np.zeros((len(topics), len(word_list)))
        
        for i, topic in enumerate(topics):
            for word, weight in topic['words']:
                if word in word_list:
                    j = word_list.index(word)
                    matrix[i, j] = weight
        
        return matrix, word_list
    
    def create_topic_wordcloud(self, topic_id, save_path=None):
        """Create word cloud for a specific topic."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        # Get topic words and weights
        topic_words = dict(self.model.show_topic(topic_id, topn=50))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=50,
            colormap='viridis'
        ).generate_from_frequencies(topic_words)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_id} Word Cloud', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        return wordcloud
    
    def plot_topic_distribution(self, save_path=None):
        """Plot topic distribution across documents."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        doc_topics = self.get_document_topics()
        
        # Create topic distribution matrix
        topic_dist = np.zeros((len(doc_topics), self.model.num_topics))
        
        for i, doc_topic in enumerate(doc_topics):
            for topic_id, prob in doc_topic:
                topic_dist[i, topic_id] = prob
        
        # Calculate average topic probabilities
        avg_topic_prob = np.mean(topic_dist, axis=0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(self.model.num_topics), avg_topic_prob, 
                      color='skyblue', alpha=0.7)
        plt.xlabel('Topic ID')
        plt.ylabel('Average Probability')
        plt.title('Average Topic Distribution Across Documents')
        plt.xticks(range(self.model.num_topics))
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{avg_topic_prob[i]:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_pyldavis_visualization(self, save_path=None):
        """Create interactive pyLDAvis visualization."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        print("Creating pyLDAvis visualization...")
        
        # Prepare visualization
        vis = gensimvis.prepare(self.model, self.corpus, self.dictionary)
        
        if save_path:
            pyLDAvis.save_html(vis, save_path)
            print(f"Interactive visualization saved to {save_path}")
        else:
            return vis
        
        return vis

class TopicOptimizer:
    """Optimize number of topics using various metrics."""
    
    def __init__(self, max_topics=20, random_state=42):
        self.max_topics = max_topics
        self.random_state = random_state
    
    def optimize_topic_number(self, corpus, dictionary, texts, topic_range=None):
        """Find optimal number of topics using coherence and perplexity."""
        if topic_range is None:
            topic_range = range(2, min(self.max_topics + 1, 21))
        
        coherence_scores = []
        perplexity_scores = []
        models = []
        
        print("Optimizing number of topics...")
        
        for num_topics in topic_range:
            print(f"Testing {num_topics} topics...")
            
            # Train model
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=self.random_state,
                passes=10,
                alpha='auto'
            )
            
            # Calculate coherence
            coherence_model = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)
            
            # Calculate perplexity
            perplexity = model.log_perplexity(corpus)
            perplexity_scores.append(perplexity)
            
            models.append(model)
        
        return {
            'topic_range': list(topic_range),
            'coherence_scores': coherence_scores,
            'perplexity_scores': perplexity_scores,
            'models': models
        }
    
    def plot_optimization_results(self, optimization_results, save_path=None):
        """Plot coherence and perplexity scores."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        topic_range = optimization_results['topic_range']
        coherence_scores = optimization_results['coherence_scores']
        perplexity_scores = optimization_results['perplexity_scores']
        
        # Coherence scores
        ax1.plot(topic_range, coherence_scores, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Coherence Score')
        ax1.set_title('Topic Coherence vs Number of Topics')
        ax1.grid(True, alpha=0.3)
        
        # Mark best coherence score
        best_coherence_idx = np.argmax(coherence_scores)
        best_coherence_topics = topic_range[best_coherence_idx]
        ax1.axvline(x=best_coherence_topics, color='red', linestyle='--', alpha=0.7)
        ax1.text(best_coherence_topics, max(coherence_scores), 
                f'Best: {best_coherence_topics}', ha='center', va='bottom')
        
        # Perplexity scores
        ax2.plot(topic_range, perplexity_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Topics')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Perplexity vs Number of Topics')
        ax2.grid(True, alpha=0.3)
        
        # Mark best perplexity score (lowest)
        best_perplexity_idx = np.argmin(perplexity_scores)
        best_perplexity_topics = topic_range[best_perplexity_idx]
        ax2.axvline(x=best_perplexity_topics, color='red', linestyle='--', alpha=0.7)
        ax2.text(best_perplexity_topics, min(perplexity_scores), 
                f'Best: {best_perplexity_topics}', ha='center', va='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        return fig