"""
Model evaluation module for clustering and topic modeling.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)
from gensim.models.coherencemodel import CoherenceModel

import pandas as pd

class ModelEvaluator:
    """Comprehensive evaluation for clustering and topic modeling."""
    
    def __init__(self):
        pass
    
    def evaluate_clustering(self, X, cluster_labels, true_labels=None):
        """Evaluate clustering performance with multiple metrics."""
        metrics = {}
        
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Internal metrics (don't require true labels)
        if len(np.unique(cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(X_dense, cluster_labels)
        else:
            metrics['silhouette_score'] = -1
        
        # Cluster statistics
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        metrics['n_clusters'] = len(unique_labels)
        metrics['cluster_sizes'] = dict(zip(unique_labels, counts))
        metrics['largest_cluster_ratio'] = max(counts) / len(cluster_labels)
        metrics['smallest_cluster_ratio'] = min(counts) / len(cluster_labels)
        metrics['cluster_balance'] = min(counts) / max(counts)
        
        # External metrics (require true labels)
        if true_labels is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, cluster_labels)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, cluster_labels)
            metrics['homogeneity_score'] = homogeneity_score(true_labels, cluster_labels)
            metrics['completeness_score'] = completeness_score(true_labels, cluster_labels)
            metrics['v_measure_score'] = v_measure_score(true_labels, cluster_labels)
        
        return metrics
    
    def evaluate_topic_model(self, lda_model, texts, dictionary, corpus):
        """Evaluate topic model performance."""
        metrics = {}
        
        # Coherence scores
        try:
            # C_v coherence (most common)
            coherence_cv = CoherenceModel(
                model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v'
            ).get_coherence()
            metrics['coherence_c_v'] = coherence_cv
            
            # U_mass coherence
            coherence_umass = CoherenceModel(
                model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass'
            ).get_coherence()
            metrics['coherence_u_mass'] = coherence_umass
            
        except Exception as e:
            print(f"Error calculating coherence: {e}")
            metrics['coherence_c_v'] = 0
            metrics['coherence_u_mass'] = 0
        
        # Perplexity
        try:
            metrics['perplexity'] = lda_model.log_perplexity(corpus)
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            metrics['perplexity'] = float('inf')
        
        # Topic diversity metrics
        topics = lda_model.show_topics(num_topics=-1, num_words=10, formatted=False)
        
        # Calculate topic diversity (percentage of unique words across topics)
        all_words = set()
        topic_words = []
        
        for topic_id, words in topics:
            words_only = [word for word, _ in words]
            topic_words.append(words_only)
            all_words.update(words_only)
        
        total_words = sum(len(words) for words in topic_words)
        unique_words = len(all_words)
        metrics['topic_diversity'] = unique_words / total_words if total_words > 0 else 0
        
        # Average topic coherence per topic
        topic_coherences = []
        for i in range(lda_model.num_topics):
            try:
                topic_coherence = CoherenceModel(
                    topics=[lda_model.show_topic(i, topn=10)],
                    texts=texts,
                    dictionary=dictionary,
                    coherence='c_v'
                ).get_coherence()
                topic_coherences.append(topic_coherence)
            except:
                topic_coherences.append(0)
        
        metrics['avg_topic_coherence'] = np.mean(topic_coherences)
        metrics['std_topic_coherence'] = np.std(topic_coherences)
        metrics['min_topic_coherence'] = np.min(topic_coherences)
        metrics['max_topic_coherence'] = np.max(topic_coherences)
        
        # Topic size distribution
        doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
        topic_counts = np.zeros(lda_model.num_topics)
        
        for doc_topic in doc_topics:
            if doc_topic:  # If document has topics
                dominant_topic = max(doc_topic, key=lambda x: x[1])[0]
                topic_counts[dominant_topic] += 1
        
        if np.sum(topic_counts) > 0:
            metrics['topic_balance'] = np.min(topic_counts) / np.max(topic_counts)
            metrics['largest_topic_ratio'] = np.max(topic_counts) / np.sum(topic_counts)
        else:
            metrics['topic_balance'] = 0
            metrics['largest_topic_ratio'] = 0
        
        return metrics
    
    def compare_models(self, clustering_metrics, topic_metrics):
        """Compare clustering and topic modeling results."""
        comparison = {
            'clustering': clustering_metrics,
            'topic_modeling': topic_metrics
        }
        
        # Create summary scores
        clustering_score = clustering_metrics.get('silhouette_score', 0)
        topic_score = topic_metrics.get('coherence_c_v', 0)
        
        comparison['summary'] = {
            'clustering_performance': 'Good' if clustering_score > 0.5 else 'Fair' if clustering_score > 0.2 else 'Poor',
            'topic_modeling_performance': 'Good' if topic_score > 0.5 else 'Fair' if topic_score > 0.3 else 'Poor',
            'better_method': 'Clustering' if clustering_score > topic_score else 'Topic Modeling'
        }
        
        return comparison
    
    def create_evaluation_report(self, clustering_metrics, topic_metrics, save_path=None):
        """Create a comprehensive evaluation report."""
        report = {
            'Clustering Evaluation': clustering_metrics,
            'Topic Modeling Evaluation': topic_metrics,
            'Model Comparison': self.compare_models(clustering_metrics, topic_metrics)
        }
        
        # Create formatted report
        report_text = "="*60 + "\n"
        report_text += "MODEL EVALUATION REPORT\n"
        report_text += "="*60 + "\n\n"
        
        # Clustering section
        report_text += "CLUSTERING RESULTS\n"
        report_text += "-"*30 + "\n"
        for metric, value in clustering_metrics.items():
            if isinstance(value, dict):
                report_text += f"{metric}:\n"
                for k, v in value.items():
                    report_text += f"  {k}: {v}\n"
            else:
                report_text += f"{metric}: {value:.4f}\n"
        
        report_text += "\n"
        
        # Topic modeling section
        report_text += "TOPIC MODELING RESULTS\n"
        report_text += "-"*30 + "\n"
        for metric, value in topic_metrics.items():
            if isinstance(value, dict):
                report_text += f"{metric}:\n"
                for k, v in value.items():
                    report_text += f"  {k}: {v}\n"
            else:
                report_text += f"{metric}: {value:.4f}\n"
        
        report_text += "\n"
        
        # Summary section
        comparison = self.compare_models(clustering_metrics, topic_metrics)
        report_text += "SUMMARY\n"
        report_text += "-"*30 + "\n"
        for metric, value in comparison['summary'].items():
            report_text += f"{metric}: {value}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")
        
        return report_text
    
    def cross_validate_clustering(self, X, clusterer, cv_folds=5):
        """Perform cross-validation for clustering."""
        # This is a simplified version - true clustering CV is complex
        # because cluster labels are not consistent across folds
        
        n_samples = X.shape[0]
        fold_size = n_samples // cv_folds
        scores = []
        
        for i in range(cv_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < cv_folds - 1 else n_samples
            
            # Use subset for training
            X_subset = X[start_idx:end_idx]
            
            if hasattr(X_subset, 'toarray'):
                X_subset_dense = X_subset.toarray()
            else:
                X_subset_dense = X_subset
            
            # Fit and evaluate
            labels = clusterer.fit_predict(X_subset_dense)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_subset_dense, labels)
                scores.append(score)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }
    
    def stability_analysis(self, X, clusterer, n_runs=10):
        """Analyze clustering stability across multiple runs."""
        all_labels = []
        scores = []
        
        for run in range(n_runs):
            # Set different random state for each run
            clusterer.random_state = run
            clusterer._initialize_model()
            
            if hasattr(X, 'toarray'):
                X_dense = X.toarray()
            else:
                X_dense = X
            
            labels = clusterer.fit_predict(X_dense)
            all_labels.append(labels)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_dense, labels)
                scores.append(score)
        
        # Calculate stability (average ARI between runs)
        stability_scores = []
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                stability_scores.append(ari)
        
        return {
            'mean_silhouette': np.mean(scores),
            'std_silhouette': np.std(scores),
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'all_scores': scores,
            'stability_scores': stability_scores
        }