"""
Document clustering module using K-means and other clustering algorithms.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class DocumentClusterer:
    """Document clustering using various algorithms."""
    
    def __init__(self, n_clusters=8, algorithm='kmeans', random_state=42):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize clustering model based on algorithm."""
        if self.algorithm == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
        elif self.algorithm == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
        elif self.algorithm == 'dbscan':
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def fit(self, X):
        """Fit clustering model to data."""
        print(f"Fitting {self.algorithm} clustering model...")
        
        # For sparse matrices, convert to dense if needed
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Fit model
        self.model.fit(X_dense)
        self.labels_ = self.model.labels_
        
        print(f"Clustering completed. Found {len(np.unique(self.labels_))} clusters.")
        return self
    
    def fit_predict(self, X):
        """Fit model and return cluster labels."""
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_dense)
        else:
            # For algorithms without predict method, use fit_predict
            return self.model.fit_predict(X_dense)
    
    def get_cluster_centers(self):
        """Get cluster centers (for K-means)."""
        if self.algorithm == 'kmeans' and self.model is not None:
            return self.model.cluster_centers_
        else:
            return None
    
    def get_cluster_statistics(self, X):
        """Get statistics for each cluster."""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet.")
        
        stats = {}
        unique_labels = np.unique(self.labels_)
        
        for label in unique_labels:
            cluster_mask = self.labels_ == label
            cluster_size = np.sum(cluster_mask)
            
            if hasattr(X, 'toarray'):
                cluster_data = X[cluster_mask].toarray()
            else:
                cluster_data = X[cluster_mask]
            
            stats[label] = {
                'size': cluster_size,
                'percentage': (cluster_size / len(self.labels_)) * 100,
                'mean_features': np.mean(cluster_data, axis=0),
                'std_features': np.std(cluster_data, axis=0)
            }
        
        return stats
    
    def get_top_terms_per_cluster(self, X, feature_names, top_n=10):
        """Get top terms for each cluster based on TF-IDF scores."""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet.")
        
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        cluster_terms = {}
        unique_labels = np.unique(self.labels_)
        
        for label in unique_labels:
            cluster_mask = self.labels_ == label
            cluster_data = X_dense[cluster_mask]
            
            # Calculate mean TF-IDF scores for this cluster
            mean_scores = np.mean(cluster_data, axis=0)
            
            # Get top terms
            top_indices = np.argsort(mean_scores)[-top_n:][::-1]
            top_terms = [(feature_names[i], mean_scores[i]) for i in top_indices]
            
            cluster_terms[label] = top_terms
        
        return cluster_terms
    
    def evaluate_clustering(self, X, true_labels=None):
        """Evaluate clustering performance."""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet.")
        
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        metrics = {}
        
        # Silhouette score
        if len(np.unique(self.labels_)) > 1:
            metrics['silhouette_score'] = silhouette_score(X_dense, self.labels_)
        else:
            metrics['silhouette_score'] = -1
        
        # If true labels are available
        if true_labels is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, self.labels_)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, self.labels_)
        
        # Inertia (for K-means)
        if self.algorithm == 'kmeans':
            metrics['inertia'] = self.model.inertia_
        
        # Cluster distribution
        unique, counts = np.unique(self.labels_, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique, counts))
        metrics['n_clusters'] = len(unique)
        
        return metrics
    
    def plot_cluster_distribution(self, save_path=None):
        """Plot cluster size distribution."""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet.")
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, color='skyblue', alpha=0.7)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Documents')
        plt.title('Cluster Size Distribution')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

class ClusterOptimizer:
    """Optimize number of clusters using various methods."""
    
    def __init__(self, max_clusters=20, random_state=42):
        self.max_clusters = max_clusters
        self.random_state = random_state
    
    def elbow_method(self, X, cluster_range=None):
        """Find optimal number of clusters using elbow method."""
        if cluster_range is None:
            cluster_range = range(2, min(self.max_clusters + 1, len(X) // 2))
        
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        inertias = []
        silhouette_scores = []
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_dense)
            
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X_dense, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(-1)
        
        return {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
    
    def plot_optimization_curves(self, optimization_results, save_path=None):
        """Plot elbow curve and silhouette scores."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        cluster_range = optimization_results['cluster_range']
        inertias = optimization_results['inertias']
        silhouette_scores = optimization_results['silhouette_scores']
        
        # Elbow curve
        ax1.plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(cluster_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        return fig