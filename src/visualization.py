"""
Visualization module for document clustering and topic modeling results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os

class Visualizer:
    """Comprehensive visualization for clustering and topic modeling."""
    
    def __init__(self, style='seaborn', figsize=(12, 8)):
        self.style = style
        self.figsize = figsize
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color palettes
        self.cluster_colors = px.colors.qualitative.Set3
        self.topic_colors = px.colors.qualitative.Pastel
    
    def plot_tsne_clusters(self, X, labels, save_path=None, perplexity=30):
        """Create t-SNE visualization of clusters."""
        print("Creating t-SNE visualization...")
        
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_dense)
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': X_tsne[:, 0],
            'y': X_tsne[:, 1],
            'cluster': labels.astype(str)
        })
        
        # Create interactive plot
        fig = px.scatter(
            df, x='x', y='y', color='cluster',
            title='t-SNE Visualization of Document Clusters',
            labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
            color_discrete_sequence=self.cluster_colors
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"t-SNE plot saved to {save_path}")
        
        return fig
    
    def plot_pca_clusters(self, X, labels, save_path=None):
        """Create PCA visualization of clusters."""
        print("Creating PCA visualization...")
        
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_dense)
        
        # Create DataFrame
        df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'cluster': labels.astype(str)
        })
        
        # Create interactive plot
        fig = px.scatter(
            df, x='PC1', y='PC2', color='cluster',
            title=f'PCA Visualization of Document Clusters<br>Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}',
            color_discrete_sequence=self.cluster_colors
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"PCA plot saved to {save_path}")
        
        return fig
    
    def create_topic_wordclouds(self, lda_model, save_dir=None, max_words=50):
        """Create word clouds for all topics."""
        print("Creating topic word clouds...")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        wordclouds = []
        
        for topic_id in range(lda_model.num_topics):
            # Get topic words and weights
            topic_words = dict(lda_model.show_topic(topic_id, topn=max_words))
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=max_words,
                colormap='viridis',
                relative_scaling=0.5
            ).generate_from_frequencies(topic_words)
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic_id} Word Cloud', fontsize=16, fontweight='bold')
            
            if save_dir:
                save_path = os.path.join(save_dir, f'topic_{topic_id}_wordcloud.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Word cloud for topic {topic_id} saved to {save_path}")
            
            plt.close()
            wordclouds.append(wordcloud)
        
        return wordclouds
    
    def plot_topic_distribution(self, lda_model, corpus, save_path=None):
        """Plot topic distribution across documents."""
        print("Creating topic distribution plot...")
        
        # Get document-topic distributions
        doc_topics = []
        for doc in corpus:
            topic_dist = lda_model.get_document_topics(doc, minimum_probability=0.01)
            doc_topics.append(topic_dist)
        
        # Create topic distribution matrix
        topic_matrix = np.zeros((len(doc_topics), lda_model.num_topics))
        
        for i, doc_topic in enumerate(doc_topics):
            for topic_id, prob in doc_topic:
                topic_matrix[i, topic_id] = prob
        
        # Calculate statistics
        topic_means = np.mean(topic_matrix, axis=0)
        topic_stds = np.std(topic_matrix, axis=0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Topic': [f'Topic {i}' for i in range(lda_model.num_topics)],
            'Mean_Probability': topic_means,
            'Std_Probability': topic_stds
        })
        
        # Create interactive bar plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Topic'],
            y=df['Mean_Probability'],
            error_y=dict(type='data', array=df['Std_Probability']),
            marker_color=self.topic_colors[:len(df)],
            name='Topic Distribution'
        ))
        
        fig.update_layout(
            title='Average Topic Distribution Across Documents',
            xaxis_title='Topics',
            yaxis_title='Average Probability',
            width=800,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Topic distribution plot saved to {save_path}")
        
        return fig
    
    def create_pyldavis(self, lda_model, corpus, dictionary, save_path=None):
        """Create interactive pyLDAvis visualization."""
        print("Creating pyLDAvis visualization...")
        
        try:
            # Prepare visualization
            vis = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
            
            if save_path:
                pyLDAvis.save_html(vis, save_path)
                print(f"Interactive LDA visualization saved to {save_path}")
            
            return vis
        
        except Exception as e:
            print(f"Error creating pyLDAvis: {e}")
            return None
    
    def plot_model_comparison(self, cluster_labels, lda_model, corpus, save_path=None):
        """Compare clustering and topic modeling results."""
        print("Creating model comparison visualization...")
        
        # Get topic assignments for documents
        doc_topics = []
        for doc in corpus:
            topic_dist = lda_model.get_document_topics(doc)
            if topic_dist:
                dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
            else:
                dominant_topic = -1
            doc_topics.append(dominant_topic)
        
        # Create confusion matrix
        unique_clusters = np.unique(cluster_labels)
        unique_topics = np.unique(doc_topics)
        
        confusion_matrix = np.zeros((len(unique_clusters), len(unique_topics)))
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_topics = np.array(doc_topics)[cluster_mask]
            
            for topic_id in unique_topics:
                if topic_id >= 0:  # Valid topic
                    count = np.sum(cluster_topics == topic_id)
                    confusion_matrix[cluster_id, topic_id] = count
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=[f'Topic {i}' for i in unique_topics if i >= 0],
            y=[f'Cluster {i}' for i in unique_clusters],
            colorscale='Blues',
            text=confusion_matrix.astype(int),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='K-means Clusters vs LDA Topics Overlap',
            xaxis_title='LDA Topics',
            yaxis_title='K-means Clusters',
            width=600,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def plot_topic_evolution(self, lda_models, save_path=None):
        """Plot how topics evolve with different numbers of topics."""
        print("Creating topic evolution plot...")
        
        # This would require multiple models with different topic numbers
        # For now, create a placeholder
        
        fig = go.Figure()
        fig.add_annotation(
            text="Topic Evolution Plot<br>Requires multiple LDA models with different topic numbers",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        
        fig.update_layout(
            title='Topic Evolution Analysis',
            width=800,
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dashboard_summary(self, clustering_metrics, topic_metrics, save_path=None):
        """Create a summary dashboard with key metrics."""
        print("Creating summary dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Clustering Metrics', 'Topic Metrics', 
                          'Cluster Sizes', 'Topic Distribution'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Clustering metrics
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=clustering_metrics.get('silhouette_score', 0),
                title={'text': "Silhouette Score"},
                gauge={'axis': {'range': [-1, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [-1, 0], 'color': "lightgray"},
                                {'range': [0, 1], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.5}}
            ),
            row=1, col=1
        )
        
        # Topic metrics
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=topic_metrics.get('coherence_score', 0),
                title={'text': "Coherence Score"},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 1], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.7}}
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Model Performance Dashboard',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Dashboard saved to {save_path}")
        
        return fig