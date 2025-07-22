"""
Streamlit dashboard for interactive document clustering and topic modeling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_20newsgroups
import pickle
import os
from io import StringIO

# Import our modules
from src.preprocessing import TextPreprocessor, preprocess_custom_dataset
from src.clustering import DocumentClusterer, ClusterOptimizer
from src.topic_modeling import TopicModeler, TopicOptimizer
from src.visualization import Visualizer
from src.evaluation import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="Document Clustering & Topic Modeling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_20newsgroups():
    """Load and cache 20 Newsgroups dataset."""
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    documents = list(newsgroups_train.data) + list(newsgroups_test.data)
    targets = list(newsgroups_train.target) + list(newsgroups_test.target)
    target_names = newsgroups_train.target_names
    
    return {
        'documents': documents,
        'targets': targets,
        'target_names': target_names
    }

@st.cache_data
def preprocess_data(documents, max_docs=None):
    """Preprocess documents with caching."""
    if max_docs:
        documents = documents[:max_docs]
    
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)
    tfidf_matrix = preprocessor.create_tfidf_matrix(documents)
    bow_corpus, dictionary = preprocessor.create_bow_matrix(processed_docs)
    
    return {
        'processed_docs': processed_docs,
        'tfidf_matrix': tfidf_matrix,
        'bow_corpus': bow_corpus,
        'dictionary': dictionary,
        'preprocessor': preprocessor
    }

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Document Clustering & Topic Modeling Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Dataset selection
    dataset_option = st.sidebar.selectbox(
        "Choose Dataset",
        ["20 Newsgroups", "Upload Custom Dataset"]
    )
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load data
    if dataset_option == "20 Newsgroups":
        if st.sidebar.button("Load 20 Newsgroups Dataset"):
            with st.spinner("Loading 20 Newsgroups dataset..."):
                data = load_20newsgroups()
                st.session_state.raw_data = data
                st.session_state.data_loaded = True
                st.success("Dataset loaded successfully!")
    
    else:  # Custom dataset
        uploaded_file = st.sidebar.file_uploader(
            "Upload your dataset",
            type=['txt', 'csv'],
            help="Upload a text file (.txt) or CSV file (.csv)"
        )
        
        if uploaded_file is not None:
            if st.sidebar.button("Process Custom Dataset"):
                with st.spinner("Processing custom dataset..."):
                    try:
                        if uploaded_file.type == "text/plain":
                            # Text file
                            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                            documents = stringio.readlines()
                        else:
                            # CSV file
                            df = pd.read_csv(uploaded_file)
                            text_column = st.sidebar.selectbox(
                                "Select text column",
                                df.columns.tolist()
                            )
                            documents = df[text_column].tolist()
                        
                        st.session_state.raw_data = {
                            'documents': documents,
                            'targets': None,
                            'target_names': None
                        }
                        st.session_state.data_loaded = True
                        st.success("Custom dataset loaded successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
    
    # Main content
    if st.session_state.data_loaded:
        
        # Dataset info
        st.subheader("📋 Dataset Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", len(st.session_state.raw_data['documents']))
        
        with col2:
            if st.session_state.raw_data['target_names']:
                st.metric("Categories", len(st.session_state.raw_data['target_names']))
            else:
                st.metric("Categories", "N/A (Custom Dataset)")
        
        with col3:
            avg_length = np.mean([len(doc.split()) for doc in st.session_state.raw_data['documents']])
            st.metric("Avg Document Length", f"{avg_length:.0f} words")
        
        # Preprocessing options
        st.sidebar.subheader("Preprocessing Options")
        max_docs = st.sidebar.slider(
            "Maximum documents to process",
            min_value=100,
            max_value=min(5000, len(st.session_state.raw_data['documents'])),
            value=min(1000, len(st.session_state.raw_data['documents'])),
            step=100,
            help="Reduce for faster processing"
        )
        
        # Model parameters
        st.sidebar.subheader("Model Parameters")
        n_clusters = st.sidebar.slider("Number of Clusters (K-means)", 2, 20, 8)
        n_topics = st.sidebar.slider("Number of Topics (LDA)", 2, 20, 8)
        
        # Process data
        if st.sidebar.button("🚀 Run Analysis"):
            
            # Preprocessing
            with st.spinner("Preprocessing documents..."):
                preprocessed_data = preprocess_data(
                    st.session_state.raw_data['documents'], 
                    max_docs
                )
                st.session_state.preprocessed_data = preprocessed_data
            
            # Clustering
            with st.spinner("Performing K-means clustering..."):
                clusterer = DocumentClusterer(n_clusters=n_clusters)
                cluster_labels = clusterer.fit_predict(preprocessed_data['tfidf_matrix'])
                
                evaluator = ModelEvaluator()
                clustering_metrics = evaluator.evaluate_clustering(
                    preprocessed_data['tfidf_matrix'],
                    cluster_labels,
                    st.session_state.raw_data.get('targets')
                )
                
                st.session_state.cluster_labels = cluster_labels
                st.session_state.clustering_metrics = clustering_metrics
                st.session_state.clusterer = clusterer
            
            # Topic modeling
            with st.spinner("Performing LDA topic modeling..."):
                topic_modeler = TopicModeler(n_topics=n_topics)
                lda_model = topic_modeler.fit_lda(
                    preprocessed_data['bow_corpus'],
                    preprocessed_data['dictionary']
                )
                
                topic_metrics = evaluator.evaluate_topic_model(
                    lda_model,
                    preprocessed_data['processed_docs'],
                    preprocessed_data['dictionary'],
                    preprocessed_data['bow_corpus']
                )
                
                st.session_state.lda_model = lda_model
                st.session_state.topic_metrics = topic_metrics
                st.session_state.topic_modeler = topic_modeler
            
            st.success("Analysis completed!")
        
        # Display results
        if hasattr(st.session_state, 'cluster_labels'):
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Clustering", "📝 Topics", "⚖️ Comparison"])
            
            with tab1:
                st.subheader("Model Performance Overview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### K-means Clustering")
                    metrics = st.session_state.clustering_metrics
                    st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
                    st.metric("Number of Clusters", metrics['n_clusters'])
                    st.metric("Cluster Balance", f"{metrics['cluster_balance']:.3f}")
                
                with col2:
                    st.markdown("### LDA Topic Modeling")
                    metrics = st.session_state.topic_metrics
                    st.metric("Coherence Score", f"{metrics['coherence_c_v']:.3f}")
                    st.metric("Topic Diversity", f"{metrics['topic_diversity']:.3f}")
                    st.metric("Topic Balance", f"{metrics['topic_balance']:.3f}")
            
            with tab2:
                st.subheader("K-means Clustering Results")
                
                # Cluster distribution
                cluster_counts = pd.Series(st.session_state.cluster_labels).value_counts().sort_index()
                
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cluster ID', 'y': 'Number of Documents'},
                    title='Cluster Size Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top terms per cluster
                st.subheader("Top Terms per Cluster")
                feature_names = st.session_state.preprocessed_data['preprocessor'].get_feature_names()
                cluster_terms = st.session_state.clusterer.get_top_terms_per_cluster(
                    st.session_state.preprocessed_data['tfidf_matrix'],
                    feature_names
                )
                
                for cluster_id, terms in cluster_terms.items():
                    with st.expander(f"Cluster {cluster_id} ({cluster_counts[cluster_id]} documents)"):
                        terms_df = pd.DataFrame(terms, columns=['Term', 'TF-IDF Score'])
                        st.dataframe(terms_df, use_container_width=True)
            
            with tab3:
                st.subheader("LDA Topic Modeling Results")
                
                # Topic distribution
                lda_model = st.session_state.lda_model
                topics = lda_model.show_topics(num_topics=-1, num_words=10, formatted=False)
                
                # Display topics
                for topic_id, words in topics:
                    with st.expander(f"Topic {topic_id}"):
                        words_df = pd.DataFrame(words, columns=['Word', 'Probability'])
                        st.dataframe(words_df, use_container_width=True)
                        
                        # Word cloud (simplified)
                        word_freq = {word: prob for word, prob in words}
                        st.write("Top words:", ", ".join([f"{word} ({prob:.3f})" for word, prob in words[:5]]))
            
            with tab4:
                st.subheader("Model Comparison")
                
                # Performance comparison
                comparison_data = {
                    'Metric': ['Silhouette Score', 'Coherence Score', 'Balance Score'],
                    'K-means': [
                        st.session_state.clustering_metrics['silhouette_score'],
                        0,  # N/A for clustering
                        st.session_state.clustering_metrics['cluster_balance']
                    ],
                    'LDA': [
                        0,  # N/A for topic modeling
                        st.session_state.topic_metrics['coherence_c_v'],
                        st.session_state.topic_metrics['topic_balance']
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Recommendations
                st.subheader("Recommendations")
                
                sil_score = st.session_state.clustering_metrics['silhouette_score']
                coh_score = st.session_state.topic_metrics['coherence_c_v']
                
                if sil_score > 0.5:
                    st.success("✅ K-means clustering shows good performance")
                elif sil_score > 0.2:
                    st.warning("⚠️ K-means clustering shows moderate performance")
                else:
                    st.error("❌ K-means clustering shows poor performance")
                
                if coh_score > 0.5:
                    st.success("✅ LDA topic modeling shows good performance")
                elif coh_score > 0.3:
                    st.warning("⚠️ LDA topic modeling shows moderate performance")
                else:
                    st.error("❌ LDA topic modeling shows poor performance")
    
    else:
        st.info("👆 Please select and load a dataset from the sidebar to begin analysis.")
        
        # Show sample data info
        st.subheader("About the 20 Newsgroups Dataset")
        st.write("""
        The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, 
        partitioned across 20 different newsgroups. It's a popular dataset for text classification 
        and clustering experiments.
        
        **Categories include:**
        - Computer technology (comp.*)
        - Recreation activities (rec.*)
        - Science topics (sci.*)
        - Politics and religion (talk.*)
        - For sale items (misc.forsale)
        """)

if __name__ == "__main__":
    main()