"""
Flask web application for document clustering and topic modeling.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.datasets import fetch_20newsgroups

# Import our modules
from src.preprocessing import TextPreprocessor, preprocess_custom_dataset
from src.clustering import DocumentClusterer
from src.topic_modeling import TopicModeler
from src.visualization import Visualizer
from src.evaluation import ModelEvaluator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# Global variables to store analysis results
analysis_results = {}

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the uploaded file
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents = f.readlines()
            elif filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                documents = df.iloc[:, 0].tolist()  # First column
            
            # Store in global results
            analysis_results['raw_data'] = {
                'documents': documents,
                'filename': filename,
                'n_documents': len(documents)
            }
            
            return jsonify({
                'success': True,
                'filename': filename,
                'n_documents': len(documents)
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/load_newsgroups', methods=['POST'])
def load_newsgroups():
    """Load 20 Newsgroups dataset."""
    try:
        # Load dataset
        newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        
        documents = list(newsgroups_train.data) + list(newsgroups_test.data)
        targets = list(newsgroups_train.target) + list(newsgroups_test.target)
        target_names = newsgroups_train.target_names
        
        # Store in global results
        analysis_results['raw_data'] = {
            'documents': documents,
            'targets': targets,
            'target_names': target_names,
            'filename': '20_newsgroups',
            'n_documents': len(documents)
        }
        
        return jsonify({
            'success': True,
            'filename': '20 Newsgroups Dataset',
            'n_documents': len(documents),
            'n_categories': len(target_names)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error loading dataset: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Perform clustering and topic modeling analysis."""
    try:
        # Get parameters
        data = request.get_json()
        n_clusters = data.get('n_clusters', 8)
        n_topics = data.get('n_topics', 8)
        max_docs = data.get('max_docs', 1000)
        
        if 'raw_data' not in analysis_results:
            return jsonify({'error': 'No data loaded'}), 400
        
        documents = analysis_results['raw_data']['documents'][:max_docs]
        
        # Preprocessing
        preprocessor = TextPreprocessor()
        processed_docs = preprocessor.preprocess_documents(documents)
        tfidf_matrix = preprocessor.create_tfidf_matrix(documents)
        bow_corpus, dictionary = preprocessor.create_bow_matrix(processed_docs)
        
        # Clustering
        clusterer = DocumentClusterer(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(tfidf_matrix)
        
        # Topic modeling
        topic_modeler = TopicModeler(n_topics=n_topics)
        lda_model = topic_modeler.fit_lda(bow_corpus, dictionary)
        
        # Evaluation
        evaluator = ModelEvaluator()
        clustering_metrics = evaluator.evaluate_clustering(
            tfidf_matrix, cluster_labels,
            analysis_results['raw_data'].get('targets')
        )
        topic_metrics = evaluator.evaluate_topic_model(
            lda_model, processed_docs, dictionary, bow_corpus
        )
        
        # Visualization
        visualizer = Visualizer()
        
        # Create t-SNE plot
        tsne_fig = visualizer.plot_tsne_clusters(tfidf_matrix, cluster_labels)
        tsne_fig.write_html('static/plots/tsne_clusters.html')
        
        # Create topic distribution plot
        topic_fig = visualizer.plot_topic_distribution(lda_model, bow_corpus)
        topic_fig.write_html('static/plots/topic_distribution.html')
        
        # Get topics and cluster info
        topics = lda_model.show_topics(num_topics=-1, num_words=10, formatted=False)
        feature_names = preprocessor.get_feature_names()
        cluster_terms = clusterer.get_top_terms_per_cluster(tfidf_matrix, feature_names)
        
        # Store results
        analysis_results.update({
            'clustering_metrics': clustering_metrics,
            'topic_metrics': topic_metrics,
            'topics': topics,
            'cluster_terms': cluster_terms,
            'n_clusters': n_clusters,
            'n_topics': n_topics,
            'processed': True
        })
        
        return jsonify({
            'success': True,
            'clustering_metrics': clustering_metrics,
            'topic_metrics': topic_metrics,
            'n_clusters': n_clusters,
            'n_topics': n_topics
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/results')
def results():
    """Display analysis results."""
    if not analysis_results.get('processed'):
        return redirect(url_for('index'))
    
    return render_template('results.html', results=analysis_results)

@app.route('/api/topics')
def get_topics():
    """API endpoint to get topic information."""
    if not analysis_results.get('processed'):
        return jsonify({'error': 'No analysis results available'}), 400
    
    topics_data = []
    for topic_id, words in analysis_results['topics']:
        topics_data.append({
            'id': topic_id,
            'words': [{'word': word, 'probability': prob} for word, prob in words]
        })
    
    return jsonify({'topics': topics_data})

@app.route('/api/clusters')
def get_clusters():
    """API endpoint to get cluster information."""
    if not analysis_results.get('processed'):
        return jsonify({'error': 'No analysis results available'}), 400
    
    clusters_data = []
    for cluster_id, terms in analysis_results['cluster_terms'].items():
        clusters_data.append({
            'id': cluster_id,
            'terms': [{'term': term, 'score': score} for term, score in terms]
        })
    
    return jsonify({'clusters': clusters_data})

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'csv'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)