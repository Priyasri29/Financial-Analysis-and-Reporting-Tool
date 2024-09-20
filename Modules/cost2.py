import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Streamlit app title
st.title("Agglomerative Clustering with User-Uploaded Data")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Display the original data
    st.write("Original Data:")
    st.write(data.head())

    # Feature selection
    features = st.multiselect("Select features for clustering", data.columns.tolist())

    if features:
        # Extract selected features
        features_data = data[features]
        
        # Handle categorical data
        categorical_features = features_data.select_dtypes(include=['object']).columns.tolist()
        numeric_features = features_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Define preprocessing steps
        if categorical_features:
            # Apply one-hot encoding to categorical features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(sparse=False), categorical_features)  # sparse=False to return dense array
                ]
            )
            features_preprocessed = preprocessor.fit_transform(features_data)
        else:
            # Only numeric features, scale them
            scaler = StandardScaler()
            features_preprocessed = scaler.fit_transform(features_data)
        
        # User input for number of clusters
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        # Perform Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        clusters = agg_clustering.fit_predict(features_preprocessed)

        # Add cluster information to the original data
        data['Cluster'] = clusters

        # Print silhouette score
        silhouette_avg = silhouette_score(features_preprocessed, clusters)
        st.write(f'Silhouette Score: {silhouette_avg:.2f}')

        # Reduce dimensions for visualization (optional)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_preprocessed)

        # Plot clusters
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', marker='o')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Agglomerative Clustering Results')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)

        # Display the plot in Streamlit
        st.pyplot(plt)
