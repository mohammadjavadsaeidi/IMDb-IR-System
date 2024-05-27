import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks

if __name__ == "__main__":

    wandb.login(key="4b77efc93dfc702229a106ec1e610e26af4593f7")

    # 0. Embedding Extraction
    print("0. Embedding Extraction")
    file_path = '/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/classification/IMDB_Dataset.csv'
    fasttext_model_path = '/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/word_embedding/FastText_model.bin'

    review_loader = ReviewLoader(file_path, fasttext_model_path)
    review_loader.load_data()

    fasttext = FastText()
    fasttext.load_model(fasttext_model_path)

    embeddings, sentiments = review_loader.get_embeddings()

    # 1. Dimension Reduction
    print("1. Dimension Reduction")
    dim_reduction = DimensionReduction()

    # Perform PCA
    n_components = 50
    pca_embeddings = dim_reduction.pca_reduce_dimension(embeddings, n_components)

    # Plot PCA explained variance
    dim_reduction.wandb_plot_explained_variance_by_components(
        embeddings, project_name="ClusteringProject", run_name="PCA Explained Variance"
    )

    # Perform t-SNE
    tsne_embeddings = dim_reduction.convert_to_2d_tsne(pca_embeddings)

    # Plot t-SNE results
    dim_reduction.wandb_plot_2d_tsne(
        tsne_embeddings, project_name="ClusteringProject", run_name="t-SNE Visualization"
    )

    # 2. Clustering
    print("2. Clustering")
    clustering_utils = ClusteringUtils()
    clustering_metrics = ClusteringMetrics()

    # K-Means Clustering
    k_values = range(2, 9)

    # Evaluate K-Means clustering with different k values
    for k in k_values:
        cluster_centers, cluster_labels = clustering_utils.cluster_kmeans(pca_embeddings, k)
        silhouette = clustering_metrics.silhouette_score(pca_embeddings, cluster_labels)
        purity = clustering_metrics.purity_score(sentiments, cluster_labels)
        print(f"K-Means Clustering with k={k}: Silhouette Score = {silhouette}, Purity = {purity}")

    # Plot K-Means clustering scores
    clustering_utils.plot_kmeans_cluster_scores(
        pca_embeddings, sentiments, list(k_values), project_name="ClusteringProject", run_name="K-Means Scores"
    )

    # Visualize K-Means clustering for the optimal k value
    optimal_k = 5
    clustering_utils.visualize_kmeans_clustering_wandb(
        pca_embeddings, optimal_k, project_name="ClusteringProject", run_name=f"K-Means Clustering with k={optimal_k}"
    )

    # Hierarchical Clustering
    linkage_methods = ['single', 'complete', 'average', 'ward']

    for method in linkage_methods:
        cluster_labels = getattr(clustering_utils, f'cluster_hierarchical_{method}')(pca_embeddings)
        clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(
            pca_embeddings, project_name="ClusteringProject", linkage_method=method,
            run_name=f"Hierarchical Clustering ({method})"
        )

    # 3. Evaluation
    print("3. Evaluation")
    # Using clustering metrics to evaluate the final chosen clustering method
    final_cluster_labels = clustering_utils.cluster_kmeans(pca_embeddings, optimal_k)[1]
    silhouette_final = clustering_metrics.silhouette_score(pca_embeddings, final_cluster_labels)
    purity_final = clustering_metrics.purity_score(sentiments, final_cluster_labels)
    adjusted_rand_final = clustering_metrics.adjusted_rand_score(sentiments, final_cluster_labels)

    print(f"Final Evaluation for K-Means with k={optimal_k}:")
    print(f"Silhouette Score = {silhouette_final}")
    print(f"Purity = {purity_final}")
    print(f"Adjusted Rand Index = {adjusted_rand_final}")
