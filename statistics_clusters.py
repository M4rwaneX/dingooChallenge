import pandas as pd
import numpy as np
import sklearn
import kmeans,data_base
import matplotlib.pyplot as plt


class statistics_clusters:
    def __init__(self,database):
        self.database = database

    def plot_silhouette_scores(self,n):
        """
    Plots the silhouette scores for centroids and barycenters across multiple random seeds.

    This function computes silhouette scores for K-Means clustering using both centroids 
    and barycenters across `n` different random initializations. It then visualizes the 
    variation in silhouette scores as a function of the seed number.

    Parameters:
    ----------
    n : int
        The number of different random seeds to test.

    Instance Attributes Required:
    ----------------------------
    - self.database : object
        An object that provides a `get_random_sample(size, seed)` method to generate 
        random samples of data.
    - kmeans.KMeans_plots : class
        A class that handles K-Means clustering and provides:
        - `get_silouhette_score_centroid()`: Returns the silhouette score based on centroids.
        - `get_silouhette_score_barycenters()`: Returns the silhouette score based on barycenters.

    Returns:
    -------
    None (Displays a line plot comparing silhouette scores)
        """
        silhouette_score_centroids = []
        silhouette_score_barycenters = []
        for i in range(n):

            # Creates and fit kmeans number i associated to a seed 
            kmeans_plots = kmeans.KMeans_plots(25,
                                               self.database.get_random_sample(250, i),
                                               i)
            silhouette_score_barycenters.append(kmeans_plots.get_silouhette_score_barycenters())
            silhouette_score_centroids.append(kmeans_plots.get_silouhette_score_centroid())
        silhouette_score_centroids = np.array(silhouette_score_centroids)
        silhouette_score_barycenters = np.array(silhouette_score_barycenters)
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, n), silhouette_score_centroids, marker='o', linestyle='-', label='Silhouette score centroids')
        plt.plot(range(0, n), silhouette_score_barycenters, marker='s', linestyle='--', label='Silhouette score barcenters')
        plt.legend(loc='best')
        plt.xlabel('Seed number i')
        plt.ylabel('Silhouette Score')
        print("Mean centroid = ", np.mean(silhouette_score_centroids))
        print("Barycenter centroid = ", np.mean(silhouette_score_barycenters))
        plt.show()

    def plt_silhouette_score_differents_k(self,n_start,n_end):
        """
    Plots the silhouette scores for different values of `k` (number of clusters) in K-Means clustering.

    This function evaluates K-Means clustering performance for a range of cluster numbers 
    (from `n_start` to `n_end`) by computing silhouette scores using centroids. It visualizes 
    the silhouette score as a function of the number of clusters, helping identify the optimal 
    number of clusters.

    Parameters:
    ----------
    n_start : int
        The starting number of clusters to evaluate (inclusive). It should be a value greater than or equal to 5.
    
    n_end : int
        The ending number of clusters to evaluate (exclusive). This should be greater than `n_start`.

    Instance Attributes Required:
    ----------------------------
    - self.database : object
        An object that provides a `get_random_sample(size, seed)` method to generate 
        random samples of data.
    - kmeans.KMeans_plots : class
        A class that handles K-Means clustering and provides:
        - `get_silouhette_score_centroid()`: Returns the silhouette score based on centroids.

    Returns:
    -------
    None (Displays a line plot showing silhouette scores for different values of `k`)
        """
        silhouette_score_centroids = []
        for i in range(n_start,n_end):

            kmeans_plots = kmeans.KMeans_plots(i,
                                               self.database.get_random_sample(250, 42),
                                               42)
            silhouette_score_centroids.append(kmeans_plots.get_silouhette_score_centroid())

        plt.figure(figsize=(10, 5))
        plt.plot(range(n_start, n_end), silhouette_score_centroids, marker='o', linestyle='-',
                 label='Silhouette score centroids')

        plt.legend(loc='best')
        plt.xlabel('Number of locations per clusters')
        plt.ylabel('Silhouette Score')
        print("Mean = ", np.mean(silhouette_score_centroids))
        plt.title("Maximum silhouette for seed 42")
        plt.show()

    def plt_max_silhouette(self,seeds,n_start,n_end):
        """
    Plots the maximum silhouette score for different numbers of clusters across multiple seeds.

    This function evaluates the maximum silhouette score for various values of `k` (number of clusters)
    within a given range (`n_start` to `n_end`), using multiple random seeds. For each number of clusters, 
    the function computes silhouette scores for `seeds` number of random initializations, then plots the 
    maximum silhouette score for each `k`.

    Parameters:
    ----------
    seeds : int
        The number of different random seeds to test for each value of `k`.
    
    n_start : int
        The starting number of clusters to evaluate (inclusive).
    
    n_end : int
        The ending number of clusters to evaluate (exclusive).

    Instance Attributes Required:
    ----------------------------
    - self.database : object
        An object that provides a `get_random_sample(size, seed)` method to generate random samples of data.
    - kmeans.KMeans_plots : class
        A class that handles K-Means clustering and provides:
        - `get_silouhette_score_centroid()`: Returns the silhouette score based on centroids.
        - `loc_per_clusters`: The number of locations per cluster.

    Returns:
    -------
    None (Displays a plot showing maximum silhouette scores for each number of clusters)

        """
        max_silhouette_score_centroids = np.empty((0, 2))

        for locations_per_clusters in range(n_start, n_end):
            silhouette_score_centroids = []

            for seed in range(seeds):
                kmeans_plots = kmeans.KMeans_plots(locations_per_clusters,
                                                   self.database.get_random_sample(250, seed),
                                                   seed)
                silhouette_score_centroids.append(kmeans_plots.get_silouhette_score_centroid())

            # Find the max of silhouette score for all the seed
            maximum = max(silhouette_score_centroids)

            # Index is the locations per clusters
            index = locations_per_clusters
            max_silhouette_score_centroids = np.vstack([max_silhouette_score_centroids, np.array([index, maximum])])
        plt.figure(figsize=(10, 5))
        plt.plot(max_silhouette_score_centroids[:,0],max_silhouette_score_centroids[:,1],
                 marker='o', linestyle='-')
        plt.xlabel('Number of locations per clusters')
        plt.ylabel('Silhouette Score')
        print("Silhouette score max = ",max(max_silhouette_score_centroids[:,1]))
        plt.title("Maximum silhouette for 100 differents seed")
        plt.show()


