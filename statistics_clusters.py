import pandas as pd
import numpy as np
import sklearn
import kmeans,data_base
import matplotlib.pyplot as plt


class statistics_clusters:
    def __init__(self,database):
        self.database = database

    def plot_silhouette_scores(self,n):
        silhouette_score_centroids = []
        silhouette_score_barycenters = []
        for i in range(n):
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



