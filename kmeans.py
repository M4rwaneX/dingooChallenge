import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

class KMeans_plots:

    def __init__(self, loc_per_clusters,sample,seed = None):
        self.loc_per_clusters = loc_per_clusters
        self.sample = sample
        if seed is not None:
            self.seed = seed
        self.kmeans = sklearn.cluster.KMeans(n_clusters=self.get_k(), random_state=self.seed)
        self.sample['Cluster_kmeans'] = self.kmeans.fit_predict(self.sample[['GPS - Latitude', 'GPS - Longitude']])
        self.scale_data()
        self.sample['Cluster_kmeans'] = self.kmeans.fit_predict(self.sample[['Latitude_scaled', 'Longitude_scaled']])
        barycenters = self.compute_barycenters(self.kmeans)
        self.kmeans_barycenters = sklearn.cluster.KMeans(n_clusters=self.get_k(), init=barycenters)
        self.sample['Cluster_kmeans_barycenters'] = self.kmeans_barycenters.fit_predict(
            self.sample[['Latitude_scaled', 'Longitude_scaled']])

    def max_mean_distance(self,centers):
        max_distances = pd.Series(dtype=float)
        mean_distances = pd.Series(dtype=float)
        for i in range(len(centers)):
            points_cluster = self.sample.loc[self.kmeans.labels_ == i,
            ['Longitude_scaled', 'Latitude_scaled']].values
            points_cluster = np.array(points_cluster)
            distances = np.linalg.norm(points_cluster - centers[i], axis=1)
            max_distances[i] = np.max(distances)
            mean_distances[i] = np.mean(distances)
        return max_distances, mean_distances

    def compute_barycenters(self,kmeans):
        barycenters = []
        for i in range(len(kmeans.cluster_centers_)):
            points_cluster = self.sample.loc[kmeans.labels_ == i,
            ['Longitude_scaled', 'Latitude_scaled']].values
            sum_x = 0
            sum_y = 0
            for i in range(len(points_cluster)):
                sum_x = sum_x + points_cluster[i][0]
                sum_y = sum_y + points_cluster[i][1]
            barycenters.append([sum_x / len(points_cluster), sum_y / len(points_cluster)])
        return barycenters

    def get_k(self):
        return len(self.sample)//self.loc_per_clusters

    def get_silouhette_score_barycenters(self):
        return sklearn.metrics.silhouette_score(
            self.sample[['Latitude_scaled', 'Longitude_scaled']],
            self.sample['Cluster_kmeans_barycenters'])

    def get_silouhette_score_centroid(self):
        return sklearn.metrics.silhouette_score(
            self.sample[['Latitude_scaled', 'Longitude_scaled']],
            self.sample['Cluster_kmeans'])

    def plot_kmeans_no_scaling(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.sample['GPS - Longitude'], y=self.sample['GPS - Latitude'], hue=self.sample['Cluster_kmeans'],
                        palette='viridis')
        plt.title("Clustered Delivery Routes using KMEANS")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(title="Cluster")
        plt.show()

    def scale_data(self):
        scaler = sklearn.preprocessing.StandardScaler()
        self.sample[['Latitude_scaled', 'Longitude_scaled']] = scaler.fit_transform(
            self.sample[['GPS - Latitude', 'GPS - Longitude']])


    def plot_kmeans_scaled(self):
        self.scale_data()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.sample['Longitude_scaled'], y=self.sample['Latitude_scaled'], hue=self.sample['Cluster_kmeans'],
                        palette='viridis')
        plt.title("Clustered Delivery Routes using KMEANS")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(title="Cluster")
        plt.show()

    def plot_kmeans_circles_centroids(self):
        max_distances_centroid, mean_distances_centroid = self.max_mean_distance(self.kmeans.cluster_centers_)
        plt.figure(figsize=(10, 6))
        for n in range(4, 7):
            points_cluster = self.sample.loc[self.kmeans.labels_ == n,
            ['Longitude_scaled', 'Latitude_scaled']].values
            sns.scatterplot(x=points_cluster[:, 0],
                            y=points_cluster[:, 1],
                            alpha=0.5)

            circle = plt.Circle(self.kmeans.cluster_centers_[n],
                                max_distances_centroid[n],
                                color='gray',
                                fill=False,
                                linestyle="dashed")

            plt.gca().add_patch(circle)
            plt.scatter(self.kmeans.cluster_centers_[n, 0],
                        self.kmeans.cluster_centers_[n, 1],
                        marker='x',
                        s=200,
                        label="Centroïdes " + str(n))

        print("Silhouette score = ",sklearn.metrics.silhouette_score(
            self.sample[['Latitude_scaled', 'Longitude_scaled']],
            self.sample['Cluster_kmeans']))

        plt.title("Clusters")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(title="Cluster")
        plt.show()

    def plot_kmeans_circles_barycenters(self):
        barycenters = self.compute_barycenters(self.kmeans)
        max_distances_barycenter, mean_distances_barycenter = self.max_mean_distance(barycenters)
        plt.figure(figsize=(10, 6))
        for n in range(6):
            points_cluster = self.sample.loc[self.kmeans.labels_ == n,
            ['Longitude_scaled', 'Latitude_scaled']].values
            sns.scatterplot(x=points_cluster[:, 0],
                            y=points_cluster[:, 1],
                            alpha=0.5)

            circle = plt.Circle(barycenters[n],
                                max_distances_barycenter[n],
                                color='gray',
                                fill=False,
                                linestyle="dashed")

            plt.gca().add_patch(circle)
            plt.scatter(barycenters[n][0],
                        barycenters[n][1],
                        marker='x',
                        s=200,
                        label="Barycenter " + str(n))

        plt.title("Clusters")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(title="Cluster")
        plt.show()

    def plot_kmeans_based_on_barycenters(self):
        barycenters = self.compute_barycenters(self.kmeans_barycenters)
        max_distances_barycenter, mean_distances_barycenter = self.max_mean_distance(barycenters)
        plt.figure(figsize=(10, 6))
        for n in range(len(self.kmeans_barycenters.cluster_centers_)):
            points_cluster = self.sample.loc[self.kmeans_barycenters.labels_ == n,
            ['Longitude_scaled', 'Latitude_scaled']].values
            sns.scatterplot(x=points_cluster[:, 0],
                            y=points_cluster[:, 1],
                            alpha=0.5)

            circle = plt.Circle(barycenters[n],
                                max_distances_barycenter[n],
                                color='gray',
                                fill=False,
                                linestyle="dashed")

            plt.gca().add_patch(circle)
            plt.scatter(barycenters[n][0],
                        barycenters[n][1],
                        marker='x',
                        s=200,
                        label="Barycenter " + str(n))

        print("Silhouette score = ",sklearn.metrics.silhouette_score(
            self.sample[['Latitude_scaled', 'Longitude_scaled']],
            self.sample['Cluster_kmeans_barycenters']))

        plt.title("Clusters")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(title="Cluster")
        plt.show()
