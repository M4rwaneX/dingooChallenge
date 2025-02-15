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
        
        # Fit KMeans on unscaled data
        self.kmeans = sklearn.cluster.KMeans(n_clusters=self.get_k(), random_state=self.seed)
        self.sample['Cluster_kmeans_unscaled'] = self.kmeans.fit_predict(self.sample[['GPS - Latitude', 'GPS - Longitude']])
        self.scale_data()

        # Fit KMeans on scaled data
        self.sample['Cluster_kmeans'] = self.kmeans.fit_predict(self.sample[['Latitude_scaled', 'Longitude_scaled']])
        
        # Compute barycenters of clusters
        barycenters = self.compute_barycenters(self.kmeans)

        # Fit KMeans with centroids initialized on barycenters computed before
        self.kmeans_barycenters = sklearn.cluster.KMeans(n_clusters=self.get_k(), init=barycenters)
        self.sample['Cluster_kmeans_barycenters'] = self.kmeans_barycenters.fit_predict(
            self.sample[['Latitude_scaled', 'Longitude_scaled']])

    def max_mean_distance(self,centers):
        """
    Computes the maximum and mean Euclidean distances of data points 
    from their respective cluster centers.

    Parameters:
    ----------
    centers : np.ndarray
        A NumPy array of shape (n_clusters, 2) containing the coordinates 
        of the cluster centers.

    Returns:
    -------
    max_distances : pd.Series
        A Pandas Series where each entry represents the maximum distance 
        of a data point from its assigned cluster center.
    
    mean_distances : pd.Series
        A Pandas Series where each entry represents the mean distance 
        of all points in a cluster from their respective cluster center.
        """
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
        """
    Computes the barycenters (geometric centers) of clusters based on 
    the assigned data points.

    Parameters:
    ----------
    kmeans : sklearn.cluster.KMeans
        A fitted KMeans model containing `cluster_centers_` and `labels_` attributes.

    Returns:
    -------
    barycenters : list of lists
        A list where each element is a `[longitude, latitude]` pair representing 
        the computed barycenter of a cluster.
        """
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
        """
    Plots the clustered delivery routes using K-Means clustering without scaling.

    The function visualizes GPS locations, color-coded by their assigned cluster.

    Parameters:
    ----------
    None (Uses instance attributes from `self`)

    Instance Attributes Required:
    ----------------------------
    - self.sample : pd.DataFrame
        A Pandas DataFrame containing at least the following columns:
        - 'GPS - Longitude': Longitude values of delivery points.
        - 'GPS - Latitude': Latitude values of delivery points.
        - 'Cluster_kmeans': Cluster labels assigned by K-Means.

    Returns:
    -------
    None (Displays a scatter plot)

        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.sample['GPS - Longitude'], y=self.sample['GPS - Latitude'], hue=self.sample['Cluster_kmeans_unscaled'],
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

        """
    Plots the clustered delivery routes using K-Means clustering on scaled data.

    This function first scales the GPS coordinates and then visualizes 
    the clusters using a scatter plot.

    Parameters:
    ----------
    None (Uses instance attributes from `self`)

    Instance Attributes Required:
    ----------------------------
    - self.sample : pd.DataFrame
        A Pandas DataFrame containing at least the following columns:
        - 'Longitude_scaled': Scaled longitude values of delivery points.
        - 'Latitude_scaled': Scaled latitude values of delivery points.
        - 'Cluster_kmeans': Cluster labels assigned by K-Means.
    - self.scale_data() : Method
        A method that scales the original longitude and latitude data.

    Returns:
    -------
    None (Displays a scatter plot)
        """
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
        """
    Plots K-Means clusters with their centroids and maximum distance circles.

    The function visualizes clusters using scaled coordinates and draws dashed circles 
    around centroids with radii equal to the maximum distance of points in each cluster.

    Parameters:
    ----------
    None (Uses instance attributes from `self`)

    Instance Attributes Required:
    ----------------------------
    - self.sample : pd.DataFrame
        A Pandas DataFrame containing at least:
        - 'Longitude_scaled': Scaled longitude values of delivery points.
        - 'Latitude_scaled': Scaled latitude values of delivery points.
        - 'Cluster_kmeans': Cluster labels assigned by K-Means.
    - self.kmeans : sklearn.cluster.KMeans
        A fitted KMeans model with `cluster_centers_` and `labels_` attributes.
    - self.max_mean_distance() : Method
        A method that computes the maximum and mean distances of data points from 
        their respective cluster centers.

    Returns:
    -------
    None (Displays a scatter plot with circles around centroids)
        """
        max_distances_centroid, mean_distances_centroid = self.max_mean_distance(self.kmeans.cluster_centers_)
        plt.figure(figsize=(10, 6))

        # Change values 4 and 7 to range the clusters ploted
        for n in range(4, 7):

            # Find the points of cluster n
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
        """
    Plots K-Means clusters with their barycenters and maximum distance circles.

    The function visualizes clusters using scaled coordinates and draws dashed circles 
    around barycenters with radii equal to the maximum distance of points in each cluster.

    Parameters:
    ----------
    None (Uses instance attributes from `self`)

    Instance Attributes Required:
    ----------------------------
    - self.sample : pd.DataFrame
        A Pandas DataFrame containing at least:
        - 'Longitude_scaled': Scaled longitude values of delivery points.
        - 'Latitude_scaled': Scaled latitude values of delivery points.
        - 'Cluster_kmeans': Cluster labels assigned by K-Means.
    - self.kmeans : sklearn.cluster.KMeans
        A fitted KMeans model with `cluster_centers_` and `labels_` attributes.
    - self.compute_barycenters() : Method
        A method that computes the barycenters of clusters.
    - self.max_mean_distance() : Method
        A method that computes the maximum and mean distances of data points 
        from their respective barycenters.

    Returns:
    -------
    None (Displays a scatter plot with circles around barycenters)
        """
        barycenters = self.compute_barycenters(self.kmeans)
        max_distances_barycenter, mean_distances_barycenter = self.max_mean_distance(barycenters)
        plt.figure(figsize=(10, 6))
        for n in range(6):

             # Find the points of cluster n
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
            
            # Find the points of cluster n
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
