import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import data_base
import kmeans
import statistics_clusters

# Initialize database
database = data_base.DataBase(r"inputs\data_base.xlsx")
kmeans_plots = kmeans.KMeans_plots(25,
                                   database.get_random_sample(250,42),
                                   42)


# Plot the locations based on Longitude as x axis and Latitude as y axis
"""
data_base.plot_locations()
"""

# Plot the density heatmap
"""
data_base.plot_heatmap()
"""


# Plot the locations based on Longitude as x axis and Latitude as y axis but this time sorted by postal code
# This part is very heavy because it has to sort by postal code

"""
data_base.plot_locations_by_postal_code()
"""

# Plot clusters without scaling
"""
kmeans_plots.plot_kmeans_no_scaling()
"""


# Plot clusters using kmeans
"""
kmeans_plots.plot_kmeans_scaled()
"""


# Plot n cluster using kmeans with circles based centroids
"""
kmeans_plots.plot_kmeans_circles_centroids()
"""

# Plot n clusters using kmeans with circles based on barycenters
"""
kmeans_plots.plot_kmeans_circles_barycenters()
"""
# Plot kmeans initialized with barycenters
"""
kmeans_plots.plot_kmeans_based_on_barycenters()
"""


"""
barycenters = compute_barycenters(kmeans,kmeans.cluster_centers_,sample)
kmeans_barycenters = sklearn.cluster.KMeans(n_clusters=num_clusters,init=barycenters,n_init=1)
sample['Cluster_kmeans_barycenters'] = kmeans_barycenters.fit_predict(sample[['Latitude_scaled', 'Longitude_scaled']])
barycenters = compute_barycenters(kmeans_barycenters,kmeans_barycenters.cluster_centers_,sample)
max_distances_barycenter,mean_distances_barycenter = max_mean_distance_barycenter(kmeans_barycenters,barycenters,sample)
plt.figure(figsize=(10, 6))
for n in range(6):
    points_cluster = sample.loc[kmeans_barycenters.labels_ == n,
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

#print(sklearn.metrics.silhouette_score(sample[['Latitude_scaled', 'Longitude_scaled']],sample['Cluster_kmeans']))
#print(sklearn.metrics.silhouette_score(sample[['Latitude_scaled', 'Longitude_scaled']],sample['Cluster_kmeans_barycenters']))

"""


stats = statistics_clusters.statistics_clusters(database)
stats.plot_silhouette_scores(50)

