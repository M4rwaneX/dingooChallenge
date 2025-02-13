import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.cluster
import sklearn.metrics

def max_mean_distance_centroids(kmeans,centroids,sample):
    max_distances = pd.Series(dtype=float)
    mean_distances = pd.Series(dtype=float)
    for i in range(len(centroids)):
        points_cluster = sample.loc[kmeans.labels_ == i,
        ['Longitude_scaled', 'Latitude_scaled']].values
        points_cluster = np.array(points_cluster)
        distances = np.linalg.norm(points_cluster - centroids[i], axis=1)
        max_distances[i] = np.max(distances)
        mean_distances[i] = np.mean(distances)
    return max_distances,mean_distances

def max_mean_distance_barycenter(kmeans,barycenters,sample):
    max_distances = pd.Series(dtype=float)
    mean_distances = pd.Series(dtype=float)
    for i in range(len(barycenters)):
        points_cluster = sample.loc[kmeans.labels_ == i,
        ['Longitude_scaled', 'Latitude_scaled']].values
        points_cluster = np.array(points_cluster)
        distances = np.linalg.norm(points_cluster - barycenters[i], axis=1)
        max_distances[i] = np.max(distances)
        mean_distances[i] = np.mean(distances)
    return max_distances,mean_distances

def compute_barycenters(kmeans,centroids,sample):
    barycenters = []
    for i in range(len(centroids)):
        points_cluster = sample.loc[kmeans.labels_ == i,
        ['Longitude_scaled', 'Latitude_scaled']].values
        sum_x = 0
        sum_y = 0
        for i in range(len(points_cluster)):
            sum_x = sum_x + points_cluster[i][0]
            sum_y = sum_y + points_cluster[i][1]
        barycenters.append([sum_x/len(points_cluster),sum_y/len(points_cluster)])
    return barycenters





# Initialize database
data_base = pd.read_excel(r"inputs\data_base.xlsx")

# Plot the locations based on Longitude as x axis and Latitude as y axis

"""
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_base['GPS - Longitude'], y=data_base['GPS - Latitude'], alpha=0.5)
plt.title("Delivery Locations Distribution")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
"""

# Plot the density heatmap

"""
plt.figure(figsize=(10, 6))
sns.kdeplot(
    x=data_base['GPS - Longitude'], 
    y=data_base['GPS - Latitude'], 
    cmap="Reds", 
    fill=True)
plt.title("Delivery Density Heatmap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
"""


# Plot the locations based on Longitude as x axis and Latitude as y axis but this time sorted by postal code
# This part is very heavy because it has to sort by postal code
"""
data_postal_code = data_base.sort_values(by="Cód.Postal", ascending=True)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_postal_code['GPS - Longitude'], y=data_postal_code['GPS - Latitude'],hue=data_postal_code["Cód.Postal"], alpha=0.5)
plt.title("Delivery Locations Distribution with postal code")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
"""

# Selects 200 to 300 deliveries from the dataset
sample = data_base.sample(n=np.random.randint(200, 301), random_state=42)

# We want about 20 to 30 locations for each clusters
num_clusters = len(sample) // 25



# Plot clusters without scaling

"""
kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=42)
dbscan = sklearn.cluster.DBSCAN(eps=0.5)
sample['Cluster_kmeans'] = kmeans.fit_predict(sample[['GPS - Latitude', 'GPS - Longitude']])
plt.figure(figsize=(10, 6))
sns.scatterplot(x=sample['GPS - Longitude'], y=sample['GPS - Latitude'], hue=sample['Cluster_kmeans'], palette='viridis')
plt.title("Clustered Delivery Routes using KMEANS")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Cluster")
plt.show()
"""

# Scaling Latitude and Longitude
scaler = sklearn.preprocessing.StandardScaler()
sample[['Latitude_scaled', 'Longitude_scaled']] = scaler.fit_transform(sample[['GPS - Latitude', 'GPS - Longitude']])
kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=42)
dbscan = sklearn.cluster.DBSCAN(eps=0.5)
sample['Cluster_kmeans'] = kmeans.fit_predict(sample[['Latitude_scaled', 'Longitude_scaled']])
sample['Cluster_DBSCAN'] = kmeans.fit_predict(sample[['Latitude_scaled', 'Longitude_scaled']])




# Plot clusters using kmeans
"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x=sample['Longitude_scaled'], y=sample['Latitude_scaled'], hue=sample['Cluster_kmeans'], palette='viridis')
circle = plt.Circle(kmeans.cluster_centers_[0], max_distances[0], color='gray', fill=False, linestyle="dashed")
plt.gca().add_patch(circle)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label="Centroïdes")
plt.title("Clustered Delivery Routes using KMEANS")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Cluster")
plt.show()
"""

# Plot n cluster using kmeans with circles based centroids
"""
max_distances_centroid,mean_distances_centroid = max_mean_distance_centroids(kmeans,kmeans.cluster_centers_,sample)
barycenters = compute_barycenters(kmeans,kmeans.cluster_centers_,sample)
max_distances_barycenter,mean_distances_barycenter = max_mean_distance_barycenter(kmeans,barycenters,sample)
plt.figure(figsize=(10, 6))
for n in range(4,7):
    points_cluster = sample.loc[kmeans.labels_ == n,
            ['Longitude_scaled', 'Latitude_scaled']].values
    sns.scatterplot(x=points_cluster[:, 0],
                    y=points_cluster[:, 1],
                    alpha=0.5)

    circle = plt.Circle(kmeans.cluster_centers_[n],
                        max_distances_centroids[n],
                        color='gray',
                        fill=False,
                        linestyle="dashed")

    plt.gca().add_patch(circle)
    plt.scatter(kmeans.cluster_centers_[n, 0],
                kmeans.cluster_centers_[n, 1],
                marker='x',
                s=200,
                label="Centroïdes " + str(n))

plt.title("Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Cluster")
plt.show()
"""

# Plot n clusters using kmeans with circles based on barycenters

"""
max_distances_centroid,mean_distances_centroid = max_mean_distance_centroids(kmeans,kmeans.cluster_centers_,sample)
barycenters = compute_barycenters(kmeans,kmeans.cluster_centers_,sample)
max_distances_barycenter,mean_distances_barycenter = max_mean_distance_barycenter(kmeans,barycenters,sample)
plt.figure(figsize=(10, 6))
for n in range(6):
    points_cluster = sample.loc[kmeans.labels_ == n,
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
"""

# Plot kmeans initialized with barycenters

"""
barycenters = compute_barycenters(kmeans,kmeans.cluster_centers_,sample)
kmeans_barycenters = sklearn.cluster.KMeans(n_clusters=num_clusters,init=barycenters)
sample['Cluster_kmeans_barycenters'] = kmeans_barycenters.fit_predict(sample[['Latitude_scaled', 'Longitude_scaled']])
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
"""


barycenters = compute_barycenters(kmeans,kmeans.cluster_centers_,sample)
kmeans_barycenters = sklearn.cluster.KMeans(n_clusters=num_clusters,init=barycenters)
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

print(sklearn.metrics.silhouette_score(sample))

plt.title("Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Cluster")
plt.show()


# Plot clusters using DBSCAN
"""
plt.figure(figsize=(10, 6))
sns.scatterplot(x=sample['GPS - Longitude'], y=sample['GPS - Latitude'], hue=sample['Cluster_DBSCAN'], palette='viridis')
plt.title("Clustered Delivery Routes using DBSCAN")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Cluster")
plt.show()
"""

#daily_sample.to_csv("clustered_deliveries.csv", index=False)
