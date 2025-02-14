import pandas as pd
import numpy as np
import sklearn


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


for n in range(10):
    # Selects 200 to 300 deliveries from the dataset
    sample = data_base.sample(n=np.random.randint(200, 301))

    # We want about 20 to 30 locations for each clusters
    num_clusters = len(sample) // 25

    scaler = sklearn.preprocessing.StandardScaler()
    sample[['Latitude_scaled', 'Longitude_scaled']] = scaler.fit_transform(
        sample[['GPS - Latitude', 'GPS - Longitude']])
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=n)
    sample['Cluster_kmeans'] = kmeans.fit_predict(sample[['Latitude_scaled', 'Longitude_scaled']])
    barycenters = compute_barycenters(kmeans,kmeans.cluster_centers_,sample)
    kmeans_barycenters = sklearn.cluster.KMeans(n_clusters=num_clusters,init=barycenters)
    sample['Cluster_kmeans_barycenters'] = kmeans_barycenters.fit_predict(
        sample[['Latitude_scaled', 'Longitude_scaled']])
    print(sklearn.metrics.silhouette_score(sample[['Latitude_scaled', 'Longitude_scaled']],sample['Cluster_kmeans']))
    print(sklearn.metrics.silhouette_score(sample[['Latitude_scaled', 'Longitude_scaled']],sample['Cluster_kmeans_barycenters']))

