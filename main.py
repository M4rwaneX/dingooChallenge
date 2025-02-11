import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.cluster


# Initialize database
data_base = pd.read_excel(r"inputs\data_base.xlsx")

print(data_base.info())

# Plot the locations based on Longitude as x axis and Latitude as y axis
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_base['GPS - Longitude'], y=data_base['GPS - Latitude'], alpha=0.5)
plt.title("Delivery Locations Distribution")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Plot the density heatmap
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

#
num_clusters = len(sample) // 25

# Scaling Latitude and Longitude
scaler = sklearn.preprocessing.StandardScaler()
sample[['Latitude_scaled', 'Longitude_scaled']] = scaler.fit_transform(sample[['GPS - Latitude', 'GPS - Longitude']])

kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=42)
dbscan = sklearn.cluster.DBSCAN(eps=0.5)

print(kmeans.cluster_centers_)

sample['Cluster_kmeans'] = kmeans.fit_predict(sample[['Latitude_scaled', 'Longitude_scaled']])
sample['Cluster_DBSCAN'] = kmeans.fit_predict(sample[['Latitude_scaled', 'Longitude_scaled']])

# Plot clusters using kmeans
plt.figure(figsize=(10, 6))
sns.scatterplot(x=sample['GPS - Longitude'], y=sample['GPS - Latitude'], hue=sample['Cluster_kmeans'], palette='viridis')
plt.title("Clustered Delivery Routes using KMEANS")
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
