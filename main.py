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
# Plot 1
"""
data_base.plot_locations()
"""

# Plot the density heatmap
# Plot 2
"""
data_base.plot_heatmap()
"""


# Plot the locations based on Longitude as x axis and Latitude as y axis but this time sorted by postal code
# This part is very heavy because it has to sort by postal code
# Plot 3

"""
data_base.plot_locations_by_postal_code()
"""

# Plot clusters without scaling
# Plot 4
"""
kmeans_plots.plot_kmeans_no_scaling()
"""


# Plot clusters using kmeans
# Plot 5
"""
kmeans_plots.plot_kmeans_scaled()
"""


# Plot n cluster using kmeans with circles based centroids
# Plot 6
"""
kmeans_plots.plot_kmeans_circles_centroids()
"""

# Plot n clusters using kmeans with circles based on barycenters
# Plot 7
"""
kmeans_plots.plot_kmeans_circles_barycenters()
"""
# Plot kmeans initialized with barycenters
# Plot 8
"""
kmeans_plots.plot_kmeans_based_on_barycenters()
"""

# Statistics plots

# Initialize database for statistics
stats = statistics_clusters.statistics_clusters(database)

# Plot 9
"""
stats.plot_silhouette_scores(100)
"""

# Plot 10
"""
stats.plt_silhouette_score_differents_k(5,60)
"""

# Plot 11
# Change n_start and n_end values to 20 and 31 to get the score I got.
"""
stats.plt_max_silhouette(100,15,35)
"""


# Plot 12
"""
kmeans_plots_stats = kmeans.KMeans_plots(60,
                                   database.get_random_sample(250,42),
                                   42)
"""

# Plot 13
"""
kmeans_plots_stats.plot_kmeans_scaled()
"""

