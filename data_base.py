import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

class DataBase:
    def __init__(self,file):
        self.data_base = pd.read_excel(file)

    def plot_locations(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data_base['GPS - Longitude'], y=self.data_base['GPS - Latitude'], alpha=0.5)
        plt.title("Delivery Locations Distribution")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def plot_heatmap(self):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            x=self.data_base['GPS - Longitude'],
            y=self.data_base['GPS - Latitude'],
            cmap="Reds",
            fill=True)
        plt.title("Delivery Density Heatmap")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def plot_locations_by_postal_code(self):
        data_postal_code = self.data_base.sort_values(by="Cód.Postal", ascending=True)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data_postal_code['GPS - Longitude'], y=data_postal_code['GPS - Latitude'],
                        hue=data_postal_code["Cód.Postal"], alpha=0.5)
        plt.title("Delivery Locations Distribution with postal code")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def get_random_sample(self,n=None,random_state=None):
        if random_state is None:
            if n is None:
                return self.data_base.sample(n=np.random.randint(200,301))
            else:
                return self.data_base.sample(n=n)
        else:
            return self.data_base.sample(n=n,random_state=random_state)

