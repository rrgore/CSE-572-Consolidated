import numpy as np
import csv
from sklearn.cluster import KMeans


# create bins
def create_bins(meal_data):
    meal_range = max(meal_data) - min(meal_data)
    n_bins = meal_range/20
    if meal_range % 20 != 0:
        n_bins += 1

    return n_bins


def get_meal_data():
    data = None
    meal_data = []
    with open('InsulinData.csv', 'r') as f:
        data = f.readlines()[1:]
    for r in data:
        meal = r.split(',')[24]
        if meal != '':
            meal_data.append(int(meal))
    print(len(meal_data))
    return meal_data


if __name__ == '__main__':
    X = np.array(get_meal_data())
    n_bins = create_bins(X)
    X = X.reshape(-1, 1)
    km = KMeans(n_clusters=n_bins, random_state=0)
    km.fit_predict(X)
    
    # sse kmeans
    sse_kmeans = km.inertia_

    # entropy kmeans
    # sse dbscan
    