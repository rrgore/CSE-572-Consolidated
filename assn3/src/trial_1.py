# Assignment 3
# Author: Rahul Gore

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import datetime
import csv
from os.path import exists
from sklearn.preprocessing import StandardScaler
import math

meal_features = None
labels = None
if not exists('meal_features.npy'):
    meal_dataset = []
    if not exists('meal_data.csv'):
        # create data
        data = None
        insulin_data = []
        with open('InsulinData.csv', 'r') as f:
            data = f.readlines()[1:]
        for r in data:
            data_tuple = r.split(',')
            carb = data_tuple[24]
            date = datetime.datetime.strptime(data_tuple[1], "%m/%d/%Y")
            time = datetime.datetime.strptime(data_tuple[2], "%H:%M:%S").time()
            if carb != '' and carb != '0':
                carb = int(carb)/20
                insulin_tuple = [carb, datetime.datetime.combine(date, time)]
                insulin_data.append(insulin_tuple)
        insulin_data.reverse()

        # keep valid data
        valid_insulin_data = []
        i = 0
        while i < len(insulin_data)-1:
            diff = insulin_data[i+1][1] - insulin_data[i][1]
            if diff < datetime.timedelta(minutes=30):
                i += 2
                continue
            elif diff >= datetime.timedelta(hours=2):
                valid_insulin_data.append(insulin_data[i])
            i += 1

        # get windows
        meal_windows = []
        for insulin_tuple in valid_insulin_data:
            meal_windows.append([insulin_tuple[0], insulin_tuple[1]-datetime.timedelta(minutes=30), insulin_tuple[1]+datetime.timedelta(hours=2)])


        cgm_data = None
        with open('CGMData.csv', 'r') as f:
            cgm_data = f.readlines()[1:]
            cgm_data = [tup.split(',') for tup in cgm_data]

        # get cgm values
        mpt = len(meal_windows)-1
        i = 0
        while i < len(cgm_data) and mpt >= 0:
            date = datetime.datetime.strptime(cgm_data[i][1], "%m/%d/%Y")
            time = datetime.datetime.strptime(cgm_data[i][2], "%H:%M:%S").time()
            ts = datetime.datetime.combine(date, time)
            if ts < meal_windows[mpt][2]:
                meal = []
                meal.append(meal_windows[mpt][0])
                for _ in range(30):
                    if i >= len(cgm_data):
                        break
                    meal.append(cgm_data[i][30])
                    i += 1
                meal_dataset.append(meal)
                mpt -= 1
            else:
                i += 1

        # remove nan data
        valid_meal_dataset = []
        for row in meal_dataset:
            if '' not in row:
                valid_meal_dataset.append(row)

        meal_dataset = valid_meal_dataset
        with open('meal_data.csv', 'w') as f:
            wr = csv.writer(f)
            for row in meal_dataset:
                wr.writerow(row)

    else:
        with open('meal_data.csv', 'r') as f:
            meal_dataset = f.readlines()[1:]
            meal_dataset = [row.split(',') for row in meal_dataset]
                    
    meal_dataset = [[float(cell) for cell in row] for row in meal_dataset]
    meal_dataset = np.array(meal_dataset)
    labels = meal_dataset[:, 0]
    meal_dataset = meal_dataset[:, 1:]

    # extract features
    i = 0
    for row in meal_dataset:
        f1 = np.array([max(row) - min(row)])
        t_del = np.argmax(row) - np.argmin(row)
        f8 = np.array(abs(t_del))
        f_all = np.concatenate((f1, f8), axis=None)
        if i == 0:
            meal_features = f_all
            i = 1
        else:
            meal_features = np.vstack((meal_features, f_all))

    # scaling
    scaler = StandardScaler()
    meal_features = scaler.fit_transform(meal_features)

    np.save('meal_features.npy', meal_features)
    np.save('labels.npy', labels)

else:
    meal_features = np.load('meal_features.npy')
    labels = np.load('labels.npy')

final_output = [0]*6

# kmeans
kmeans = KMeans(n_clusters=7, init='random', n_init=10, max_iter=100, random_state=0)
kmeans.fit_predict(meal_features)
label_set = set(kmeans.labels_)

# kmeans sse
final_output[0] = kmeans.inertia_

clusters = [[] for _ in label_set]
for i in range(len(labels)):
    clusters[kmeans.labels_[i]].append(labels[i])
p = np.zeros((7, 7))
for i in range(7):
    for j in range(len(clusters[i])):
        ind = int(clusters[i][j])
        p[i][ind] += 1
for i in range(7):
    psum = sum(p[i])
    p[i] /= psum

# kmeans entropy
entropy = np.zeros((7,))
for i in range(7):
    for j in range(7):
        if p[i][j] != 0:
            entropy[i] += (-1*p[i][j]*math.log(p[i][j]))
kmeans_entropy = 0
for i in range(7):
    kmeans_entropy += len(clusters[i])*entropy[i]
kmeans_entropy /= len(labels)
final_output[2] = kmeans_entropy

# kmeans purity
purity = np.zeros((7,))
for i in range(7):
    purity[i] = max(p[i])
kmeans_purity = 0
for i in range(7):
    kmeans_purity += len(clusters[i])*purity[i]
kmeans_purity /= len(labels)
final_output[4] = kmeans_purity

# dbscan
EPS = 0.01
dbscan = None
label_set = None
while EPS < 10.00:
    dbscan = DBSCAN(eps=EPS, min_samples=5)
    dbscan.fit_predict(meal_features)
    label_set = set(dbscan.labels_)
    if len(label_set) == 8:
        break
    EPS += 0.01

clusters = [[] for _ in label_set]
clusters = clusters[:-1]
noise = []
for i in range(len(labels)):
    if (dbscan.labels_[i] != -1):
        clusters[dbscan.labels_[i]].append(labels[i])
    else:
        noise.append(meal_features[i])
# TODO: reduce noise

# dbscan sse
dbscan_sse = 0
for i in range(len(clusters)):
    mean = sum(clusters[i])/(len(clusters[i]) * 1.0)
    for item in clusters[i]:
        dbscan_sse += (item - mean)**2
final_output[1] = dbscan_sse

p = np.zeros((7, 7))
for i in range(7):
    for j in range(len(clusters[i])):
        ind = int(clusters[i][j])
        p[i][ind] += 1
for i in range(7):
    psum = sum(p[i])
    p[i] /= psum

# dbscan entropy
entropy = np.zeros((7,))
for i in range(7):
    for j in range(7):
        if p[i][j] != 0:
            entropy[i] += (-1*p[i][j]*math.log(p[i][j]))
dbscan_entropy = 0
for i in range(7):
    dbscan_entropy += len(clusters[i])*entropy[i]
kmeans_entropy /= len(labels)
final_output[3] = dbscan_entropy

# kmeans purity
purity = np.zeros((7,))
for i in range(7):
    purity[i] = max(p[i])
dbscan_purity = 0
for i in range(7):
    dbscan_purity += len(clusters[i])*purity[i]
dbscan_purity /= len(labels)
final_output[5] = dbscan_purity

# write to Results.csv
with open('Results.csv', 'w') as f:
    wr = csv.writer(f)
    wr.writerow(final_output)