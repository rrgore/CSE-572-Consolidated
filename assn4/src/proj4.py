# CSE 572: Assignment 4
# Author: Rahul Gore

import datetime
import numpy as np

np.set_printoptions(precision=2, threshold=5, linewidth=200)

def readFile(fname):
    data = None
    with open(fname, 'r') as f:
        data = f.readlines()[1:]
        data = [tup.split(',') for tup in data]
    return data

def convInt(str):
    if str == '' or str == 'nan':
        return 0
    else:
        return int(str)

# getting all meal timestamps
ins_data = readFile('InsulinData.csv')
insulin_data = []
for r in ins_data:
    carb = r[24]
    date = datetime.datetime.strptime(r[1], "%m/%d/%Y")
    time = datetime.datetime.strptime(r[2], "%H:%M:%S").time()
    if carb != '' and carb != '0':
        insulin_data.append(datetime.datetime.combine(date, time))
insulin_data.reverse()

# remove invalid data
valid_insulin_data = []
i = 0
while i < len(insulin_data)-1:
    diff = insulin_data[i+1] - insulin_data[i]
    if diff < datetime.timedelta(minutes=30):
        i += 2
        continue
    elif diff >= datetime.timedelta(hours=2):
        valid_insulin_data.append(insulin_data[i])
    i += 1

# get windows of t-30 - t+120
meal_windows = []
for insulin_tuple in valid_insulin_data:
    meal_windows.append([insulin_tuple-datetime.timedelta(minutes=30), insulin_tuple+datetime.timedelta(hours=2)])

# get cgm values within meal windows
cgm_data = readFile('CGMData.csv')
meal_dataset = []
mpt = len(meal_windows)-1
i = 0
while i < len(cgm_data) and mpt >= 0:
    date = datetime.datetime.strptime(cgm_data[i][1], "%m/%d/%Y")
    time = datetime.datetime.strptime(cgm_data[i][2], "%H:%M:%S").time()
    ts = datetime.datetime.combine(date, time)
    if ts < meal_windows[mpt][1]:
        meal = []
        # meal.append(meal_windows[mpt][0])
        for _ in range(30):
            if i >= len(cgm_data):
                break
            meal.append(cgm_data[i][30])
            i += 1
        meal_dataset.append(meal)
        mpt -= 1
    else:
        i += 1
 

meal_dataset = [[convInt(cell) for cell in row] for row in meal_dataset]
meal_dataset = np.array(meal_dataset)

# calculate cgm_min and cgm_max
cgm_min = np.min(meal_dataset)
cgm_max = np.max(meal_dataset)

n_bins = int((cgm_max - cgm_min)/20)+1
bin_list = [[] for _ in range(n_bins)]
i = 0
for row in meal_dataset:
    for cell in row:
        i = int((cell - cgm_min)/20)
        bin_list[i].append(cell)

# get insulin values within meal windows
# bolus_data = []
# mpt = len(meal_windows)-1
# i = 0
# while i < len(ins_data) and mpt >= 0:
#     date = datetime.datetime.strptime(ins_data[i][1], "%m/%d/%Y")
#     time = datetime.datetime.strptime(ins_data[i][2], "%H:%M:%S").time()
#     ts = datetime.datetime.combine(date, time)
#     if ts < meal_windows[mpt][1]:
#         b_tuple = []
#         for _ in range(30):
#             if i >= len(ins_data):
#                 break
#             b_tuple.append(ins_data[i][19])
#             i += 1
#         bolus_data.append(b_tuple)
#         mpt -= 1
#     else:
#         i += 1
# print(len(bolus_data))
# print(bolus_data[0:5])


bolus_data = []
i = len(valid_insulin_data) - 1
for r in ins_data:
    date = datetime.datetime.strptime(r[1], "%m/%d/%Y")
    time = datetime.datetime.strptime(r[2], "%H:%M:%S").time()
    stamp = datetime.datetime.combine(date, time)
    if valid_insulin_data[i] == stamp:
        bolus_data.append(r[19])
        i -= 1
print(len(bolus_data))
print(bolus_data[0:10])
print(valid_insulin_data[-10:-1])

# step 3: calculate bin number of max(CGM) in each meal data
b_max_arr = [max(a) for a in meal_dataset]
b_max_arr = [int((tup - cgm_min)/20) for tup in b_max_arr]
print(len(b_max_arr))

# step 4: calculate bin number of meal data [6]
b_meal_arr = [a[6] for a in meal_dataset]
b_meal_arr = [int((tup - cgm_min)/20) for tup in b_meal_arr]
print(len(b_meal_arr))