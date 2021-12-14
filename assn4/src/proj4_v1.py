#!/usr/bin/env python
# coding: utf-8

# In[146]:


import datetime
import numpy as np
from functools import cmp_to_key
from pprint import pprint
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# In[140]:


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
    
def compare(x, y):
    return x[1] > y[1]

def writeFile(fname, data):
    with open(fname, 'w') as f:
        for line in data:
            f.write(line)


# In[101]:


ins_data = readFile('InsulinData.csv')
insulin_data = []
for r in ins_data:
    carb = r[24]
    date = datetime.datetime.strptime(r[1], "%m/%d/%Y")
    time = datetime.datetime.strptime(r[2], "%H:%M:%S").time()
    if carb != '' and carb != '0':
        insulin_data.append(datetime.datetime.combine(date, time))
insulin_data.reverse()


# In[102]:


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


# In[103]:


meal_windows = []
for insulin_tuple in valid_insulin_data:
    meal_windows.append([insulin_tuple-datetime.timedelta(minutes=30), insulin_tuple+datetime.timedelta(hours=2)])
    
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
        for _ in range(30):
            if i >= len(cgm_data):
                break
            meal.append(cgm_data[i][30])
            i += 1
        meal_dataset.append(meal)
        mpt -= 1
    else:
        i += 1


# In[85]:


valid_meal_dataset = []
for row in meal_dataset:
    if '' not in row:
        valid_meal_dataset.append(row)
meal_dataset = valid_meal_dataset


# In[86]:


meal_dataset[0]


# In[105]:


meal_dataset = [[convInt(cell) for cell in row] for row in meal_dataset]
meal_dataset = np.array(meal_dataset)

cgm_min = np.min(meal_dataset)
cgm_max = np.max(meal_dataset)

n_bins = int((cgm_max - cgm_min)/20)+1
bin_list = [[] for _ in range(n_bins)]
i = 0
for row in meal_dataset:
    for cell in row:
        i = int((cell - cgm_min)/20)
        bin_list[i].append(cell)


# In[106]:


b_max_arr = [max(a) for a in meal_dataset]
b_max_arr = [int((tup - cgm_min)/20) for tup in b_max_arr]
print(len(b_max_arr))

b_meal_arr = [a[6] for a in meal_dataset]
b_meal_arr = [int((tup - cgm_min)/20) for tup in b_meal_arr]
print(len(b_meal_arr))


# In[127]:


bolus_data = []
i = len(valid_insulin_data) - 1
for r in ins_data:
    date = datetime.datetime.strptime(r[1], "%m/%d/%Y")
    time = datetime.datetime.strptime(r[2], "%H:%M:%S").time()
    stamp = datetime.datetime.combine(date, time)
    if valid_insulin_data[i] == stamp and r[24] != '' and r[24] != '0':
        bolus_data.append(int(float(r[19])))
        i -= 1
print(len(bolus_data))


# In[128]:


COUNT = len(bolus_data)
dataset = []
for i in range(COUNT):
    row = ["max_"+str(b_max_arr[i]), "meal_"+str(b_meal_arr[i]), "ins_"+str(bolus_data[i])]
    dataset.append(row)
print(dataset[0:5])


# In[129]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)


# In[204]:


frequent_itemsets = apriori(df, min_support=0.002, use_colnames=True)
f_items = frequent_itemsets
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets[(frequent_itemsets['length'] == 3)]
frequent_itemsets = frequent_itemsets[frequent_itemsets['support']==frequent_itemsets['support'].max()]
frequent_itemsets[['itemsets','support']]


# In[186]:


freq_data = []
for i, r in frequent_itemsets.iterrows():
    r_str = ''
    for s in r['itemsets']:
        r_str += s[-1]
        r_str += ','
    r_str += str(r['support'])
    r_str += '\n'
    freq_data.append(r_str)
writeFile('Result_1.csv', freq_data)


# In[206]:


rules = association_rules(f_items, metric="confidence", min_threshold=0.1)
rules['ante_len'] = rules['antecedents'].apply(lambda x: len(x))
rules['cons_len'] = rules['consequents'].apply(lambda x: len(x))
rules = rules[(rules['ante_len'] == 2) & (rules['cons_len'] == 1)]
rules


# In[207]:


def func(row):
    for e in row['antecedents']:
        if e[:-1] != 'max_' and e[:-1] != 'meal_':
            return False
    return True
rules = rules[rules.apply(func, axis=1)]
rules_1 = rules[rules['confidence']==rules['confidence'].max()]
rules_1[['antecedents','consequents','confidence']]


# In[200]:


rules_1_data = []
for i, r in rules_1.iterrows():
    r_str = ''
    for s in r['antecedents']:
        r_str += s[-1]
        r_str += ','
    for s in r['consequents']:
        r_str += s[-1]
        r_str += ','
    r_str += str(r['confidence'])
    r_str += '\n'
    rules_1_data.append(r_str)
writeFile('Result_2.csv', rules_1_data)


# In[208]:


rules_2 = rules[rules['confidence']<0.15]
rules_2 = rules_2.sort_values(by='confidence')
rules_2[['antecedents','consequents','confidence']]


# In[203]:


rules_2_data = []
for i, r in rules_2.iterrows():
    r_str = ''
    for s in r['antecedents']:
        r_str += s[-1]
        r_str += ','
    for s in r['consequents']:
        r_str += s[-1]
        r_str += ','
    r_str += str(r['confidence'])
    r_str += '\n'
    rules_2_data.append(r_str)
writeFile('Result_3.csv', rules_2_data)


# In[ ]:




