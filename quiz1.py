# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, f1_score, recall_score
import time

mark_df = pd.read_csv('/content/DSAI-LVA-DATASET for Quiz.csv')
print(mark_df.head())

for i, rows in mark_df.iterrows():
    if rows['Pass'] == 'Yes' and rows['PreviousTestScore'] >= 75:
        mark_df.loc[i, 'Result'] = 'HGPass'
    elif rows['Pass'] == 'Yes' and rows['PreviousTestScore'] < 75:
        mark_df.loc[i, 'Result'] = 'LGPass'
    elif rows['Pass'] == 'No':
        mark_df.loc[i, 'Result'] = 'Fail'
mark_df = mark_df.drop('Pass', axis=1)

print(mark_df.head())

parent_edu = ['Masters', 'Bachelor''s', 'College', 'High School', 'Not Educated']
mark_df['Parent_Education'] = np.random.choice(parent_edu, size = len(mark_df['StudyTime']))

mark_df = mark_df.drop('ParentEducation', axis=1)
print(mark_df.head())

mark_df_shuffled = mark_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.7 * len(mark_df_shuffled))

train_set = mark_df_shuffled.iloc[:train_size]
test_set = mark_df_shuffled.iloc[train_size:]

train_set.to_csv('train.csv', index=False)
test_set.to_csv('test.csv', index=False)

test_df=pd.read_csv('/content/test.csv')
train_df=pd.read_csv('/content/train.csv')

lbl = LabelEncoder()
train_df['Parent_Education'] = lbl.fit_transform(train_df['Parent_Education'])
test_df['Parent_Education'] = lbl.transform(test_df['Parent_Education'])
train_df['Result'] = lbl.fit_transform(train_df['Result'])
test_df['Result'] = lbl.transform(test_df['Result'])

X_train = train_df.drop('Result', axis=1)
X_test = test_df.drop('Result', axis=1)
y_train = train_df['Result']
y_test = test_df['Result']

model_name = [
    ('Decision Tree Classifier', DecisionTreeClassifier()),
    ('K Nearest Neighbors', KNeighborsClassifier(n_neighbors=2)),
    ('SVM', SVC()),
    ('XGB Classifier', XGBClassifier(learning_rate=0.01, gamma=3))
]

results = {}
for name, model in model_name:
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    time_taken = round((time.time() - start_time), 2)
    accuracy = accuracy_score(y_pred, y_test)
    results[name] = accuracy
    print(f'{name} \nAccuracy : {accuracy*100:.2f}% \nTime Taken: {time_taken} sec\n ')

# Plotting the results
model_plot = pd.DataFrame(results.values(), index=results.keys(), columns=['Accuracy'])
model_plot.plot(kind='barh')
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()