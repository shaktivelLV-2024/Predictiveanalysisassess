# -*- coding: utf-8 -*-
"""Q2_LVADSUSR115_SHAKTIVELR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14xlHYsJI9WcYzaM6Q2MzcnjNPCyBGMkp
"""

import pandas as pd
df = pd.read_csv("/content/booking.csv")
df.head()

#1
duplicates = df[df.duplicated()]
print("\nDuplicate rows:")
print(duplicates)
df.drop_duplicates(inplace=True)

print("\nMissing values in the dataset:")
print(df.isnull().sum())

numerical_features = ['number of week nights', 'number of adults','number of children','number of weekend nights','number of week nights','lead time','average price']
for feature in numerical_features:
    lower_bound = df[feature].quantile(0.05)
    upper_bound = df[feature].quantile(0.95)
    df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

df.head()

#2
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df["type of meal"] = label_encoder.fit_transform(df["type of meal"])
df["car parking space"] = label_encoder.fit_transform(df["car parking space"])
df["room type"] = label_encoder.fit_transform(df["room type"])
df["repeated"] = label_encoder.fit_transform(df["repeated"])
df["P-C"] = label_encoder.fit_transform(df["P-C"])
df["P-not-C"] = label_encoder.fit_transform(df["P-not-C"])
df["booking status"] = label_encoder.fit_transform(df["booking status"])
df["market segment type"] = label_encoder.fit_transform(df["market segment type"])
print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

from sklearn.tree import DecisionTreeClassifier

df = df.drop(['Booking_ID', 'date of reservation'], axis=1)

X = df.drop('booking status', axis=1)
y = df['booking status']

clf = DecisionTreeClassifier()
clf.fit(X, y)

feature_importances = dict(zip(X.columns, clf.feature_importances_))
print("Feature Importances:")
for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.4f}")

#dropping insignificant features

df = df.drop(['P-not-C', 'P-C', 'repeated'], axis=1)

"""The sigmoid function is used widely in binary
 classification by converting catogorical values into probabilites and
 fitting them in the sigmoid curve to classify data points
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

X = df.drop('booking status', axis=1)
y = df['booking status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import f1_score
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

f1_scores = f1_score(y_test, y_pred, average=None)

# Print F1 scores for each class
for i, f1 in enumerate(f1_scores):
    print(f"F1 Score for class {i}: {f1}")

