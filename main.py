# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset[['battery_power', 'ram']].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.31, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM model on the Training set
from sklearn.svm import SVC

classifier = SVC(C=0.3, kernel='rbf', random_state=0, gamma=0.13)
classifier.fit(X_train, y_train)

# Predicting the Test set results, comparison between real values and predicted ones
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# Predicting a new result
users_data_bp = input('Battery power value: ')
users_data_ram = input('Ram value: ')
print(f'Price range classification: {classifier.predict(sc.transform([[users_data_bp, users_data_ram]]))[0]}')

