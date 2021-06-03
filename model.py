import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Import some Data from the iris Data Set
X,y = datasets.load_iris(return_X_y= True)

# Take only the first two features of Data.
# To avoid the slicing, Two-Dim Dataset can be used

#X = pd.DataFrame(iris.data).values
#y = pd.DataFrame(iris.target).values
X_data = pd.DataFrame(X)
y_data = pd.DataFrame(y)
X_data = X_data.rename(columns = {0:'sepal length',1:'sepal width',2:'petal length',3:'petal width'})
y_data = y_data.replace([0,1,2],["Setosa","Versicolour","Virginica"])
y_data = y_data.rename(columns = {0:'iris'})
#y_datadummy = pd.get_dummies(y_data["target"],drop_first= True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state=5)
knn = KNeighborsClassifier(n_neighbors = 1,weights= 'distance')
knn.fit(X_train, y_train)
predictedydata = knn.predict(X_test)
score = knn.score(X_test,y_test)
cv_scores = cross_val_score(knn, X_data, y_data, cv=20)
print((np.mean(cv_scores)))

import pickle
filename = 'irismodell.pkl'
pickle.dump(knn,open(filename,'wb'))