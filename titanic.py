# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:29:34 2018

@author: Archit Bhatia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Importing Dataset
dataset=pd.read_csv('train.csv')
'''#replace age by mean  , then categorical, then make x and y then feature scaling , then model fitting , then back elim , visual
Xd= dataset.iloc[:,[1,2,4,5,6,7,9,11]]
X=Xd.iloc[:,1:9].values
y= dataset.iloc[:,1].values'''

'''
#Visualisation using sns
sns.barplot(x="Parch", y="Survived", data=dataset)
plt.show()'''

#Checking missing data
print(pd.isnull(dataset).sum())

#Qiuck check to contents of dataset
dataset.describe(include="all")

'''
print("Number of people embarking in Southampton (S):")
southampton =dataset[dataset["Embarked"] == "S"].shape[0]
print(southampton)
'''

# Taking care of missing data in age 

dataset["Age"].mean()
median=dataset["Age"].median()
dataset["Age"].mode()

# Replacing age by its median

dataset["Age"] = dataset["Age"].fillna(median)
print(pd.isnull(dataset).sum())

#Replacing  Embarked missing values

dataset["Embarked"].value_counts(normalize='True')
dataset["Embarked"].value_counts()

#Clearly S dominates so replacing missing values by S

dataset["Embarked"]=dataset["Embarked"].fillna('S')

pd.isnull(dataset).sum()

'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 2:3])
X[:,2:3] = imputer.transform(X[:,2:3])'''

#Dropping not important features for predictiong from the dataset
train = dataset
train = train.drop(['PassengerId'],axis=1)
train = train.drop(['Name'],axis=1)
train = train.drop(['Ticket'],axis=1)
train = train.drop(['Cabin'],axis=1)

'''
#Encoding categorical data(gender,  embarked)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X)
np.set_printoptions(threshold=np.nan)'''

#Encoding categorical data(gender,  embarked)

train["Sex"][train["Sex"]=="female"] = 1
train["Sex"][train["Sex"] == "male"] = 0

train["Embarked"][train["Embarked"]=='S'] = 0
train["Embarked"][train["Embarked"]=='C'] = 1
train["Embarked"][train["Embarked"]=='Q'] = 2

#Splitting training and test set

X = train.iloc[:,1:].values
y = train.iloc[:,:1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 0)

#Applying Model

# Multiple Linear Regression

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
 accuracy_score(y_pred, y_test) * 100

#Accuracy = 82.23350253807106
 
 #K-NN
 # Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test) * 100

#Accuracy = 84.26395939086294

#Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test) * 100

#Accuracy = 81.7258883248731

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test) * 100

#Accuracy = 79.69543147208121

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test) * 100

#Accuracy(10) = 85.27918781725889 Accuracy(5000)=84.77157360406092

'''
So far it can be seen that random forest regression with n=10 was the best...
So I would use the same to predict test set...
Now I would do the cleaning and and processing of test data to make it ready for predictions....

'''
#IMPORT
data = pd.read_csv('test.csv')
#check missing
pd.isnull(data).sum()

data.describe(include="all")

#taking care of missing data
data["Age"].mean()
med=data["Age"].median()
data["Age"]=data["Age"].fillna(med)

#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(data["Fare"])):
    if pd.isnull(data["Fare"][x]):
        pclass = data["Pclass"][x] #Pclass = 3 , entry 152
        data["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

#Dropping useless features
test = data        
test = test.drop(['PassengerId'],axis=1)
test = test.drop(['Name'],axis=1)
test = test.drop(['Ticket'],axis=1)
test = test.drop(['Cabin'],axis=1)

# Encoding Sex and Embarked
test["Sex"][test["Sex"]=="female"] = 1
test["Sex"][test["Sex"] == "male"] = 0

test["Embarked"][test["Embarked"]=='S'] = 0
test["Embarked"][test["Embarked"]=='C'] = 1
test["Embarked"][test["Embarked"]=='Q'] = 2

#Making matrice of features
Xt=test.iloc[:,:].values

#Feature Scaling
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Xt = sc_X.fit_transform(Xt)

# Predicting test set with random forest
pred=classifier.predict(Xt)

#set ids as PassengerId  
ids = data['PassengerId']

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': pred })
output.to_csv('submission.csv', index=False)