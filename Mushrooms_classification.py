#------------------------------------------------------------------------------
#--- Mushrooms Classification--------------------------------------------------
#------------------------------------------------------------------------------
import os
os.chdir(...)
os.getcwd()

#---Data Loading---------------------------------------------------------------
import pandas as pd

df = pd.read_csv("mushrooms.csv", na_values='NA')

df.columns
df.isnull().sum()
df.info()
df.shape
df.groupby('odor').size()

#---Data encoding--------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
for col in df.columns:
    df[col] = labelEncoder.fit_transform(df[col])

#---Correlation of the all variables to veriable "habitat"---------------------
correlation = df.corr()
correlation['odor'].sort_values(ascending=False)

#---Splitting and Scaling------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[:].drop(['odor'] ,axis=1)
y = df.iloc[:,5] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#------------------------------------------------------------------------------
#---Logistic Regression--------------------------------------------------------
#------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

#---GridSearch-----------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {'C': [0.01,0.1,1],
              'penalty':['l1','l2']
             }

GS = GridSearchCV(logreg, parameters,cv=5)
GS.fit(X_train,y_train)

GS.best_params_
GS.best_score_

#------------------------------------------------------------------------------
#---KNN------------------------------------------------------------------------
#------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

#---GridSearch-----------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {'n_neighbors':[25,30,35],
              'weights':['uniform','distance'],
              'algorithm':['auto', 'ball_tree', 'kd_tree']
             }

knn = KNeighborsClassifier()
GS = GridSearchCV(knn, parameters,cv=5)
GS.fit(X_train,y_train)

GS.best_params_
GS.score(X_test, y_test)

#------------------------------------------------------------------------------
#---Random Forest--------------------------------------------------------------
#------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

#---GridSearch-----------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[50,60,70,80,90,100],
              'criterion':['gini', 'entropy'],
              'max_depth':[3,4,5,6],
             }

GS = GridSearchCV(rf, parameters,cv=5)
GS.fit(X_train,y_train)

GS.best_params_
GS.best_score_

#------------------------------------------------------------------------------
#---SVM------------------------------------------------------------------------
#------------------------------------------------------------------------------
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

svm = SVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

#---GridSearch-----------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {'C':[0.01,0.1,1],
              'kernel':['rbf','poly', 'linear'],
              'decision_function_shape':['ovr', 'ovo']
              }

GS = GridSearchCV(svm, parameters,cv=5)
GS.fit(X_train,y_train)

GS.best_params_
GS.best_score_

#---End------------------------------------------------------------------------
