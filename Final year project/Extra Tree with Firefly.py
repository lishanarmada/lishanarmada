pip install sklearn-nature-inspired-algorithms
from sklearn.tree import DecisionTreeClassifier
clf = ExtraTreeClassifier(random_state=42)
 param_grid = {
   'criterion': ['gini', 'entropy'],
   'max_depth': [None, 2, 4, 6, 8, 10],
   'min_samples_split': [2, 4, 6, 8, 10],
   'min_samples_leaf': [1, 2, 4, 6, 8],
   'max_features': [None, 'auto', 'sqrt', 'log2']
}

import pandas as pd
import numpy as np
data = pd.read_csv('/content/MSCAD.csv')
X = data.iloc[:,:-1]
print("The attributes are:",X)
y = data.iloc[:,-1]
print("The target is:",y)
from sklearn.preprocessing import LabelEncoder
le_output = LabelEncoder()
y.output = le_output.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'train size - {len(X_train)}\ntest size - {len(X_test)}')
from sklearn_nature_inspired_algorithms.model_selection import NatureInspiredSearchCV
nia_search = NatureInspiredSearchCV(
    clf,
    param_grid,
    cv=5,
    verbose=1,
    algorithm='fa', 
    population_size=25,
    max_n_gen=100,
    max_stagnating_gen=10,    
    runs=5,
    scoring='f1_macro',
    random_state=42,
)

nia_search.fit(X_train, y_train)
print(nia_search.best_params_)
from sklearn.metrics import classification_report
clf = ExtraTreeClassifier(**nia_search.best_params_, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
