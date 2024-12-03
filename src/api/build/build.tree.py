
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

sorted_headers_relevance = [
  'polyuria',
  'polydipsia',
  'age',
  'gender',
  'sudden_weight_loss',
  'partial_paresis',
  'polyphagia',
  'irritability',
  'alopecia',
  'visual_blurring',
  'weakness',
  'muscle_stiffness',
  'genital_thrush',
  'obesity',
  'delayed_healing',
  'itching',
]

data = pd.read_csv('./dataset-full.csv')

data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

label_encoder = LabelEncoder()

for column in sorted_headers_relevance + ['class']:
  data[column] = label_encoder.fit_transform(data[column])

X = data[sorted_headers_relevance]
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,random_state=42) 

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)


param_grid = {
  'criterion': ['gini', 'entropy'],
  'max_depth': [None, 10, 20, 30],
  'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='recall')
grid_search.fit(X_train_bal, y_train_bal)

print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

best_model = grid_search.best_estimator_

print(f"Melhor modelo encontrado: {best_model}")

y_pred_best = best_model.predict(X_test)

recall = recall_score(y_test, y_pred_best)

print('Recall:', recall)

with open('tree_model.pkl', 'wb') as file:
  pickle.dump(best_model, file)