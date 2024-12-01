import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
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
data['gender'] = data['gender'].map({'Male': 1, 'Female': 2})
label_encoder = LabelEncoder()
for column in sorted_headers_relevance + ['class']:
    data[column] = label_encoder.fit_transform(data[column])

y = data['class']
X = data[sorted_headers_relevance]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

modelo_generico = GaussianNB()
modelo_generico.fit(X_train, y_train)
y_pred = modelo_generico.predict(X_test)

recall = recall_score(y_test, y_pred)
print('Recall:', recall)


with open('./naive_model.pkl', 'wb') as file:
  pickle.dump(modelo_generico, file)