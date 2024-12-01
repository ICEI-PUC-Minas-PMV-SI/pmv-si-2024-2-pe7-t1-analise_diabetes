import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import pickle

sorted_headers_relevance = [
  'polyuria', 'polydipsia', 'age', 'gender', 'sudden_weight_loss',
  'partial_paresis', 'polyphagia', 'irritability', 'alopecia', 'visual_blurring',
  'weakness', 'muscle_stiffness', 'genital_thrush', 'obesity', 'delayed_healing', 'itching'
]

path = './dataset-full.csv'
dados = pd.read_csv(path)

dados.columns = dados.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
dados['gender'] = dados['gender'].map({'Male': 1, 'Female': 2})
label_encoder = LabelEncoder()
for column in sorted_headers_relevance + ['class']:
  dados[column] = label_encoder.fit_transform(dados[column])

X = dados[sorted_headers_relevance].to_numpy()
y = dados['class'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

recall = recall_score(y_test, y_pred)
print('Recall:', recall)

with open('./random_model.pkl', 'wb') as file:
  pickle.dump(modelo, file)
