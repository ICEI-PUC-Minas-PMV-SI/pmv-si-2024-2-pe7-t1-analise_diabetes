import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Colunas do dataset ordenadas pela relevancia em relacao a classificacao
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

# Carregar o CSV
data = pd.read_csv('./dataset-full.csv')

# Renomear os cabeçalhos para snake_case
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Tratar a coluna 'gender' manualmente (1 para Male, 2 para Female) o encoder nao suporta nativamente essas nomeclaturas
data['gender'] = data['gender'].map({'Male': 1, 'Female': 2})

# Converter outras colunas descritivas para numéricas com LabelEncoder
label_encoder = LabelEncoder()

# Utiliza o encoder para converter os dados de todas as colunas para numéricos
for column in sorted_headers_relevance + ['class']:
    data[column] = label_encoder.fit_transform(data[column])

# Separar o nosso alvo, resultado final, classificacao de diabetes
y = data['class']


# Captura todas colunas com excessão do alvo
X = data[sorted_headers_relevance]
# Separar os dados em treino e teste usando estratificação pela "class"
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

# Seleciona o algoritmo
modelo_generico = GaussianNB()
# Treina o modelo
modelo_generico.fit(X_train, y_train)
# Calcula a predição do modelo
y_pred = modelo_generico.predict(X_test)

# Exibir o relatório de classificação
relatorio_classificacao = classification_report(y_test, y_pred)
print(relatorio_classificacao)
#               precision    recall  f1-score   support
# 
#            0       0.90      0.90      0.90        60
#            1       0.94      0.94      0.94        96
# 
#     accuracy                           0.92       156
#    macro avg       0.92      0.92      0.92       156
# weighted avg       0.92      0.92      0.92       156



# Armazenar os resultados de recall para serem exibidos posteriormente
# nos graficos
recall_class_0 = []
recall_class_1 = []
model_accuracy = []
f1_score_ls = []
# Iterar sobre os índices das colunas para formar diferentes subconjuntos
for i in range(1, len(sorted_headers_relevance) + 1):
    # Inicializar o modelo Naive Bayes Gaussiano
    gnb = GaussianNB()

    # Selecionar as primeiras i colunas
    X = data[sorted_headers_relevance[:i]]

    # Separar os dados em treino e teste usando estratificação pela "class"
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)

    # Treinar o modelo
    gnb.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = gnb.predict(X_test)

    # Avaliar o desempenho do modelo
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Salvar na lista juntamente com a quantidade de atributos utilizados
    recall_class_0.append((i, recall_0))
    recall_class_1.append((i, recall_1))
    model_accuracy.append((i, accuracy))
    f1_score_ls.append((i, f1))


# Controi datasets com as duas metricas
recall_df_0 = pd.DataFrame(recall_class_0, columns=['Number of Features', 'Recall Class 0'])
recall_df_1 = pd.DataFrame(recall_class_1, columns=['Number of Features', 'Recall Class 1'])
accuracy_df = pd.DataFrame(model_accuracy, columns=['Number of Features', 'Accuracy'])
f1_df = pd.DataFrame(f1_score_ls, columns=['Number of Features', 'F1 Score'])

# Cria um grafico
plt.figure(figsize=(10, 6))

# Adiciona as linhas relacionadas com o recall
plt.plot(recall_df_0['Number of Features'], recall_df_0['Recall Class 0'], label='Recall Class 0', marker='o', linestyle='-', color='b')
plt.plot(recall_df_1['Number of Features'], recall_df_1['Recall Class 1'], label='Recall Class 1', marker='o', linestyle='-', color='r')
plt.plot(accuracy_df['Number of Features'], accuracy_df['Accuracy'], label='Accuracy', marker='o', linestyle='-', color='g')
plt.plot(f1_df['Number of Features'], f1_df['F1 Score'], label='F1 Score', marker='o', linestyle='-', color='y')

plt.title('Recall vs Number of Features for Class 0 and Class 1')
plt.xlabel('Number of Features')
plt.ylabel('Metrics')
plt.grid(True)
plt.legend()

plt.plot()
