# Preparação dos dados

<!-- Guia do Doc -->
<!--
Nesta etapa, deverão ser descritas todas as técnicas utilizadas para pré-processamento/tratamento dos dados.

Algumas das etapas podem estar relacionadas à:

* Limpeza de Dados: trate valores ausentes: decida como lidar com dados faltantes, seja removendo linhas, preenchendo com médias, medianas ou usando métodos mais avançados; remova _outliers_: identifique e trate valores que se desviam significativamente da maioria dos dados.

* Transformação de Dados: normalize/padronize: torne os dados comparáveis, normalizando ou padronizando os valores para uma escala específica; codifique variáveis categóricas: converta variáveis categóricas em uma forma numérica, usando técnicas como _one-hot encoding_.

* _Feature Engineering_: crie novos atributos que possam ser mais informativos para o modelo; selecione características relevantes e descarte as menos importantes.

* Tratamento de dados desbalanceados: se as classes de interesse forem desbalanceadas, considere técnicas como _oversampling_, _undersampling_ ou o uso de algoritmos que lidam naturalmente com desbalanceamento.

* Separação de dados: divida os dados em conjuntos de treinamento, validação e teste para avaliar o desempenho do modelo de maneira adequada.
  
* Manuseio de Dados Temporais: se lidar com dados temporais, considere a ordenação adequada e técnicas específicas para esse tipo de dado.
  
* Redução de Dimensionalidade: aplique técnicas como PCA (Análise de Componentes Principais) se a dimensionalidade dos dados for muito alta.

* Validação Cruzada: utilize validação cruzada para avaliar o desempenho do modelo de forma mais robusta.

* Monitoramento Contínuo: atualize e adapte o pré-processamento conforme necessário ao longo do tempo, especialmente se os dados ou as condições do problema mudarem.

* Entre outras....

Avalie quais etapas são importantes para o contexto dos dados que você está trabalhando, pois a qualidade dos dados e a eficácia do pré-processamento desempenham um papel fundamental no sucesso de modelo(s) de aprendizado de máquina. É importante entender o contexto do problema e ajustar as etapas de preparação de dados de acordo com as necessidades específicas de cada projeto. -->
## Naive bayes

Para a preparação dos dados no treinamento de modelos de machine learning, foram seguidos dois passos principais:

### Codificação de categorias
Para codificar as categorias binárias foi usado a estratégia de _label encoding_, onde colunas categóricas serão convertidas para atributos numéricos
```python3
# Aqui as colunas são normalizadas para o processamento seguinte
data = pd.read_csv('../dataset-full.csv')
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
........
# Aqui um objeto da classe LabelEncoder é criado, ele é o responsável por converter os atributos Yes e No para 0 e 1 respectivamente
label_encoder = LabelEncoder()

# O gênero é convertido manualmente pois o a classe de processamento LabelEncoder não suporta nativamente Male e Female
data['gender'] = data['gender'].map({'Male': 1, 'Female': 2})

# Finalmente ao realizar um loop na lista de atributos do dataset todos são categorizados
for column in sorted_headers_relevance + ['class']:
    data[column] = label_encoder.fit_transform(data[column])
```

As características categóricas foram transformadas em valores numéricos para facilitar o cálculo de probabilidades pelos algoritmos. Campos como "Sim" e "Não" foram codificados como 1 e 0, respectivamente, e a mesma lógica foi aplicada às variáveis "classe" (positivo e negativo para risco de diabetes) e "gênero" (masculino e feminino).

Sobre o gênero é importante notar que foi definido Male = 1 e Female = 2, essa escolha não apresenta efeitos colaterais na análise, desde que os rótulos sejam respeitados durante todo o ciclo de vida do modelo.

A idade foi preservada e não foi convertida para uma categoria.

### Separação dos dados
O conjunto de dados foi dividido em 70% para treinamento e 30% para teste, permitindo que o modelo aprenda com a maior parte dos dados enquanto é testado em dados inéditos.

```python3
 X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
```
Utilizando da função `train_test_split` que é capaz de separar a base de dados em uma porção de teste e outra porção para validação. Do parâmetros utilizados os mais importantes são:

- train_size: Descreve o valor da proporção em que o dataset irá ser quebrado, nesse caso o valor 0.7, representa o mesmo que 70%. Logo 70% será separado para treinamento e 30% para validação.
- stratify: Realiza amostragem estratificada pela classificação de diabetes. isso significa que a mesma proporção de diagnósticos positivos e negativos que existe no dataset completo será utilizado na amostra de testes e consequentemente na amostra de validação.
- random_state: É uma semente utilizada para construir números pseudoaleatórios, sendo assim em toda execução os mesmos dados de testes serão extraídos.


</br>
</br>

[Link para Código de preparação Simplificado](/src/dataPreparation.py)

# Descrição dos modelos

<!-- Guid do Doc -->
<!--
Nesta seção, conhecendo os dados e de posse dos dados preparados, é hora de descrever os algoritmos de aprendizado de máquina selecionados para a construção dos modelos propostos. Inclua informações abrangentes sobre cada algoritmo implementado, aborde conceitos fundamentais, princípios de funcionamento, vantagens/limitações e justifique a escolha de cada um dos algoritmos. 

Explore aspectos específicos, como o ajuste dos parâmetros livres de cada algoritmo. Lembre-se de experimentar parâmetros diferentes e principalmente, de justificar as escolhas realizadas.

Como parte da comprovação de construção dos modelos, um vídeo de demonstração com todas as etapas de pré-processamento e de execução dos modelos deverá ser entregue. Este vídeo poderá ser do tipo _screencast_ e é imprescindível a narração contemplando a demonstração de todas as etapas realizadas. -->

## Naive bayes
O algoritmo Naive Bayes, baseado no Teorema de Bayes, foi selecionado por tratar todas as variáveis de entrada como independentes entre si, mesmo que, na prática, essa suposição nem sempre seja válida. Essa simplicidade torna o Naive Bayes um modelo atrativo, rápido e eficiente para tarefas de classificação, especialmente com dados categóricos e binários, como os presentes no dataset analisado.

Foi utilizado o objeto GaussianNB, que assume que os atributos seguem uma distribuição normal para cada classe.
```python3
gnb = GaussianNB()
```
Isso foi escolhido para preservar os detalhes relacionados ao atributo idade, já que ele não é uma categoria e a sua conversão em faixas etárias causaria a perda de detalhes. É importante deixar claro que esse algoritmo somente se mostrou adequado nesse caso pois o atributo "idade" apresenta uma distribuição normal, atributos com uma variação muito grande em sua distribuição não seriam adequados.

A análise do heatmap indicou a necessidade de ordenar as colunas de acordo com o valor de correlação com a classificação de diabetes.
</br> </br>![Heatmap correlação variáveis](/docs/img/heatmap.png) </br> </br>
Seguindo essa lógica, as variáveis foram ordenadas com base no valor de correlação com a classificação de diabetes, priorizando aquelas que apresentaram maior influência na determinação da presença ou ausência da condição. A lista abaixo reflete essa priorização:

1. polyuria
2. polydipsia
3. age
4. gender
5. sudden_weight_loss
6. partial_paresis
7. polyphagia
8. irritability
9. alopecia
10. visual_blurring
11. weakness
12. muscle_stiffness
13. genital_thrush
14. obesity
15. delayed_healing
16. itching

Para construir essa lista foi necessário carregar o dataset em memória utilizando o pandas
```python3
df = pd.read_csv('./dataset-full.csv')
```

Converter as variáveis categóricas como gênero e classificações binárias (Sim/Não) para valores numéricos
```python3
df.replace({'Yes': 1, 'No': 0, 'Positive': 1, 'Negative': 0, 'Male': 1, 'Female': 2}, inplace=True)
```

E finalmente calcular a matriz de correlação entre todas as variáveis

```python3
# Define a função que vai calcular a correlação entre duas variáveis tendo como entrada duas listas com valores medidos
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return (chi2 / (n * (min(contingency_table.shape) - 1))) ** 0.5

# Para cada coluna do dataset aplica a validação com todas as colunas do dataset
# Dessa forma todos os atributos serão testados com todos os atributos
# Essa parte é essencial para construir o heatmap porém é descartável para construir a lista
correlation_matrix = df.apply(lambda x: df.apply(lambda y: cramers_v(x, y)))
```

Então é calculado a correlação para todas as colunas do dataset em relação ao diagnóstico com excessão do proprio diagnóstico
```python3
class_correlations = {col: cramers_v(df[col], df['class']) for col in df.columns if col != 'class'}
```

O resultado é um dicionário relacionando a coluna ao valor de correlação com o diagnóstico, então basta ordenar e exibir os valores
```python3
sorted_relevance = sorted(class_correlations.items(), key=lambda item: item[1], reverse=True)

print("Relevância das variáveis em relação à variável 'class':")
for variable, relevance in sorted_relevance:
    print(f"{variable}: {relevance:.4f}")
```

Associando a ordem com suas respectivas colunas o seguinte ranking é formado:
```python3
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

```

Devido à presença de diversos atributos, para analisar até que ponto o aumento deles influencia a qualidade do modelo, será adotada a seguinte estratégia:
1. Selecionar o 1º atributo
2. Treinar o modelo utilizando apenas esse atributo.
3. Verificar o valor do recall.
4. Retornar à etapa 1, adicionando o próximo atributo na sequência
5. Repetir o fluxo até que todos os atributos sejam utilizados.

### Normalizando colunas do dataset
1. Remover espaços, tabulações e quebras de linhas das colunas
2. Converter todos os caracteres para minúsculos
3. Substituir espaços em branco por underscore
4. Remover caracteres de parênteses
```python3
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
```
No final a saída disso é o nome de todas as colunas no formato [snake_case](https://developer.mozilla.org/en-US/docs/Glossary/Snake_case), isso vai ser essencial ao comparar com o ranking de atributos relacionados com o diagnóstico.


### Iteração consumindo os atributos
Para iterar sobre as colunas disponíveis e apresentar uma análise gráfica, foi utilizada a estrutura `for`, que permite controlar a quantidade de atributos a serem utilizados no modelo. O loop começa com 1 coluna e segue até a 16ª coluna, incrementando a cada iteração. Essa abordagem permite treinar o modelo progressivamente, ajustando o número de atributos e avaliando o impacto nas métricas de desempenho, como o recall. O gráfico gerado ao final mostrará como a inclusão de atributos adicionais influencia a performance do modelo à medida que mais colunas são adicionadas.

```python3
for i in range(1, len(sorted_headers_relevance) + 1):
```

Todas as operações a seguir são feitas dentro do loop.
Para garantir um modelo sempre limpo, para a execução anterior não atrapalhar nas métricas, dentro do loop é criado o objeto que representa o algoritmo de Naive Bayes.
```python3
gnb = GaussianNB()
```

E utilizando a notação de lista do python é possível pegar as primeiras colunas e extrair elas do dataframe
```python3
X = data[sorted_headers_relevance[:i]]
```

Separa as amostras de teste e de validação
```python3
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
```

Utiliza a função fit do algoritmo GaussianNB para transformar os dados, calcular as taxas de erro e ajustar os parâmetros internos. Todos esses passos são abstraídos pela biblioteca.
```python3
gnb.fit(X_train, y_train)
```

E agora com o modelo pronto, deve ser analisado como ele se comporta com novos dados. Então o restante dos dados é utilizado para validação.
```python3
y_pred = gnb.predict(X_test)
```

Com a predição do modelo é possível calcular as métricas de qualidade.
```python3
recall_0 = recall_score(y_test, y_pred, pos_label=0)
recall_1 = recall_score(y_test, y_pred, pos_label=1)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

As métricas são salvas em uma lista no formato (número de atributos, valor da métrica), isso será útil para construir o gráfico de evolução do modelo baseado na quantidade de atributos.
```python3
recall_class_0.append((i, recall_0))
recall_class_1.append((i, recall_1))
model_accuracy.append((i, accuracy))
f1_score_ls.append((i, f1))
```


### Exibição dos gráficos
Contendo várias listas que apresentam o formato (número de atributos, valor da métrica), elas devem ser convertidas para dataframes para permitir melhor integração com a biblioteca de gráficos.
```python3
recall_df_0 = pd.DataFrame(recall_class_0, columns=['Number of Features', 'Recall Class 0'])
recall_df_1 = pd.DataFrame(recall_class_1, columns=['Number of Features', 'Recall Class 1'])
accuracy_df = pd.DataFrame(model_accuracy, columns=['Number of Features', 'Accuracy'])
f1_df = pd.DataFrame(f1_score_ls, columns=['Number of Features', 'F1 Score'])
```


Define um tamanho para a imagem
```python3
plt.figure(figsize=(10, 6))
```

Adiciona as várias linhas no gráfico, diferenciando pela cor
```python3
plt.plot(recall_df_0['Number of Features'], recall_df_0['Recall Class 0'], label='Recall Class 0', marker='o', linestyle='-', color='b')
plt.plot(recall_df_1['Number of Features'], recall_df_1['Recall Class 1'], label='Recall Class 1', marker='o', linestyle='-', color='r')
plt.plot(accuracy_df['Number of Features'], accuracy_df['Accuracy'], label='Accuracy', marker='o', linestyle='-', color='g')
plt.plot(f1_df['Number of Features'], f1_df['F1 Score'], label='F1 Score', marker='o', linestyle='-', color='y')
```


Configura informações de legenda para melhor legibilidade do gráfico
```python3
plt.title('Recall vs Number of Features for Class 0 and Class 1')
plt.xlabel('Number of Features')
plt.ylabel('Metrics')
plt.grid(True)
plt.legend()
```

Exibe o gráfico de resultado
```python3
plt.plot()
```


O código completo pode ser conferido aqui [Link para Código](/src/NaiveBayers.py)

## Random Forest

Floresta Aleatória, ou Florestas de Decisão Aleatórias, é um método de aprendizado de conjunto utilizado para tarefas de classificação, regressão e outras análises preditivas [^1]. Uma hipótese para o uso do Random Forest é que sua capacidade de combinar múltiplas árvores de decisão pode gerar previsões mais estáveis e confiáveis, mesmo em cenários com dados complexos ou ruidosos [^2]. Dessa forma, optei por testar esta abordagem da seguinte forma:

```
%pip install pandas scikit-learn
%pip install graphviz pydotplus
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, classification_report
import pydotplus
from IPython.display import Image
import graphviz
```

Iniciei com a instalação das bibliotecas necessárias para manipulação de dados, criação do modelo de aprendizado de máquina Random Forest e visualização das árvores de decisão, utilizando ferramentas como pandas, scikit-learn, graphviz e pydotplus, além de integrar a função Image para exibição das visualizações no ambiente de desenvolvimento.

```
path = '/content/diabetes_risk_prediction_dataset.csv'

dados = pd.read_csv(path)

X = dados.drop('class', axis=1)
y = dados['class']
```
Carreguei o conjunto de dados a partir de um arquivo CSV para trabalhar com as informações sobre o risco de diabetes. Separei as colunas de dados em duas partes: as colunas com as características dos pacientes, como idade, níveis de glicose, pressão arterial, etc., que foram colocadas na variável X; e a coluna 'class', que indica se a pessoa tem ou não diabetes, que foi colocada na variável y. A coluna 'class' foi escolhida para y porque ela contém o que queremos prever: se a pessoa tem diabetes (1) ou não (0). Essa separação é necessária para que o modelo de aprendizado de máquina consiga aprender com as características dos pacientes e fazer previsões sobre o risco de diabetes com base nos dados fornecidos.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Dividi os dados em dois conjuntos: um para treinar o modelo e outro para testar o desempenho do modelo depois de treinado. Usei a função train_test_split para fazer essa divisão, onde 80% dos dados foram destinados ao treinamento do modelo e 20% foram reservados para testar as previsões do modelo, o que é especificado pelo parâmetro test_size=0.2. A divisão foi feita de forma aleatória, mas para garantir que os resultados sejam reproduzíveis, usei o parâmetro random_state=42, que fixa a semente da aleatoriedade. Dessa forma, qualquer pessoa que rodar o código terá a mesma divisão dos dados.

```
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
```
Neste trecho eu crio o modelo de aprendizado de máquina usando o algoritmo Random Forest. Defini que o modelo deveria ter 100 árvores (n_estimators=100), o que ajuda a melhorar a precisão das previsões. Além disso, fixei a semente da aleatoriedade com random_state=42, garantindo que os resultados possam ser reproduzidos. Em seguida, treinei o modelo utilizando os dados de treinamento (X_train e y_train), ou seja, fiz o modelo aprender a partir das características dos pacientes e seus respectivos diagnósticos de diabetes.

Importante dizer que a função `RandomForestClassifier` oferece outros parâmetros que podem ser ajustados para otimizar o desempenho do modelo. No entanto, neste momento, optei por utilizar os valores padrão para simplificar a configuração inicial e focar na implementação.

```
y_pred = modelo.predict(X_test)

precisao = accuracy_score(y_test, y_pred)
relatorio_classificacao = classification_report(y_test, y_pred)

print(f'A precisão do modelo é: {precisao:.2f}')
print("\nRelatório de Classificação:")
print(relatorio_classificacao)
```
Aqui eu utilizei o modelo treinado para fazer previsões com os dados de teste, armazenando os resultados na variável y_pred. Em seguida, calculei a precisão do modelo utilizando a função accuracy_score, que compara as previsões feitas (y_pred) com os valores reais de teste (y_test). Também gerei um relatório detalhado de classificação com a função classification_report, que fornece métricas como precisão, recall e F1-score para cada classe. Por fim, imprimi a precisão do modelo e o relatório de classificação para avaliar seu desempenho nas previsões do risco de diabetes.

É possível verificar o código completo [neste link](/src/random-forest.ipynb).


## Decision Tree
O tratamento e treinamento de dados, bem como a proporção de 70% para treinamento e 30% para teste foi usada exatamente igual ao modelo Naive bayes citado acima.
Para o primeiro teste utilizando árvore de decisão, foi utilizado o SMOTE para balanceamento dos dados.

# Colunas do dataset ordenadas pela relevância em relação à classificação
```python3
sorted_headers_relevance = [
    'polyuria', 'polydipsia', 'age', 'gender', 'sudden_weight_loss',
    'partial_paresis', 'polyphagia', 'irritability', 'alopecia', 'visual_blurring',
    'weakness', 'muscle_stiffness', 'genital_thrush', 'obesity', 'delayed_healing', 'itching'
]
```

# Carregar o CSV
```python3
data = pd.read_csv('../dataset-full.csv')
```

# Renomear os cabeçalhos para snake_case
```python3
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
```

# Tratar a coluna 'gender' manualmente (1 para Male, 2 para Female)
```python3
data['gender'] = data['gender'].map({'Male': 1, 'Female': 2})
```

# Converter outras colunas descritivas para numéricas com LabelEncoder
```python3
label_encoder = LabelEncoder()
```

# Utiliza o encoder para converter os dados de todas as colunas para numéricos
```python3
for column in sorted_headers_relevance + ['class']:
    data[column] = label_encoder.fit_transform(data[column])
```

# Separar o nosso alvo, resultado final, classificação de diabetes
```python3
y = data['class']
```

# captura colunas
```python3
X = data[sorted_headers_relevance]
```

# Separar os dados em treino e teste usando estratificação pela "class"
```python3
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,random_state=42)
```

### 2. Balanceamento das Classes (SMOTE) ###
```python3
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
```

# Inicializar o modelo de Árvore de Decisão
```python3
gnb = DecisionTreeClassifier(random_state=42)
```

# Treinar o modelo com dados balanceados
```python3
gnb.fit(X_train_bal, y_train_bal)
```

# Fazer previsões no conjunto de teste
```python3
y_pred = gnb.predict(X_test)
```

### 5. Cross-validation (Validação Cruzada) ###
```python3
cross_val_recall = cross_val_score(gnb, X, y, cv=5, scoring='recall')
cross_val_accuracy = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')

print(f"Mean Recall (Class 1) with Cross-Validation: {cross_val_recall.mean():.4f}")
print(f"Mean Accuracy with Cross-Validation: {cross_val_accuracy.mean():.4f}")
```

### 6. Tuning de Hiperparâmetros (GridSearchCV) ###
```python3
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='recall')
grid_search.fit(X_train_bal, y_train_bal)
```

# Imprimir os melhores parâmetros
```python3
print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
```

# Utilizando o melhor modelo encontrado
```python3
best_model = grid_search.best_estimator_

print(f"Melhor modelo encontrado: {best_model}")
```

# Prever com o melhor modelo no conjunto de teste
```python3
y_pred_best = best_model.predict(X_test)
```

### 7. Visualização da Árvore de Decisão ###
```python3
plt.figure(figsize=(12, 8))
tree.plot_tree(best_model, filled=True, feature_names=sorted_headers_relevance)
plt.show()
```

# Avaliar o desempenho do modelo
```python3
recall_0 = recall_score(y_test, y_pred_best, pos_label=0)
recall_1 = recall_score(y_test, y_pred_best, pos_label=1)
accuracy = accuracy_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)

print(f"Recall Class 0: {recall_0:.4f}")
print(f"Recall Class 1: {recall_1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
```

# Valores das métricas
```python3
metrics = {
    'Recall Class 0': recall_0,
    'Recall Class 1': recall_1,
    'Accuracy': accuracy,
    'F1 Score': f1
}
```

# Definir o título das barras e os valores
```python3
metric_names = list(metrics.keys())
metric_values = list(metrics.values())
```

# Criar o gráfico de barras
```python3
plt.figure(figsize=(10, 6))
plt.bar(metric_names, metric_values, color=['blue', 'red', 'green', 'yellow'])
```

# Configurações do gráfico
```python3
plt.title('Métricas de Desempenho do Classificador')
plt.xlabel('Métricas')
plt.ylabel('Valores')
plt.ylim(0.8, 1)  # Definir limite do eixo y já que as métricas são proporções e todas as medidas são acima de 80%
plt.grid(True, axis='y')
```

# Exibir o gráfico
```python3
plt.show()
```

# Avaliação dos modelos criados

## Métricas utilizadas

<!--Nesta seção, as métricas utilizadas para avaliar os modelos desenvolvidos deverão ser apresentadas (p. ex.: acurácia, precisão, recall, F1-Score, MSE etc.). A escolha de cada métrica deverá ser justificada, pois esta escolha é essencial para avaliar de forma mais assertiva a qualidade do modelo construído. -->

### Naive bayes

A escolha do recall como métrica principal para a classificação de diabetes é justificada pelo fato de que, nesse contexto, o objetivo é minimizar falsos negativos. Um falso negativo indicaria que uma pessoa com diabetes seria incorretamente classificada como não diabética, o que poderia resultar em consequências graves, uma vez que o tratamento adequado não seria administrado.

O recall avalia a capacidade do modelo de detectar corretamente todos os casos positivos (diabetes), medindo a proporção de verdadeiros positivos em relação ao total de casos que realmente são positivos. Um recall elevado indica que o modelo é eficiente em identificar a maioria dos indivíduos com diabetes, reduzindo ao máximo a ocorrência de falsos negativos.

Razões principais para usar o recall:

1. Impacto crítico de falsos negativos: No diagnóstico de doenças como o diabetes, deixar de identificar uma pessoa doente pode levar a complicações de saúde sérias e ao agravamento da doença.
2. Prioridade na detecção de casos positivos: O recall foca em identificar corretamente todos os casos de diabetes, mesmo que isso aumente a chance de alguns falsos positivos, o que é mais aceitável nesse cenário. Portanto, o uso do recall é adequado quando a minimização de falsos negativos é a prioridade, como no caso da classificação de doenças como diabetes.

### Random Forest
As métricas de precisão, recall e F1-Score foram escolhidas para avaliar o modelo Random Forest porque elas fornecem uma visão completa de seu desempenho. A precisão mostra a porcentagem de acertos nas previsões do modelo, enquanto o recall foca em quantos casos de diabetes o modelo conseguiu identificar corretamente. Já o F1-Score é uma média entre a precisão e o recall

### Decision Tree

# SMOTE
O SMOTE (Synthetic Minority Over-sampling Technique) é uma técnica de oversampling amplamente utilizada em problemas de aprendizado de máquina para lidar com datasets desbalanceados, onde uma classe (geralmente a classe minoritária) tem muito menos amostras do que outras.

## Como o SMOTE Funciona:
1. Identificação da Classe Minoritária: O SMOTE começa identificando a classe que está sub-representada no dataset. Por exemplo, em um problema de classificação binária com 1.000 exemplos da classe 0 (negativa) e apenas 100 exemplos da classe 1 (positiva), a classe 1 seria a classe minoritária.
2. Geração de Novas Amostras Sintéticas:
  - Em vez de simplesmente replicar exemplos da classe minoritária (como o oversampling simples faz), o SMOTE gera novos exemplos sintéticos.
  - Para cada ponto na classe minoritária, o SMOTE seleciona seus vizinhos mais próximos (k-vizinhos mais próximos, por padrão, k=5).
  - Ele então gera novos exemplos criando pontos intermediários entre o exemplo original e seus vizinhos. Isso significa que o SMOTE cria novas amostras "no meio do caminho" entre os exemplos reais da classe minoritária, em vez de simplesmente duplicá-los.
3. Novas Amostras para Equilibrar o Dataset: O processo é repetido até que o número de amostras da classe minoritária esteja mais equilibrado com o da classe majoritária.

## Etapas Visuais do SMOTE:
1. Dataset Original (Desbalanceado):
- Classe majoritária (muitos exemplos).
- Classe minoritária (poucos exemplos).
2. Aplicação do SMOTE:
- Novos pontos sintéticos são criados para a classe minoritária com base nos vizinhos mais próximos.
3. Dataset Final (Balanceado):
- Mais exemplos na classe minoritária, agora próximo ao número de exemplos da classe majoritária.


## Alguns pontos sobre a escolha
- Ao balancear o dataset, o SMOTE ajuda o modelo a aprender melhor as características da classe minoritária, o que melhora a capacidade de prever casos dessa classe.
- Diferente de oversampling simples (onde os exemplos são apenas copiados), o SMOTE gera exemplos novos e únicos, evitando que o modelo aprenda padrões repetitivos.
- O SMOTE pode levar ao overfitting, especialmente se o número de vizinhos próximos ou o número de novas amostras geradas for muito alto. Isso ocorre porque os exemplos sintéticos criados podem não capturar bem a variação dos dados reais.
- Pode criar exemplos irreais ou ambíguos, especialmente se os dados da classe minoritária forem muito dispersos ou se houver muito ruído.

Assim como do modelo Naive Bayes, foi utilizado o recall como métrica principal para classificação.

## Discussão dos resultados obtidos

<!--Nesta seção, discuta os resultados obtidos pelos modelos construídos, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade de cada um deles. Lembre-se de relacionar os resultados obtidos ao problema identificado, a questão de pesquisa levantada e estabelecendo relação com os objetivos previamente propostos. -->

### Naive bayes 

O gráfico abaixo mostra o desempenho de um modelo de Naive Bayes em função do número de características usadas, com quatro métricas principais plotadas: Revocação para a Classe 0, Revocação para a Classe 1, Acurácia e F1 Score.

</br> </br>![Resultado Naive Byers](/docs/img/NbMetrics.png) </br> </br>

A análise será direcionada para as linhas vermelhas e azuis que representam os atributos de recall.

Inicialmente, o recall para a Classe 0 apresenta um ligeiro aumento, atingindo o pico em torno de 3 características. Após esse ponto, há uma queda acentuada no recall quando o número de características chega a aproximadamente 5, seguido por uma estabilização em um nível mais baixo à medida que mais características são incluídas. Esse comportamento indica que o aumento no número de características não contribui para melhorar o recall da Classe 0 após certo ponto, possivelmente devido a overfitting ou à inclusão de características que não são informativas para essa classe.

Por outro lado, o recall para a Classe 1 geralmente melhora à medida que mais características são adicionadas, atingindo um pico entre 3 e 6 características e, em seguida, estabilizando-se em um nível elevado. Isso sugere que o modelo se torna mais eficaz em identificar instâncias da Classe 1 com a adição de mais atributos, embora essa melhora eventual se estabilize, indicando que atributos adicionais oferecem menos benefícios após um determinado ponto.

Portanto, a análise sugere que o uso de aproximadamente 3 a 6 características é ideal para o classificador Naive Bayes, proporcionando um bom equilíbrio com alta revocação para a Classe 1, estabilidade na acurácia e no F1 Score, sem uma perda significativa de revocação para a Classe 0.

### Random Forest
![Resultado Random Forest](https://github.com/user-attachments/assets/32370f24-8340-48d6-b4e0-a3bc09ef377c)

O resultado indica que o modelo apresenta um bom desempenho, com uma precisão geral de 99%, o que significa que ele acertou 99% das previsões. No relatório de classificação, para a classe 0 (sem diabetes), o modelo obteve uma precisão de 97%, recall de 100% e F1-score de 99%, indicando que ele identificou corretamente quase todas as pessoas sem diabetes, com poucos erros de classificação. Para a classe 1 (com diabetes), a precisão foi de 100%, recall de 99% e F1-score de 99%, demonstrando que o modelo foi eficaz em identificar corretamente as pessoas com diabetes, com uma pequena margem de erro. As médias ponderadas de precisão, recall e F1-score ficaram em 99%, o que sugere que o modelo teve um desempenho equilibrado e consistente em ambas as classes.

![Decision Tree](https://github.com/user-attachments/assets/ea92d959-7c2d-4aa4-9910-c510048ad97b)

```
dot_data = export_graphviz(
    arvore,
    out_file=None,
    feature_names=X.columns,
    class_names=['Sem Diabetes', 'Com Diabetes'],
    filled=True, rounded=True,
    special_characters=True
)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

```
Também foi criado um gráfico auxiliar para demonstrar a árvore de decisão utilizada pelo algoritmo. É possível notar que a classificação depende de diversas características, começando pela variável "Polyuria". Se "Polyuria" é menor ou igual a 0.5, o modelo analisa "Gender" e "Age", resultando principalmente em classificação como "No Diabets", destacado em azul. Se "Polyuria" é maior que 0.5, o foco passa para "Diabets" e "Polydipsia", com uma tendência a classificar como "Diabets", refletido em laranja.

Outro gráfico interessante foi criado para mostrar a importância de cada atributo para a correta identificação de casos de diabetes:

![Image](https://github.com/user-attachments/assets/2e62e1c8-a8a4-4872-abf5-8043dd62d98f)
```
import matplotlib.pyplot as plt
import numpy as np

importancias = modelo.feature_importances_

indices = np.argsort(importancias)[::-1]
nomes_features = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Importância dos atributos")
plt.barh(range(X.shape[1]), importancias[indices], align="center")
plt.yticks(range(X.shape[1]), nomes_features)
plt.xlabel("Importância")
plt.gca().invert_yaxis()
plt.show()

```
Ficou evidente que assim como haviamos feito as analises dos dados anteriormente, Polydipsia e Polyuria são os atributos mais determinantes para prever se uma pessoa tem ou não diabtes. No entanto, foi surpreendente para nós que a idade também é um fato decisivo, já que na etapa de conhecimento dos dados, esta informação não era óbvia de ser identificada.


### Decision Tree

O GridSearchCV é uma ferramenta que ajuda a encontrar os melhores hiperparâmetros para um modelo de aprendizado de máquina.
Para isso, ele executa uma busca exaustiva sobre uma grade (grid) de parâmetros, testando todas as combinações possíveis desses parâmetros, para identificar a melhor configuração para o modelo.

Para o teste definimos alguns parâmetros relacionados permitidos para ajuste e definimos como medida de qualidade o recall

Como nossa base de dados é pequena foi possível utilizar o método de busca exaustiva, o que talvez não seria possível em dados maiores.

No total, ele testará 2 × 4 × 3 = 24 combinações diferentes de hiperparâmetros e escolherá aquela que produzir o melhor desempenho para o modelo.

Vantagens:
- Otimização automática: Facilita a escolha dos melhores parâmetros.
- Validação cruzada: Minimiza o risco de overfitting, já que o modelo é testado em várias divisões dos dados.
- Desempenho aprimorado: Ao encontrar os melhores hiperparâmetros, o modelo pode ter um desempenho muito superior ao modelo inicial com parâmetros padrão.


## Parâmetros
1. **criterion: ['gini', 'entropy']** (Verificar outros parametros)
Esse parâmetro define o critério de divisão para escolher os nós nas árvores de decisão, ou seja, ele decide como a árvore divide os dados em cada nível.

    - Gini: Refere-se ao índice de Gini, que mede a impureza de um nó. Ele tenta minimizar a probabilidade de classificar incorretamente um item aleatório no nó. O Gini é mais simples e frequentemente mais rápido de calcular.
    - Entropy: Refere-se à entropia usada na teoria da informação. Ela mede o grau de desordem ou incerteza. O critério de entropia tenta maximizar a "informação" que cada divisão gera, e tende a resultar em árvores um pouco mais balanceadas, embora possa ser ligeiramente mais lento do que o Gini.

    **Impacto no Modelo**: Gini tende a ser mais eficiente em termos de tempo, mas entropy pode ser mais eficaz em termos de divisão da árvore quando há muitos atributos complexos. Dependendo do critério escolhido, a árvore pode produzir diferentes conjuntos de regras e, em última análise, diferentes classificações.
2. **max_depth: [None, 10, 20, 30]**
Este parâmetro define a profundidade máxima da árvore de decisão. Ele controla até que ponto a árvore pode crescer antes de parar.
    
    - None: Significa que a árvore será expandida até que todas as folhas sejam puras (100% corretas) ou contenham menos amostras do que o valor de min_samples_split.
    
    - 10, 20, 30: Limita o número máximo de níveis da árvore. Por exemplo, com max_depth=10, a árvore pode ter no máximo 10 níveis de profundidade.
    
    **Impacto no Modelo**:  
    - Profundidade maior: Quanto mais profunda a árvore, mais complexa ela será, o que pode resultar em overfitting, ou seja, o modelo se ajusta tão bem aos dados de treino que tem um desempenho ruim em novos dados.
    - Profundidade menor: Limitar a profundidade impede que a árvore cresça demais, o que ajuda a evitar o overfitting, mas pode resultar em underfitting se a árvore for muito rasa e não capturar padrões suficientes dos dados.

3.**min_samples_split: [2, 5, 10]**
Este parâmetro controla o número mínimo de amostras necessárias para dividir um nó. Se o número de amostras em um nó for menor que esse valor, o nó não será dividido.
    
    - 2: A divisão ocorre assim que houver pelo menos 2 amostras em um nó.
    - 5 ou 10: Exige um número maior de amostras para que uma divisão ocorra, o que significa que a árvore não se ramificará tão rapidamente.        

**Impacto no Modelo**:
- min_samples_split menor (2): A árvore se torna mais complexa, com mais divisões, o que pode levar a overfitting.
- min_samples_split maior (5 ou 10): A árvore se torna mais simples, dividindo-se menos vezes, o que ajuda a prevenir overfitting, mas pode resultar em underfitting se não dividir o suficiente para capturar padrões importantes.

### Resumo do Impacto Geral:
- criterion: Muda a maneira como as divisões são escolhidas (Gini é mais simples, Entropy é mais focado em informação).
- max_depth: Controla o quão profunda a árvore pode ser (profundidades maiores podem levar ao overfitting, menores ao underfitting).
- min_samples_split: Controla o número mínimo de amostras necessárias para dividir um nó, onde valores maiores tendem a fazer com que a árvore seja mais simples.

# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar  os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".

Além disso, deverá ser entregue um vídeo onde deverão ser descritas todas as etapas realizadas. O vídeo, que não tem limite de tempo, deverá ser apresentado por **todos os integrantes da equipe**, de forma que, cada integrante tenha oportunidade de apresentar o que desenvolveu e as  percepções obtidas.


# Referências
[^1]: [Floresta aleatória](https://pt.wikipedia.org/wiki/Floresta_aleat%C3%B3ria)
[^2]: [What is random forest?](https://pt.wikipedia.org/wiki/Floresta_aleat%C3%B3ria)
