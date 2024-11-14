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
Para iterar em cima das colunas disponíveis até e apresentar uma análise gráfica, foi necessário utilizar a estrutura `for` para representar a quantidade de atributos a serem utilizados. O loop irá começar com 1 coluna e vai ir até a 16º coluna.

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

## Decision Tree

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

### Decision Tree

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

### Decision Tree

# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar  os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".

Além disso, deverá ser entregue um vídeo onde deverão ser descritas todas as etapas realizadas. O vídeo, que não tem limite de tempo, deverá ser apresentado por **todos os integrantes da equipe**, de forma que, cada integrante tenha oportunidade de apresentar o que desenvolveu e as  percepções obtidas.
