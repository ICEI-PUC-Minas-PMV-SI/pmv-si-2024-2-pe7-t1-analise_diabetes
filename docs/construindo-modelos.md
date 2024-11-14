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

Para a preparação dos dados no treinamento de modelos de machine learning, foram seguidos dois passos principais:

1. Codificação de categorias binárias
As características categóricas foram transformadas em valores numéricos para facilitar o cálculo de probabilidades pelos algoritmos. Campos como "Sim" e "Não" foram codificados como 1 e 0, respectivamente, e a mesma lógica foi aplicada às variáveis "classe" (positivo e negativo para risco de diabetes) e "gênero" (masculino e feminino).

2. Separação dos dados
O conjunto de dados foi dividido em 70% para treinamento e 30% para teste, permitindo que o modelo aprenda com a maior parte dos dados enquanto é testado em dados inéditos.

O Código a seguir feito na linguagem python foi usado como base para a preparação dos dados de maneira geral. 
</br>
</br>
[Link para Código de preparação Simplificado](/src/dataPreparation.py)

# Descrição dos modelos

<!-- Guid do Doc -->
<!--
Nesta seção, conhecendo os dados e de posse dos dados preparados, é hora de descrever os algoritmos de aprendizado de máquina selecionados para a construção dos modelos propostos. Inclua informações abrangentes sobre cada algoritmo implementado, aborde conceitos fundamentais, princípios de funcionamento, vantagens/limitações e justifique a escolha de cada um dos algoritmos. 

Explore aspectos específicos, como o ajuste dos parâmetros livres de cada algoritmo. Lembre-se de experimentar parâmetros diferentes e principalmente, de justificar as escolhas realizadas.

Como parte da comprovação de construção dos modelos, um vídeo de demonstração com todas as etapas de pré-processamento e de execução dos modelos deverá ser entregue. Este vídeo poderá ser do tipo _screencast_ e é imprescindível a narração contemplando a demonstração de todas as etapas realizadas. -->

- <h3>Naive Bayers</h3>
O algoritmo Naive Bayes, baseado no Teorema de Bayes, foi selecionado por tratar todas as variáveis de entrada como independentes entre si, mesmo que, na prática, essa suposição nem sempre seja válida. Essa simplicidade torna o Naive Bayes um modelo atrativo, rápido e eficiente para tarefas de classificação, especialmente com dados categóricos e binários, como os presentes no dataset analisado.

Foi utilizado o objeto GaussianNB, que assume que os atributos seguem uma distribuição normal para cada classe. Como o dataset inclui a variável "idade" e, para preservar os detalhes, optou-se por não convertê-la em faixas etárias, esse algoritmo se mostrou adequado, dado que a "idade" apresenta uma distribuição aproximadamente normal.

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
  
Devido à presença de diversos atributos, para analisar até que ponto o aumento deles influencia a qualidade do modelo, será adotada a seguinte estratégia:

1. Selecionar o 1º atributo
2. Treinar o modelo utilizando apenas esse atributo.
3. Verificar o valor do recall.
4. Retornar à etapa 1, adicionando o próximo atributo na sequência
5. Repetir o fluxo até que todos os atributos sejam utilizados.

Com essa estratégia, o modelo é treinado a cada iteração com um número crescente de atributos, permitindo acompanhar como o aumento do número de variáveis influencia nas métricas de desempenho.

O código abaixo, desenvolvido em Python, foi utilizado para realizar o treinamento
</br>
</br>
[Link para Código](/src/NaiveBayers.py)

- <h3>Random Forest</h3>

- <h3>Decision Tree</h3>

# Avaliação dos modelos criados

## Métricas utilizadas

<!--Nesta seção, as métricas utilizadas para avaliar os modelos desenvolvidos deverão ser apresentadas (p. ex.: acurácia, precisão, recall, F1-Score, MSE etc.). A escolha de cada métrica deverá ser justificada, pois esta escolha é essencial para avaliar de forma mais assertiva a qualidade do modelo construído. -->

- <h3>Naive Bayers</h3>

A escolha do recall como métrica principal para a classificação de diabetes é justificada pelo fato de que, nesse contexto, o objetivo é minimizar falsos negativos. Um falso negativo indicaria que uma pessoa com diabetes seria incorretamente classificada como não diabética, o que poderia resultar em consequências graves, uma vez que o tratamento adequado não seria administrado.

O recall avalia a capacidade do modelo de detectar corretamente todos os casos positivos (diabetes), medindo a proporção de verdadeiros positivos em relação ao total de casos que realmente são positivos. Um recall elevado indica que o modelo é eficiente em identificar a maioria dos indivíduos com diabetes, reduzindo ao máximo a ocorrência de falsos negativos.

Razões principais para usar o recall:

1. Impacto crítico de falsos negativos: No diagnóstico de doenças como o diabetes, deixar de identificar uma pessoa doente pode levar a complicações de saúde sérias e ao agravamento da doença.
2. Prioridade na detecção de casos positivos: O recall foca em identificar corretamente todos os casos de diabetes, mesmo que isso aumente a chance de alguns falsos positivos, o que é mais aceitável nesse cenário. Portanto, o uso do recall é adequado quando a minimização de falsos negativos é a prioridade, como no caso da classificação de doenças como diabetes.

- <h3>Random Forest</h3>

- <h3>Decision Tree</h3>

## Discussão dos resultados obtidos

<!--Nesta seção, discuta os resultados obtidos pelos modelos construídos, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade de cada um deles. Lembre-se de relacionar os resultados obtidos ao problema identificado, a questão de pesquisa levantada e estabelecendo relação com os objetivos previamente propostos. -->

- <h3>Naive Bayers</h3> 

O gráfico abaixo mostra o desempenho de um modelo de Naive Bayes em função do número de características usadas, com quatro métricas principais plotadas: Revocação para a Classe 0, Revocação para a Classe 1, Acurácia e F1 Score.

</br> </br>![Resultado Naive Byers](/docs/img/NbMetrics.png) </br> </br>

A análise será direcionada para as linhas vermelhas e azuis que representam os atributos de recall.

Inicialmente, o recall para a Classe 0 apresenta um ligeiro aumento, atingindo o pico em torno de 3 características. Após esse ponto, há uma queda acentuada no recall quando o número de características chega a aproximadamente 5, seguido por uma estabilização em um nível mais baixo à medida que mais características são incluídas. Esse comportamento indica que o aumento no número de características não contribui para melhorar o recall da Classe 0 após certo ponto, possivelmente devido a overfitting ou à inclusão de características que não são informativas para essa classe.

Por outro lado, o recall para a Classe 1 geralmente melhora à medida que mais características são adicionadas, atingindo um pico entre 3 e 6 características e, em seguida, estabilizando-se em um nível elevado. Isso sugere que o modelo se torna mais eficaz em identificar instâncias da Classe 1 com a adição de mais atributos, embora essa melhora eventual se estabilize, indicando que atributos adicionais oferecem menos benefícios após um determinado ponto.

Portanto, a análise sugere que o uso de aproximadamente 3 a 6 características é ideal para o classificador Naive Bayes, proporcionando um bom equilíbrio com alta revocação para a Classe 1, estabilidade na acurácia e no F1 Score, sem uma perda significativa de revocação para a Classe 0.

- <h3>Random Forest</h3>

- <h3>Decision Tree</h3>

# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar  os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".

Além disso, deverá ser entregue um vídeo onde deverão ser descritas todas as etapas realizadas. O vídeo, que não tem limite de tempo, deverá ser apresentado por **todos os integrantes da equipe**, de forma que, cada integrante tenha oportunidade de apresentar o que desenvolveu e as  percepções obtidas.
