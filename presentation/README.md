# Apresentação da Solução

# Resumo do Projeto

Este projeto teve como objetivo desenvolver e avaliar diferentes modelos de machine learning para a classificação de diabetes, priorizando a redução de falsos negativos, dada a criticidade de diagnósticos incorretos nesse contexto.

## Etapas Desenvolvidas

1. **Preparação dos Dados**: 
   - Transformação de variáveis categóricas binárias em valores numéricos usando _label encoding_.
   - Divisão do conjunto de dados em 70% para treinamento e 30% para teste, garantindo uma avaliação robusta.
   - Ajustes nos dados para avaliar possíveis melhorias nos modelos

2. **Modelos Criados**:
   - **Naive Bayes**: Utilizou o GaussianNB para lidar com a variável contínua "idade" sem perda de detalhes. O impacto de cada atributo foi avaliado iterativamente, com destaque para o recall como métrica principal.
    - **Árvore de Decisão**: Foram realizados dois testes principais:
       - Com **SMOTE**, para balancear classes e otimizar a performance.
       - Com **estratificação** no conjunto de dados, para manter a proporção de classes.
   - **Random Forest**: Implementado com 100 árvores de decisão, ajustando parâmetros para melhorar o desempenho geral. O foco foi na análise de recall, precisão e F1-Score.
   Ambos os testes **Árvore de Decisão** e **Random Forest** exploraram validação cruzada e ajuste de hiperparâmetros via GridSearchCV.

4. **Avaliação dos Modelos**:
   - A principal métrica utilizada foi o recall, devido à necessidade de minimizar falsos negativos. Outras métricas como precisão e F1-Score foram empregadas para complementar a análise.
   - Comparação detalhada do desempenho dos modelos em cenários variados, destacando as contribuições de cada abordagem para a classificação.

## Resultado Final

A análise dos modelos evidenciou que:
- O Naive Bayes mostrou-se eficiente para identificar padrões simples e foi mais robusto com atributos bem selecionados.
- O Random Forest apresentou maior estabilidade em cenários complexos.
- As Árvores de Decisão, com ajustes apropriados, mostraram-se versáteis e eficazes.

## Vídeo de Apresentação Final

- Acesse o link abaixo e confira os testes realidados com a aplicação em produção.

[Assista ao vídeo aqui]([https://www.youtube.com/watch?v=D9qi_KkpsgI](https://drive.google.com/file/d/1_AtAd2CNUHjQ2A_dXcsG3zslaOm76ls-/view?usp=sharing)).

