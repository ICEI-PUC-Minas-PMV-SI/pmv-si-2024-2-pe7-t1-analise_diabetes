import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Carregar o CSV
df = pd.read_csv('./dataset-full.csv')

# Converter as variáveis categóricas Yes/No e Positive/Negative para 1/0
df.replace({'Yes': 1, 'No': 0, 'Positive': 1, 'Negative': 0, 'Male': 1, 'Female': 2}, inplace=True)

# Função para calcular a correlação de Cramer's V para variáveis categóricas
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return (chi2 / (n * (min(contingency_table.shape) - 1))) ** 0.5

# Calcular a correlação entre todas as variáveis categóricas
correlation_matrix = df.apply(lambda x: df.apply(lambda y: cramers_v(x, y)))

# Plotar o heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap de Correlações Categóricas')
plt.savefig('heatmap.png')




# Calcular a relevância de cada campo em relação à variável Class
class_correlations = {col: cramers_v(df[col], df['class']) for col in df.columns if col != 'class'}
# Ordenar os campos pela relevância
sorted_relevance = sorted(class_correlations.items(), key=lambda item: item[1], reverse=True)

# Output das variáveis e suas correlações
print("Relevância das variáveis em relação à variável 'class':")
for variable, relevance in sorted_relevance:
    print(f"{variable}: {relevance:.4f}")

