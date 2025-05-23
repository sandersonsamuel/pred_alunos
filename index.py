import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# criando um dataset fictício de alunos
np.random.seed(0)
n = 100
data = pd.DataFrame({
    'horas_estudo': np.random.normal(5, 1.5, n),
    'faltas': np.random.randint(0, 10, n),
    'nota_anterior': np.random.normal(6, 1.0, n),
    'participacao': np.random.uniform(0, 1, n),
})

# nota final simulada com ruído
data['nota_final'] = (
    0.4 * data['horas_estudo'] +
    -0.2 * data['faltas'] +
    0.3 * data['nota_anterior'] +
    2 * data['participacao'] +
    np.random.normal(0, 1, n)
).clip(0, 10)  # notas de 0 a 10

# X e y
X = data[['horas_estudo', 'faltas', 'nota_anterior', 'participacao']]
y = data['nota_final']

# treinar o modelo
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# calcular métricas
mape = mean_absolute_percentage_error(y, y_pred) * 100
r2 = r2_score(y, y_pred)
n_samples = X.shape[0]
n_features = X.shape[1]
r2_adjusted = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# exibir resultados
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")
print(f"R² Ajustado: {r2_adjusted:.4f}")

# valores reais vs previstos
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y, y=y_pred)
plt.xlabel('Nota Real')
plt.ylabel('Nota Prevista')
plt.title('Previsão de Notas de Alunos')
plt.grid(True)
plt.show()
