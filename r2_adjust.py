import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(2)
n = 100
data = pd.DataFrame({
    'horas_estudo': np.random.normal(5, 1.5, n),
    'faltas': np.random.randint(0, 10, n),
    'nota_anterior': np.random.normal(6, 1.0, n),
    'participacao': np.random.uniform(0, 1, n),
})
data['nota_final'] = (
    0.4 * data['horas_estudo'] +
    -0.2 * data['faltas'] +
    0.3 * data['nota_anterior'] +
    2 * data['participacao'] +
    np.random.normal(0, 1, n)
).clip(0, 10)

X = data[['horas_estudo', 'faltas', 'nota_anterior', 'participacao']]
y = data['nota_final']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
n_samples = X.shape[0]
n_features = X.shape[1]
r2_adjusted = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

print(f"R²: {r2:.4f}")
print(f"R² Ajustado: {r2_adjusted:.4f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Nota Real')
plt.ylabel('Nota Prevista')
plt.title('Previsão de Nota Final de Alunos')
plt.grid(True)
plt.show()

