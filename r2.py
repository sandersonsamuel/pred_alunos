import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(1)
n = 100
temperatura = np.random.normal(25, 5, n)
consumo = 30 + 5 * temperatura + np.random.normal(0, 10, n)

data = pd.DataFrame({'temperatura': temperatura, 'consumo': consumo})

X = data[['temperatura']]
y = data['consumo']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
print(f"R²: {r2:.4f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Consumo Real (kWh)')
plt.ylabel('Consumo Previsto (kWh)')
plt.title('Previsão de Consumo de Energia vs. Real')
plt.grid(True)
plt.show()
