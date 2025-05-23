import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n_months = 50

data = pd.DataFrame({
    'mes': np.arange(1, n_months + 1),
    'gastos_marketing': np.random.uniform(5000, 20000, n_months),  
    'num_clientes': np.random.randint(200, 1000, n_months),       
    'dias_abertos': np.random.randint(20, 31, n_months)            
})

data['faturamento'] = (
    50 * data['gastos_marketing'] / 1000 + 
    30 * data['num_clientes'] + 
    500 * data['dias_abertos'] +
    np.random.normal(0, 1000, n_months)
).clip(lower=0)

X = data[['gastos_marketing', 'num_clientes', 'dias_abertos']]
y = data['faturamento']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mape = mean_absolute_percentage_error(y, y_pred) * 100
print(f"MAPE: {mape:.2f}%")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y, y=y_pred)
plt.xlabel('Faturamento Real (R$)')
plt.ylabel('Faturamento Previsto (R$)')
plt.title('Previsão de Faturamento Mensal da Loja')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.grid(True)
plt.show()
