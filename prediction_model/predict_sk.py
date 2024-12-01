import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Пути к данным
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'normalize', 'data'))
data_minmax_path = os.path.join(data_root, 'data_minmax.csv')
data_theta_path = os.path.join(data_root, 'theta_sk.txt')

# Чтение данных
data = pd.read_csv(data_minmax_path)

# Подготовка признаков и целевой переменной
X = data.drop(columns=["price"])  # Все признаки, кроме цены
y = data["price"].values  # Цена - целевая переменная

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Инициализация модели линейной регрессии
model = LinearRegression()

# Обучение модели
model.fit(X_train_scaled, y_train)

# Сохранение лучших весов модели
best_w = np.concatenate(([model.intercept_], model.coef_))
np.savetxt(data_theta_path, best_w)

# Оценка модели на тестовых данных
predictions = model.predict(X_test_scaled)
final_cost = np.mean((predictions - y_test) ** 2)

print(f'Лучшие веса модели: {best_w}')
print(f'Ошибка на тестовых данных (MSE): {final_cost}')

# Построение графика ошибки
plt.plot(np.arange(len(y_test)), predictions - y_test)
plt.xlabel('Индекс тестового примера')
plt.ylabel('Ошибка предсказания')
plt.title('Ошибка предсказания модели')
plt.show()

