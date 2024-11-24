import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'normalize', 'data'))
data_minmax_path = os.path.join(data_root, 'data_minmax.csv')
data_theta_path = os.path.join(data_root, 'theta.txt')

# Чтение данных
data = pd.read_csv(data_minmax_path)

# Подготовка признаков и целевой переменной
X = data.drop(columns=["price"])  # Все признаки, кроме цены
y = data["price"].values  # Цена - целевая переменная

# Столбец единиц (для смещения)
X = np.c_[np.ones(X.shape[0]), X]


# Функция предсказания
def predict(X, w):
    return np.dot(X, w)


# Функция вычисления стоимости
def compute_cost(X, y, w):
    m = len(y)
    predictions = predict(X, w)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# Градиентный спуск с добавлением ранней остановки
def gradient_descent(X, y, w, learning_rate, iterations, tolerance=1e-6, early_stop_iter=50):
    m = len(y)
    cost_history = []
    prev_cost = float('inf')
    for i in range(iterations):
        predictions = predict(X, w)
        gradient = (1 / m) * np.dot(X.T, (predictions - y))

        # Обновляем веса
        w -= learning_rate * gradient

        # Стоимость на текущей итерации
        cost = compute_cost(X, y, w)
        cost_history.append(cost)

        # Проверка на раннюю остановку
        if abs(prev_cost - cost) < tolerance:
            print(f"Ранняя остановка на итерации {i}. Разница в ошибке: {abs(prev_cost - cost)}")
            break

        prev_cost = cost

        # Дополнительно проверяем на большие градиенты
        if np.any(np.abs(gradient) > 1e10):
            print("Градиенты слишком большие, остановка обучения.")
            break

    return w, cost_history


# Подбор наилучшей скорости обучения с ранней остановкой
learning_rates = [0.001, 0.01, 0.1, 0.5,0.7,1]
best_lr = None
min_cost = float('inf')
best_w = None
best_cost_history = None

for lr in learning_rates:
    initial_weights = np.zeros(X.shape[1])
    w, cost_history = gradient_descent(X, y, initial_weights, lr, 50000)
    final_cost = cost_history[-1]

    if final_cost < min_cost:
        min_cost = final_cost
        best_lr = lr
        best_w = w
        best_cost_history = cost_history

# Сохранение лучших весов модели
np.savetxt(data_theta_path, best_w)

print(f'Лучший коэффициент обучения: {best_lr}')
print(f'Лучшие веса модели: {best_w}')

# Построение графика ошибки
plt.plot(best_cost_history)
plt.xlabel('Итерации')
plt.ylabel('Ошибка')
plt.title(f'График функции стоимости для lr={best_lr}')
plt.show()
