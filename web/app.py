from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)

# Путь к данным
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../normalize', 'data'))
data_minmax_values = os.path.join(data_root, 'min_max_values.csv')
data_theta_path = os.path.join(data_root, 'theta.txt')

# Центр города (например, Москва)
CITY_LAT = 55.7558
CITY_LON = 37.6173

# Функция для загрузки Min-Max значений
def load_min_max_values():
    min_max_df = pd.read_csv(data_minmax_values)
    min_max_df.columns = min_max_df.columns.str.strip().str.lower().str.replace(' ', '_')
    min_values = min_max_df.set_index('parameter')['min'].to_dict()
    max_values = min_max_df.set_index('parameter')['max'].to_dict()
    return min_values, max_values

# Нормализация данных по Min-Max
def min_max_normalization(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Чтение весов модели из файла theta.txt
def load_theta():
    return np.loadtxt(data_theta_path)

# Функция для предсказания цены
def predict_price(X, theta):
    return np.dot(X, theta)

# Функция для расчета расстояния по формуле Хаверсина
def calculate_distance(lat1, lon1, lat2, lon2):
    # Радиус Земли в километрах
    R = 6371.0

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Главная страница с формой
# Главная страница с формой
@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = {}  # Пустой словарь по умолчанию для GET запроса

    if request.method == 'POST':
        # Получение значений из формы
        user_input = {
            "minutes_to_metro": float(request.form['minutes_to_metro']),
            "number_of_rooms": float(request.form['number_of_rooms']),
            "area": float(request.form['area']),
            "living_area": float(request.form['living_area']),
            "kitchen_area": float(request.form['kitchen_area']),
            "first_floor": float(request.form.get('first_floor', 0)),
            "type_secondary": float(request.form.get('type_secondary', 0)),
            "map_lat": float(request.form['map_lat']),
            "map_lon": float(request.form['map_lon'])
        }

        # Рассчитываем расстояние от центра города
        user_input["distance_to_centre"] = calculate_distance(CITY_LAT, CITY_LON, user_input["map_lat"],
                                                              user_input["map_lon"])

        # Загрузка Min-Max значений
        min_values, max_values = load_min_max_values()

        # Соответствие признаков с весами
        theta = load_theta()

        # Вывод значений тета для каждого параметра
        parameter_names = [
            "Intercept (смещение)", "minutes_to_metro", "number_of_rooms", "area", "living_area",
            "kitchen_area", "first_floor", "distance_to_centre", "type_secondary"
        ]

        # Нормализованные значения из формы
        normalized_input = {
            "minutes_to_metro": min_max_normalization(user_input["minutes_to_metro"], min_values["minutes_to_metro"],
                                                      max_values["minutes_to_metro"]),
            "number_of_rooms": min_max_normalization(user_input["number_of_rooms"], min_values["number_of_rooms"],
                                                     max_values["number_of_rooms"]),
            "area": min_max_normalization(user_input["area"], min_values["area"], max_values["area"]),
            "living_area": min_max_normalization(user_input["living_area"], min_values["living_area"],
                                                 max_values["living_area"]),
            "kitchen_area": min_max_normalization(user_input["kitchen_area"], min_values["kitchen_area"],
                                                  max_values["kitchen_area"]),
            "first_floor": min_max_normalization(user_input["first_floor"], min_values["first_floor"],
                                                 max_values["first_floor"]),
            "distance_to_centre": min_max_normalization(user_input["distance_to_centre"],
                                                        min_values["distance_to_centre"],
                                                        max_values["distance_to_centre"]),
            "type_secondary": min_max_normalization(user_input["type_secondary"], min_values["type_secondary"],
                                                    max_values["type_secondary"]),
        }

        for param, normalized_value in normalized_input.items():
            print(f"Параметр: {param}, Нормализованное значение: {normalized_value}")

        # Подготовка входного вектора (с учетом смещения)
        input_vector = np.array([1] + list(normalized_input.values()))

        # Умножаем параметры на соответствующие веса
        predicted_price = sum(input_vector[i] * theta[i] for i in range(len(input_vector)))


        # Правильный способ вывести значения параметров и соответствующие веса
        for i in range(len(input_vector)):
            print(f"Значение: {input_vector[i]}, Тета: {theta[i]}")

        return render_template('index.html', predicted_price=predicted_price, user_input=user_input)

    return render_template('index.html', predicted_price=None, user_input=user_input)


if __name__ == '__main__':
    app.run(debug=True)
