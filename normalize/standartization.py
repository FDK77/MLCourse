import pandas as pd
import math
import os

# Функция для вычисления расстояния по формуле Haversine
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371.0  # Радиус Земли в километрах
    return R * c  # расстояние в километрах

# Пути к файлам
project_root = os.path.abspath(os.path.dirname(__file__))
coor_file = os.path.join(project_root, 'data', 'stations_data.csv')
data_file = os.path.join(project_root, 'data', 'data.csv')
output_file = os.path.join(project_root, 'data', 'data_cleaned.csv')

# Чтение данных
coor_data = pd.read_csv(coor_file, sep=';')
data = pd.read_csv(data_file)

# Центр Москвы
moscow_center_lat = 55.7558
moscow_center_lon = 37.6173

# Создание словаря с координатами станций метро
station_coords = dict(zip(coor_data['station'], zip(coor_data['latitude'], coor_data['longitude'])))

# Функция для вычисления расстояния до центра Москвы
def get_distance_to_center(station_name):
    if station_name in station_coords:
        lat, lon = station_coords[station_name]
        return haversine(lat, lon, moscow_center_lat, moscow_center_lon)
    else:
        print(f"Станция не найдена: {station_name}")  # Выводим, если станция не найдена
        return None  # если станции нет в списке, возвращаем None

# Применение функции для расчета расстояния до центра
data['Distance to centre'] = data['Metro station'].apply(get_distance_to_center)

# Удаление строк, где расстояние не удалось вычислить (None)
data = data.dropna(subset=['Distance to centre'])

# Округление значений в столбце 'Distance to centre' до 2 знаков после запятой
data['Distance to centre'] = data['Distance to centre'].round(2)

# Удаление старого столбца 'Metro station'
data = data.drop(columns=['Metro station'])

# Применение OneHotEncoding к столбцу "Apartment type"
data_encoded = pd.get_dummies(data, columns=["Apartment type"], prefix="Type", drop_first=True)

# Преобразование всех булевых колонок (True/False) в 1/0
# Теперь мы используем map, применяя его только к столбцам с булевыми значениями
bool_columns = data_encoded.select_dtypes(include=['bool']).columns
for col in bool_columns:
    data_encoded[col] = data_encoded[col].map({True: 1, False: 0})

data_encoded["Floor"] = (data_encoded["Floor"] == 1).astype(int)

data_encoded = data_encoded.drop(columns=['Number of floors'])

# Переименование столбцов
data_encoded.rename(columns={
    'Price': 'price',
    'Type_Secondary': 'type_secondary',
    'Distance to centre': 'distance_to_centre',
    'Minutes to metro': 'minutes_to_metro',
    'Number of rooms': 'number_of_rooms',
    'Area': 'area',
    'Living area': 'living_area',
    'Kitchen area': 'kitchen_area',
    'Floor': 'first_floor',
}, inplace=True)

# Переместим параметр 'price' в последний столбец
data_encoded = data_encoded[[col for col in data_encoded.columns if col != 'price'] + ['price']]

# Сохранение результата
data_encoded.to_csv(output_file, index=False)

print("Данные успешно обновлены и сохранены в 'data_cleaned.csv'.")
