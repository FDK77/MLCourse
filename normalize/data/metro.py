import requests
import csv

# URL Overpass API
overpass_url = "http://overpass-api.de/api/interpreter"

# Запрос для получения станций метро и железнодорожных станций в Москве
overpass_query = """
[out:json];
area["name"="Москва"]->.searchArea;
(
  node["station"="subway"](area.searchArea);  
  node["railway"="station"](area.searchArea); 
);
out body;
"""

# Отправка запроса
response = requests.get(overpass_url, params={'data': overpass_query})

# Преобразуем ответ в формат JSON
data = response.json()

# Открываем CSV файл для записи
with open('stations_data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=';')  # Используем точку с запятой как разделитель
    # Записываем заголовок
    writer.writerow(['station', 'latitude', 'longitude'])

    # Извлекаем данные о станциях и записываем в CSV
    for element in data['elements']:
        name = element.get('tags', {}).get('name', 'Неизвестно')  # Название станции
        lat = element['lat']  # Широта
        lon = element['lon']  # Долгота
        # Записываем строку в CSV
        writer.writerow([name, lat, lon])

print("Данные успешно сохранены в 'stations_data.csv'.")
