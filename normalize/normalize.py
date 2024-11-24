import pandas as pd
import os

# Функция для Мин-Макс нормализации
def min_max_normalization(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    # Словарь для хранения минимальных и максимальных значений
    min_max_values = {}

    # Применение нормализации ко всем столбцам, кроме исключенных
    for column in df.columns:
        if column not in exclude_columns:
            min_val = df[column].min()
            max_val = df[column].max()

            # Сохраняем минимальное и максимальное значения
            min_max_values[column] = {'min': min_val, 'max': max_val}

            # Нормализация столбца
            df[column] = (df[column] - min_val) / (max_val - min_val)

    return df, min_max_values

# Пути к файлам
project_root = os.path.abspath(os.path.dirname(__file__))
data_file = os.path.join(project_root, 'data', 'data_cleaned.csv')
output_file = os.path.join(project_root, 'data', 'data_minmax.csv')
min_max_file = os.path.join(project_root, 'data', 'min_max_values.csv')

# Чтение данных
data = pd.read_csv(data_file)

# Столбец, который не должен быть нормализован
exclude_columns = ['price']

# Нормализация данных
normalized_data, min_max_values = min_max_normalization(data.copy(), exclude_columns=exclude_columns)

# Сохранение нормализованных данных в новый файл
normalized_data.to_csv(output_file, index=False)

# Сохранение минимальных и максимальных значений для каждого признака
min_max_df = pd.DataFrame(min_max_values).T

# Добавляем индекс как отдельный столбец, чтобы он записался в файл
min_max_df.reset_index(inplace=True)
min_max_df.rename(columns={'index': 'parameter'}, inplace=True)

# Сохраняем файл с корректным заголовком
min_max_df.to_csv(min_max_file, index=False)

print("Нормализация завершена и сохранена в файлы:")
print(f"Нормализованные данные: {output_file}")
print(f"Минимальные и максимальные значения для нормализации: {min_max_file}")
