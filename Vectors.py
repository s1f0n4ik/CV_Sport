import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

#функция расчета угла
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cosine_angle))

#определение фаз бега ---
def detect_phases(df_y):
    y_vals = df_y.values.flatten()
    velocity = np.diff(y_vals, prepend=y_vals[0])
    phases = []
    #нога на земле, если она в нижних 20% по высоте
    ground_threshold = np.percentile(y_vals, 80) 
    
    for y, vel in zip(y_vals, velocity):
        if y > ground_threshold and abs(vel) < 0.05: 
            phases.append(1) #опора
        else:
            phases.append(0) #полет
    return phases

#основная обработка
def process_runner_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    #превращаем длинную таблицу в широкую (одна строка = один кадр)
    df_wide = df.pivot(index='frame_number', columns='landmark_index', values=['x', 'y', 'z'])
    df_wide.columns = [f'{col[0]}_{col[1]}' for col in df_wide.columns]
    
    results = []
    
    for frame_idx, row in df_wide.iterrows():
        #функция для быстрого получения координат (x, y) точки по индексу
        def get_p(idx):
            return [row[f'x_{idx}'], row[f'y_{idx}']]
            
        #индексы MediaPipe: Нечетные (11, 23) = левая, Четные (12, 24) = правая
        
        #УГЛЫ НОГ (Бедро - Колено - Лодыжка)
        angle_l_knee = calculate_angle(get_p(23), get_p(25), get_p(27)) # Левое
        angle_r_knee = calculate_angle(get_p(24), get_p(26), get_p(28)) # Правое
        
        #УГЛЫ СТОП (Колено - Лодыжка - Носок) 
        angle_l_ankle = calculate_angle(get_p(25), get_p(27), get_p(31)) # Левая
        angle_r_ankle = calculate_angle(get_p(26), get_p(28), get_p(32)) # Правая
        
        #УГЛЫ БЕДЕР (Плечо - Таз - Колено) 
        angle_l_hip = calculate_angle(get_p(11), get_p(23), get_p(25)) # Левое
        angle_r_hip = calculate_angle(get_p(12), get_p(24), get_p(26)) # Правое
        
        #УГЛЫ РУК (Плечо - Локоть - Запястье) 
        angle_l_elbow = calculate_angle(get_p(11), get_p(13), get_p(15)) # Левая
        angle_r_elbow = calculate_angle(get_p(12), get_p(14), get_p(16)) # Правая

        #НАКЛОН ГОЛЕНИ (Вертикаль - Колено - Лодыжка) 
        # Для оценки "Overstriding" 
        l_knee, r_knee = get_p(25), get_p(26)
        l_vert, r_vert = [l_knee[0], l_knee[1] + 0.5], [r_knee[0], r_knee[1] + 0.5]
        
        angle_l_shank = calculate_angle(l_vert, get_p(25), get_p(27))
        angle_r_shank = calculate_angle(r_vert, get_p(26), get_p(28))

        #НАКЛОН КОРПУСА 
        #Вертикаль - Таз (23) - Плечо (11)
        hip, shoulder = get_p(23), get_p(11)
        vertical_point = [hip[0], hip[1] - 0.5]
        trunk_lean = calculate_angle(vertical_point, hip, shoulder)
        
        results.append({
            'frame_number': frame_idx,
            'left_knee': angle_l_knee, 'right_knee': angle_r_knee,
            'left_ankle': angle_l_ankle, 'right_ankle': angle_r_ankle,
            'left_hip': angle_l_hip, 'right_hip': angle_r_hip,
            'left_elbow': angle_l_elbow, 'right_elbow': angle_r_elbow,
            'left_shank': angle_l_shank, 'right_shank': angle_r_shank,
            'trunk_lean': trunk_lean,
            #координаты Y для расчета фаз (27=Лев, 28=Прав)
            'left_ankle_y': row['y_27'], 'right_ankle_y': row['y_28']
        })
        
    features_df = pd.DataFrame(results)
    
    #расчет фаз бега (1=Опора, 0=Полет)
    features_df['phase_left'] = detect_phases(features_df[['left_ankle_y']])
    features_df['phase_right'] = detect_phases(features_df[['right_ankle_y']])
    
    features_df.to_csv(output_csv, index=False)
    return features_df

#ЗАПУСК ПО ВСЕМ ФАЙЛАМ
project_folder = r'C:\Users\Dell\Documents\Nik i Danya proect\Проект'
output_folder = os.path.join(project_folder, 'Analytic_Data')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Начинаем Этап 2: Расчет биомеханики")

for i in tqdm(range(1, 22), desc="Анализ файлов"):
    input_file = os.path.join(project_folder, f'Коорд.Бег {i}.csv')
    output_file = os.path.join(output_folder, f'Признаки.Бег {i}.csv')
    
    if os.path.exists(input_file):
        try:
            df_result = process_runner_data(input_file, output_file)
            
            #строим графики 
            
            plt.figure(figsize=(10, 6))
            plt.plot(df_result['frame_number'], df_result['left_knee'], label='Левое колено')
            plt.plot(df_result['frame_number'], df_result['right_knee'], label='Правое колено')
            plt.title(f'График коленей (Бег {i})')
            plt.legend()
            plt.savefig(os.path.join(output_folder, f'График_Колени_{i}.png'))
            plt.close()
                
        except Exception as e:
            print(f"Ошибка в файле {i}: {e}")
    else:
        # print(f"Файл не найден: {input_file}") 
        pass

print(f"Результаты в папке: {output_folder}")