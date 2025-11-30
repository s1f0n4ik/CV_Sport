#библиотеки
import cv2
import mediapipe as mp
import csv
import os
#инициализация функций медиапайпа
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
#функция обработки видео
def process_video(input_video_path, output_csv_path, output_video_path):
        #задаем параметры обработки
       with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        #открытие видео
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видео {input_video_path}")
            return
        
        #параметры входного видео
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        #параметры выходного видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        #создаем файл excel
        with open(output_csv_path, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            csv_writer.writerow(['frame_number', 'landmark_index', 'x', 'y', 'z', 'visibility'])

            frame_idx = 0
            print(f"Начинаем обработку: {input_video_path}")
            #покадровая обработка
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                #преобразование из гбр в ргб
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                #преобразование из ргб в бгр
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                #
                if results.pose_landmarks:
                
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        csv_writer.writerow([
                            frame_idx, 
                            i, 
                            landmark.x, 
                            landmark.y, 
                            landmark.z, 
                            landmark.visibility
                        ])
                    #рисуем точки
                    mp_drawing.draw_landmarks(
                        image_bgr, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Цвет точек
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Цвет линий
                    )
                #записываем видео
                out.write(image_bgr)
                #показываем видео
                cv2.imshow('Pose Estimation Process', image_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1
       #закрываем видео
        cap.release()
        out.release() 
        cv2.destroyAllWindows()
        print(f"Готово!")
        print(f"CSV сохранен: {output_csv_path}")
        print(f"Видео сохранено: {output_video_path}")
#цикл из 21 видео
for i in range(1, 22):
    input_video = rf'C:\Users\Dell\Documents\Nik i Danya proect\Проект\Бег {i}.MOV'
    output_csv = rf'C:\Users\Dell\Documents\Nik i Danya proect\Проект\Коорд.Бег {i}.csv'
    output_video = rf'C:\Users\Dell\Documents\Nik i Danya proect\Проект\Обр.Бег {i}.mp4'

    #проверка и запуск
    if os.path.exists(input_video):
        process_video(input_video, output_csv, output_video)
    else:
        print(f"Файл видео не найден по пути: {input_video}")