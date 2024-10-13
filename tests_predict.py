
import os
import json
from ultralytics import YOLO
from PIL import Image

# Функция для выполнения предсказаний и сохранения результатов в JSON
def test_yolo_model(model_path, test_images_folder, output_json_path, img_size=640, conf_thresh=0.25):
    # Загрузка обученной модели YOLO
    model = YOLO(model_path)

    # Список для хранения результатов
    results_list = []

    # Перебираем все изображения в папке
    for image_file in os.listdir(test_images_folder):
        image_path = os.path.join(test_images_folder, image_file)

        # Загружаем изображение
        image = Image.open(image_path)

        # Выполняем предсказание
        results = model.predict(source=image_path, imgsz=img_size, conf=conf_thresh)

        # Формируем результат для каждого изображения
        for result in results:
            predictions = []
            for box in result.boxes:
                # Получаем координаты, метку класса и уверенность
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # координаты бокса
                conf = float(box.conf[0].item())  # уверенность
                class_id = int(box.cls[0].item())  # метка класса

                predictions.append({
                    "class_id": class_id,
                    "confidence": 0.5,
                    "bbox": [x1, y1, x2, y2]  # формат [x1, y1, x2, y2]
                })

            # Добавляем предсказания для текущего изображения в список результатов
            results_list.append({
                "image": image_file,
                "predictions": predictions
            })

    # Сохраняем результаты в JSON файл
    with open(output_json_path, 'w') as f:
        json.dump(results_list, f, indent=4)

    print(f"Результаты сохранены в {output_json_path}")

# Параметры для скрипта efsdkjasnandom320930padma
model_path = "C:\\Users\\Я\\Documents\\GitHub\inairdtct\\runs\detect\\train28\\weights\\best.pt"  # Путь к обученной модели YOLO
test_images_folder = 'C:\\Users\\work\\tests\\data'  # Путь к папке с изображениями для теста
output_json_path = 'C:\\Users\\Я\\Documents\\GitHub\\inairdtct\\runs\\detect\\train28\\test_resyolo_predictions.json'  # Путь для сохранения выходного JSON файла

# Запуск функции тестирования
test_yolo_model(model_path, test_images_folder, output_json_path)
