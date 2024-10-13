import cv2
import os

def extract_frames(video_path, output_folder):
    """
    Делит видео на отдельные кадры и сохраняет их в указанной папке.

    video_path: путь к видеофайлу
    output_folder: путь к папке, в которую будут сохранены кадры
    """

    # Открываем видеофайл
    video_capture = cv2.VideoCapture(video_path)

    # Проверяем, удалось ли открыть видео
    if not video_capture.isOpened():
        print(f"Не удалось открыть видеофайл: {video_path}")
        return

    # Проверяем, существует ли папка для сохранения кадров, если нет - создаем
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    while True:
        # Чтение кадра
        success, frame = video_capture.read()

        if not success:
            print("Конец видео или ошибка при чтении.")
            break

        # Генерируем имя файла для каждого кадра
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        
        # Сохраняем кадр как изображение
        cv2.imwrite(frame_filename, frame)
        print(f"Сохранен кадр {frame_count} в {frame_filename}")

        frame_count += 1

    # Освобождаем ресурсы
    video_capture.release()
    print(f"Обработка завершена. Всего сохранено {frame_count} кадров.")

# Пример использования
video_path = 'video.mp4'  # Замените на путь к вашему видео
output_folder = 'frames'  # Папка для сохранения кадров

extract_frames(video_path, output_folder)
