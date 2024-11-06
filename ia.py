import cv2
import torch
import numpy as np
import time
from pathlib import Path
from collections import deque
import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

os.environ['GEMINI_API_KEY'] = 'INSERT_HERE_API_KEY'
genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# Caminho do vídeo
video_path = Path.cwd() / "video" / "video1.mp4"

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Total de carros no vídeo: 6

# Parâmetros ajustados
skip_frames = 3
confidence_threshold = 0.6
min_size = 8000
tracking_distance = 100
memory_frames = 30
window_scale = 0.6  # Escala da janela (0.6 = 60% do tamanho original)

# Zona de detecção com x e y
detection_zone = {
    'x_min': 25,  # Ajuste conforme a posição da sua câmera
    'x_max': 710,  # Ajuste conforme a posição da sua câmera
    'y_min': 200,  # Ajuste conforme a posição da sua câmera
    'y_max': 900  # Ajuste conforme a posição da sua câmera
}


class CarTracker:
    def __init__(self, car_id, position, frame_num):
        self.car_id = car_id
        self.positions = deque(maxlen=memory_frames)
        self.positions.append(position)
        self.last_seen = frame_num
        self.counted = False

    def update(self, position, frame_num):
        self.positions.append(position)
        self.last_seen = frame_num

    def get_average_position(self):
        return np.mean(self.positions)


def is_in_detection_zone(x, y):
    return (detection_zone['x_min'] <= x <= detection_zone['x_max'] and
            detection_zone['y_min'] <= y <= detection_zone['y_max'])


def classify_car(car_image_path):
    file = genai.upload_file(car_image_path, mime_type="image/png")

    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            properties={
                "model": content.Schema(
                    type=content.Type.STRING,
                ),
                "brand": content.Schema(
                    type=content.Type.STRING,
                ),
            },
        ),
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    file,
                ],
            },
        ]
    )

    response = chat_session.send_message("Classify this car")
    return response.text


def resize_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height))


def save_car_image(car_image, car_id):
    car_image_path = f"./cars/car_{car_id}.png"
    cv2.imwrite(car_image_path, car_image)
    return car_image_path


# Abrir vídeo
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

car_count = 0
active_cars = []
next_car_id = 0
frame_num = 0

start_time = time.time()

# Configurar a janela
cv2.namedWindow("Car Detection", cv2.WINDOW_NORMAL)
# Adicionar lista para armazenar classificações dos carros
car_classifications = []

# Modificar o loop principal para classificar apenas carros não contados
while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_num % skip_frames != 0:
        frame_num += 1
        continue

    # Processar frame em tamanho original
    original_frame = frame.copy()
    results = model(original_frame)

    # Limpar carros antigos
    active_cars = [car for car in active_cars
                   if (frame_num - car.last_seen) <= memory_frames]

    # Processar detecções
    for detection in results.pred[0]:
        confidence = detection[4].item()
        label = results.names[int(detection[5])]
        x1, y1, x2, y2 = map(int, detection[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width, height = x2 - x1, y2 - y1
        area = width * height

        if (label == 'car' and
                confidence >= confidence_threshold and
                area >= min_size and
                is_in_detection_zone(center_x, center_y)):

            car_matched = False
            for car in active_cars:
                if abs(center_x - car.get_average_position()) < tracking_distance:
                    car.update((center_x, center_y), frame_num)
                    car_matched = True

                    if not car.counted:
                        car_count += 1
                        car.counted = True
                        print(f"Novo carro detectado! Total: {car_count}")

                        # Classificar o carro detectado
                        car_image = original_frame[y1:y2, x1:x2]
                        car_image_path = save_car_image(car_image, car.car_id)
                        car_class = classify_car(car_image_path)
                        # remove image
                        os.remove(car_image_path)
                        car_classifications.append((car.car_id, car_class))
                        print(f"Carro classificado como: {car_class}")
                    break

            if not car_matched:
                new_car = CarTracker(next_car_id, (center_x, center_y), frame_num)
                active_cars.append(new_car)
                next_car_id += 1

        # Desenhar retângulos e informações
        if label == 'car' and confidence >= confidence_threshold:
            color = (0, 255, 0) if is_in_detection_zone(center_x, center_y) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Car {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Desenhar zona de detecção
    cv2.rectangle(frame,
                  (detection_zone['x_min'], detection_zone['y_min']),
                  (detection_zone['x_max'], detection_zone['y_max']),
                  (255, 0, 0), 2)

    # Mostrar contagem atual
    cv2.putText(frame, f"Carros: {car_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Redimensionar frame para exibição
    display_frame = resize_frame(frame, window_scale)
    cv2.imshow("Car Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()

end_time = time.time()
processing_time = end_time - start_time

print(f"\nResultados finais:")
print(f"Total de carros contados: {car_count}")
print(f"Tempo total de processamento: {processing_time:.2f} segundos")

# Mostrar classificações dos carros
print(f"Classificações dos carros:")
for car_id, car_class in car_classifications:
    print(f"Carro ID {car_id}: {car_class}")
