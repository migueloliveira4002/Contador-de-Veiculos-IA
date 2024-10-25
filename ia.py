import cv2
import torch
import numpy as np
import time

# Caminho do vídeo
video_path = r"C:\Users\M I G U E L\Desktop\IA Carros\video\video.mp4"

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Definir parâmetros de processamento
skip_frames = 5  # Processar 1 frame a cada 5 para acelerar
interval_car = 1.5  # Intervalo mínimo de 2 segundos para não contar o mesmo carro
confidence_threshold = 0.6  # Ajuste para confiança equilibrada
min_size = 7000  # Aumentar o tamanho mínimo do objeto
position_threshold = 500  # Aumentar a diferença mínima na posição
idle_frame_limit = 96  # Aumentar o número de frames que um carro pode ficar parado

# Abrir vídeo
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)  # Obter FPS do vídeo
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Número total de frames
duration = total_frames / fps  # Duração total do vídeo
frame_interval = int(fps * interval_car)  # Converter intervalo de tempo em frames

# Variáveis de contagem
car_count = 0
last_frame_counted = -frame_interval  # Inicializar para não contar duplicado
total_errors = 0
last_car_positions = {}  # Armazena as posições dos carros com seus tempos de detecção

# Processamento de vídeo
start_time = time.time()  # Para calcular o tempo total de processamento
frame_num = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Pular frames para acelerar o processamento
    if frame_num % skip_frames != 0:
        frame_num += 1
        continue

    # Realizar a detecção
    results = model(frame)

    # Filtrar objetos detectados como "car" e que tenham confiança e tamanho suficientes
    for detection in results.pred[0]:
        confidence = detection[4].item()  # Confiança da detecção
        label = results.names[int(detection[5])]
        x1, y1, x2, y2 = map(int, detection[:4])  # Coordenadas do objeto
        width, height = x2 - x1, y2 - y1
        area = width * height  # Área do objeto detectado
        
        if label == 'car' and confidence >= confidence_threshold and area >= min_size:
            car_position = (x1 + x2) // 2  # Coordenada horizontal do centro do carro detectado
            
            # Verificar se o carro já foi detectado anteriormente
            if car_position in last_car_positions:
                last_detected_frame, idle_frames = last_car_positions[car_position]
                
                # Verificar se o carro está parado há muitos frames
                if frame_num - last_detected_frame > idle_frame_limit:
                    del last_car_positions[car_position]  # Remover carros parados por muito tempo
                else:
                    # Verificar se a posição mudou significativamente (se não, aumentar o tempo de inatividade)
                    if abs(car_position - last_car_positions[car_position][0]) < position_threshold:
                        last_car_positions[car_position] = (frame_num, idle_frames + 1)
                    else:
                        # O carro se moveu, resetar o contador de frames parados
                        last_car_positions[car_position] = (frame_num, 0)
            else:
                # Se o carro não foi detectado ainda ou está muito longe do último carro detectado
                if abs(car_position - last_frame_counted) > position_threshold:
                    car_count += 1
                    last_frame_counted = car_position
                    last_car_positions[car_position] = (frame_num, 0)  # Registrar o carro e começar a contagem de frames

    # Exibir o vídeo (opcional, pode ser removido para acelerar)
    cv2.imshow("Car Detection", np.squeeze(results.render()))
    
    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

# Limpar
cap.release()
cv2.destroyAllWindows()

# Calcular tempo de processamento
end_time = time.time()
processing_time = end_time - start_time

# Calcular erro percentual
total_cars_detected = car_count + total_errors
error_percentage = (total_errors / total_cars_detected) * 100 if total_cars_detected > 0 else 0

# Exibir resultados
print(f"Total de carros contados: {car_count}")
print(f"Erros de contagem: {total_errors}")
print(f"Porcentagem de erro: {error_percentage:.2f}%")
print(f"Tempo total de processamento: {processing_time:.2f} segundos")
