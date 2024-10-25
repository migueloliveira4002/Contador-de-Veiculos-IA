import cv2
import torch
import time
import tkinter as tk
from tkinter import messagebox

# Ajustar brilho e contraste
def ajustar_brilho_contraste(frame, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# Carregar o vídeo
def carregar_video(caminho_video):
    print(f"Tentando abrir o vídeo em: {caminho_video}")
    video = cv2.VideoCapture(caminho_video)
    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        return None
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frames por segundo: {fps}, Total de frames: {total_frames}")
    return video

# Carregar o modelo YOLOv5 pré-treinado
def carregar_modelo_yolo():
    print("Carregando modelo YOLOv5...")
    modelo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("Modelo YOLOv5 carregado.")
    return modelo

# Função para detectar carros no frame
def detectar_carros(frame, modelo):
    resultados = modelo(frame)
    return resultados

# Função para contar carros detectados dentro da ROI
def contar_carros(resultados, linha_passagem, objetos_contados):
    carros_detectados = 0
    
    for obj in resultados.xyxy[0]:
        id_objeto = int(obj[5])  # Classe
        x1, y1, x2, y2 = obj[:4]  # Coordenadas da caixa delimitadora

        if id_objeto == 2:  # Classe 2 para carros
            if y1 < linha_passagem and y2 > linha_passagem:  # Se o carro cruzou a linha
                if (x1, y1, x2, y2) not in objetos_contados:
                    carros_detectados += 1
                    objetos_contados.append((x1, y1, x2, y2))

    return carros_detectados, objetos_contados

# Função para mostrar informações em um pop-up
def mostrar_info(total_carros, margem_erro):
    info = f"Total de Carros Detectados: {total_carros}\nMargem de Erro: {margem_erro:.2f}%"
    root = tk.Tk()
    root.withdraw()  # Esconder a janela principal
    messagebox.showinfo("Resultados da Detecção", info)
    root.destroy()

# Processar o vídeo
def processar_video(video, modelo, intervalo_contagem=3):
    frame_count = 0
    total_carros = 0
    last_count_time = time.time()
    
    linha_passagem = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    objetos_contados = []  # Lista para armazenar objetos contados

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Erro ao ler o frame ou fim do vídeo.")
            break
        
        frame_count += 1
        
        # Ajustar brilho, contraste e conversor para RGB
        frame_ajustado = ajustar_brilho_contraste(frame)
        frame_rgb = cv2.cvtColor(frame_ajustado, cv2.COLOR_BGR2RGB)
        
        resultados = detectar_carros(frame_rgb, modelo)
        carros_detectados, objetos_contados = contar_carros(resultados, linha_passagem, objetos_contados)

        # Verificar se é hora de contar um carro
        if carros_detectados > 0 and (time.time() - last_count_time >= intervalo_contagem):
            total_carros += carros_detectados
            last_count_time = time.time()  # Atualiza o tempo da última contagem
            
        print(f"Frame {frame_count}: Carros detectados nesta contagem: {carros_detectados}, Total: {total_carros}")

        # Desenhar a linha de passagem no frame
        cv2.line(frame, (0, linha_passagem), (frame.shape[1], linha_passagem), (0, 255, 0), 2)  # Linha de passagem
        
        # Exibir o frame com detecções
        resultados.render()
        cv2.imshow('Detecção de Carros', resultados.ims[0])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    
    # Calcular margem de erro com base em uma porcentagem do total de carros detectados
    margem_erro = total_carros * 0.0043  # Exemplo: 2% do total
    mostrar_info(total_carros, margem_erro)
    
    return total_carros

# Função principal
def main():
    caminho_video = "C:\\Users\\M I G U E L\\Desktop\\IA Carros\\video\\video5.mp4"
    video = carregar_video(caminho_video)
    if video is None:
        return  # Termina se o vídeo não puder ser carregado
    
    modelo = carregar_modelo_yolo()
    total_carros = processar_video(video, modelo, intervalo_contagem=3)
    print(f"Total de carros detectados: {total_carros}")

if __name__ == '__main__':
    main()
