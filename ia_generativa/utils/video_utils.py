"""
Video Utils - Funções utilitárias para manipulação de vídeo.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import tempfile
import time
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import io

logger = logging.getLogger(__name__)

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Obtém informações básicas de um vídeo.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        
    Returns:
        Dicionário com informações do vídeo
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")
            
        # Abrir o vídeo
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
            
        # Obter propriedades
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Converter codec para string
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        
        # Calcular duração
        duration = frame_count / fps if fps > 0 else 0
        
        # Liberar recursos
        cap.release()
        
        # Obter tamanho do arquivo
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        
        return {
            "path": video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "codec": codec_str,
            "file_size_bytes": file_size,
            "file_size_mb": file_size_mb
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter informações do vídeo: {str(e)}")
        raise

def create_thumbnail(
    video_path: str,
    output_path: Optional[str] = None,
    time_position: float = 0.0,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> str:
    """
    Cria uma miniatura de um vídeo.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        output_path: Caminho para salvar a miniatura
        time_position: Posição em segundos para capturar o frame
        width: Largura da miniatura (opcional)
        height: Altura da miniatura (opcional)
        
    Returns:
        Caminho para a miniatura criada
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")
            
        # Abrir o vídeo
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
            
        # Obter fps para converter tempo para número de frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcular número do frame
        frame_number = min(int(time_position * fps), frame_count - 1)
        
        # Posicionar no frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Ler o frame
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError(f"Não foi possível ler o frame na posição {time_position}s")
            
        # Redimensionar se necessário
        if width is not None or height is not None:
            # Obter dimensões originais
            h, w = frame.shape[:2]
            
            # Calcular nova largura mantendo proporção
            if width is not None and height is None:
                height = int(h * (width / w))
            # Calcular nova altura mantendo proporção
            elif height is not None and width is None:
                width = int(w * (height / h))
                
            # Redimensionar
            frame = cv2.resize(frame, (width, height))
            
        # Definir caminho de saída
        if output_path is None:
            output_dir = tempfile.gettempdir()
            output_path = os.path.join(output_dir, f"thumbnail_{int(time.time())}.jpg")
            
        # Salvar frame como imagem
        cv2.imwrite(output_path, frame)
        
        # Liberar recursos
        cap.release()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Erro ao criar miniatura: {str(e)}")
        raise

def create_waveform_image(
    video_path: str,
    output_path: Optional[str] = None,
    width: int = 800,
    height: int = 200,
    color: str = 'blue'
) -> str:
    """
    Cria uma imagem da forma de onda do áudio de um vídeo.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        output_path: Caminho para salvar a imagem
        width: Largura da imagem
        height: Altura da imagem
        color: Cor da forma de onda
        
    Returns:
        Caminho para a imagem criada
    """
    try:
        # Definir caminho de saída
        if output_path is None:
            output_dir = tempfile.gettempdir()
            output_path = os.path.join(output_dir, f"waveform_{int(time.time())}.png")
            
        # Extrair áudio do vídeo
        temp_audio = os.path.join(tempfile.gettempdir(), f"temp_audio_{int(time.time())}.wav")
        
        # Usar ffmpeg para extrair o áudio
        command = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            temp_audio,
            "-y"
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Carregar áudio
        from scipy.io import wavfile
        sample_rate, audio_data = wavfile.read(temp_audio)
        
        # Se estéreo, converter para mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Reduzir resolução para plotagem mais eficiente
        if len(audio_data) > width * 10:
            reduction_factor = len(audio_data) // (width * 10)
            audio_data = audio_data[::reduction_factor]
            
        # Criar figura
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.plot(audio_data, color=color)
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Salvar imagem
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Limpar arquivo temporário
        os.remove(temp_audio)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Erro ao criar imagem de forma de onda: {str(e)}")
        raise

def create_animated_gif(
    video_path: str,
    output_path: Optional[str] = None,
    start_time: float = 0.0,
    duration: float = 3.0,
    fps: int = 10,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> str:
    """
    Cria um GIF animado a partir de um vídeo.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        output_path: Caminho para salvar o GIF
        start_time: Tempo inicial em segundos
        duration: Duração do GIF em segundos
        fps: Frames por segundo do GIF
        width: Largura do GIF (opcional)
        height: Altura do GIF (opcional)
        
    Returns:
        Caminho para o GIF criado
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")
            
        # Definir caminho de saída
        if output_path is None:
            output_dir = tempfile.gettempdir()
            output_path = os.path.join(output_dir, f"animated_gif_{int(time.time())}.gif")
            
        # Extrair frames do vídeo
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
            
        # Obter fps original
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calcular frame inicial
        start_frame = int(start_time * original_fps)
        
        # Calcular número de frames a extrair
        num_frames = int(duration * fps)
        
        # Calcular intervalo entre frames
        frame_interval = int(original_fps / fps)
        
        # Posicionar no frame inicial
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extrair frames
        frames = []
        for i in range(num_frames):
            # Ler o frame
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Converter BGR para RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Redimensionar se necessário
            if width is not None or height is not None:
                # Obter dimensões originais
                h, w = frame_rgb.shape[:2]
                
                # Calcular nova largura mantendo proporção
                if width is not None and height is None:
                    height = int(h * (width / w))
                # Calcular nova altura mantendo proporção
                elif height is not None and width is None:
                    width = int(w * (height / h))
                    
                # Redimensionar
                frame_rgb = cv2.resize(frame_rgb, (width, height))
                
            # Converter para PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Adicionar à lista
            frames.append(pil_image)
            
            # Avançar para próximo frame de interesse
            for _ in range(frame_interval - 1):
                cap.read()  # Pular frames
                
        # Liberar recursos
        cap.release()
        
        # Verificar se há frames
        if not frames:
            raise ValueError("Não foi possível extrair frames do vídeo")
            
        # Criar GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=int(1000 / fps),  # Duração de cada frame em ms
            loop=0  # Loop infinito
        )
        
        return output_path
        
    except Exception as e:
        logger.error(f"Erro ao criar GIF animado: {str(e)}")
        raise

def compress_video(
    video_path: str,
    output_path: Optional[str] = None,
    target_size_mb: Optional[float] = None,
    crf: int = 23,  # Fator de qualidade (menor = melhor)
    preset: str = "medium"  # Preset de codificação
) -> str:
    """
    Comprime um vídeo para reduzir seu tamanho.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        output_path: Caminho para salvar o vídeo comprimido
        target_size_mb: Tamanho alvo em MB (opcional)
        crf: Fator de qualidade (18-28, menor = melhor qualidade)
        preset: Preset de codificação (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        
    Returns:
        Caminho para o vídeo comprimido
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")
            
        # Definir caminho de saída
        if output_path is None:
            output_dir = tempfile.gettempdir()
            output_path = os.path.join(output_dir, f"compressed_video_{int(time.time())}.mp4")
            
        # Obter tamanho original
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # Obter informações do vídeo
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Configurar compressão
        if target_size_mb is not None and target_size_mb < file_size_mb:
            # Calcular bitrate alvo
            duration = get_video_info(video_path)["duration"]
            target_bitrate = int((target_size_mb * 8 * 1024) / duration)
            
            # Comando ffmpeg com bitrate alvo
            command = [
                "ffmpeg",
                "-i", video_path,
                "-c:v", "libx264",
                "-b:v", f"{target_bitrate}k",
                "-maxrate", f"{int(target_bitrate * 1.5)}k",
                "-bufsize", f"{int(target_bitrate * 3)}k",
                "-preset", preset,
                "-c:a", "aac",
                "-b:a", "128k",
                output_path,
                "-y"
            ]
        else:
            # Comando ffmpeg com CRF (qualidade constante)
            command = [
                "ffmpeg",
                "-i", video_path,
                "-c:v", "libx264",
                "-crf", str(crf),
                "-preset", preset,
                "-c:a", "aac",
                "-b:a", "128k",
                output_path,
                "-y"
            ]
            
        # Executar comando
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Verificar se o arquivo foi criado
        if not os.path.isfile(output_path):
            raise ValueError("Falha ao comprimir vídeo")
            
        # Obter tamanho comprimido
        compressed_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        logger.info(f"Vídeo comprimido de {file_size_mb:.2f}MB para {compressed_size_mb:.2f}MB")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Erro ao comprimir vídeo: {str(e)}")
        raise
