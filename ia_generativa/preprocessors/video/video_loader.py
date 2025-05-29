"""
Video Loader - Funções para carregamento e extração de conteúdo de vídeos.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import tempfile
import time
import base64
import subprocess
from PIL import Image
import io
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_video(
    video_path: str, 
    max_frames: Optional[int] = None,
    frame_interval: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Carrega um vídeo e retorna seus frames e informações.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        max_frames: Número máximo de frames a carregar
        frame_interval: Intervalo entre frames (1 = todos os frames)
        start_time: Tempo inicial para carregar (segundos)
        end_time: Tempo final para carregar (segundos)
        resize: Tuple (width, height) para redimensionar frames
        
    Returns:
        Tupla (lista de frames, dicionário de metadados)
    """
    try:
        logger.info(f"Carregando vídeo: {video_path}")
        
        # Abrir o vídeo
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
            
        # Obter propriedades do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Propriedades do vídeo: {width}x{height}, {fps} FPS, {duration:.2f}s, {total_frames} frames")
        
        # Calcular frames inicial e final com base no tempo
        start_frame = 0
        end_frame = total_frames
        
        if start_time is not None:
            start_frame = int(start_time * fps)
            
        if end_time is not None:
            end_frame = min(total_frames, int(end_time * fps))
            
        # Limitar número máximo de frames
        frames_to_extract = (end_frame - start_frame) // frame_interval
        
        if max_frames is not None and max_frames < frames_to_extract:
            frames_to_extract = max_frames
            # Ajustar intervalo para distribuir frames uniformemente
            if frames_to_extract > 0:
                frame_interval = (end_frame - start_frame) // frames_to_extract
                
        logger.info(f"Extraindo {frames_to_extract} frames com intervalo {frame_interval}")
        
        # Posicionar no frame inicial
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extrair frames
        frames = []
        frame_count = 0
        current_frame = start_frame
        
        with tqdm(total=frames_to_extract, desc="Carregando frames") as pbar:
            while current_frame < end_frame and (max_frames is None or frame_count < max_frames):
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                # Verificar intervalo
                if (current_frame - start_frame) % frame_interval == 0:
                    # Redimensionar se necessário
                    if resize is not None:
                        frame = cv2.resize(frame, resize)
                        
                    # Converter BGR para RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    frames.append(frame_rgb)
                    frame_count += 1
                    pbar.update(1)
                    
                current_frame += 1
                
        # Liberar recursos
        cap.release()
        
        # Criar dicionário de metadados
        metadata = {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration": duration,
            "loaded_frames": len(frames),
            "frame_interval": frame_interval,
            "start_time": start_time if start_time is not None else 0,
            "end_time": end_time if end_time is not None else duration
        }
        
        logger.info(f"Vídeo carregado: {len(frames)} frames")
        
        return frames, metadata
        
    except Exception as e:
        logger.error(f"Erro ao carregar vídeo: {str(e)}")
        raise

def load_video_from_base64(
    base64_string: str,
    max_frames: Optional[int] = None,
    frame_interval: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Carrega um vídeo a partir de uma string base64.
    
    Args:
        base64_string: String base64 do vídeo
        max_frames: Número máximo de frames a carregar
        frame_interval: Intervalo entre frames (1 = todos os frames)
        start_time: Tempo inicial para carregar (segundos)
        end_time: Tempo final para carregar (segundos)
        resize: Tuple (width, height) para redimensionar frames
        
    Returns:
        Tupla (lista de frames, dicionário de metadados)
    """
    try:
        # Remover prefixo se presente
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        # Decodificar string base64
        video_bytes = base64.b64decode(base64_string)
        
        # Salvar em arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(video_bytes)
            
        # Carregar do arquivo temporário
        frames, metadata = load_video(
            tmp_path,
            max_frames=max_frames,
            frame_interval=frame_interval,
            start_time=start_time,
            end_time=end_time,
            resize=resize
        )
        
        # Remover arquivo temporário
        os.unlink(tmp_path)
        
        return frames, metadata
        
    except Exception as e:
        logger.error(f"Erro ao carregar vídeo de base64: {str(e)}")
        raise

def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    format: str = "wav",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> str:
    """
    Extrai a trilha de áudio de um vídeo.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        output_path: Caminho para salvar o áudio (opcional)
        format: Formato do áudio de saída (wav, mp3)
        start_time: Tempo inicial para extração (segundos)
        end_time: Tempo final para extração (segundos)
        
    Returns:
        Caminho para o arquivo de áudio extraído
    """
    try:
        logger.info(f"Extraindo áudio do vídeo: {video_path}")
        
        # Definir caminho de saída
        if output_path is None:
            output_dir = tempfile.gettempdir()
            output_path = os.path.join(output_dir, f"audio_{int(time.time())}.{format}")
            
        # Construir comando ffmpeg
        command = ["ffmpeg", "-i", video_path]
        
        # Adicionar opções de tempo
        if start_time is not None:
            command.extend(["-ss", str(start_time)])
            
        if end_time is not None:
            command.extend(["-to", str(end_time)])
            
        # Adicionar opções de saída
        command.extend(["-q:a", "0", "-map", "a", output_path, "-y"])
        
        # Executar comando
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"Áudio extraído para: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Erro ao extrair áudio: {str(e)}")
        raise

def extract_frames_from_video(
    video_path: str,
    output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
    frame_interval: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    format: str = "jpg",
    resize: Optional[Tuple[int, int]] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extrai frames de um vídeo e os salva como imagens.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        output_dir: Diretório para salvar os frames
        max_frames: Número máximo de frames a extrair
        frame_interval: Intervalo entre frames (1 = todos os frames)
        start_time: Tempo inicial para extração (segundos)
        end_time: Tempo final para extração (segundos)
        format: Formato das imagens (jpg, png)
        resize: Tuple (width, height) para redimensionar frames
        
    Returns:
        Tupla (lista de caminhos para as imagens, dicionário de metadados)
    """
    try:
        logger.info(f"Extraindo frames do vídeo: {video_path}")
        
        # Criar diretório de saída
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), f"frames_{int(time.time())}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Carregar frames do vídeo
        frames, metadata = load_video(
            video_path,
            max_frames=max_frames,
            frame_interval=frame_interval,
            start_time=start_time,
            end_time=end_time,
            resize=resize
        )
        
        # Salvar frames como imagens
        frame_paths = []
        
        for i, frame in enumerate(tqdm(frames, desc="Salvando frames")):
            # Converter RGB para BGR para salvar com OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Gerar nome do arquivo
            frame_path = os.path.join(output_dir, f"frame_{i:06d}.{format}")
            
            # Salvar imagem
            cv2.imwrite(frame_path, frame_bgr)
            
            # Adicionar à lista de caminhos
            frame_paths.append(frame_path)
            
        logger.info(f"Frames extraídos para: {output_dir}")
        
        # Adicionar informações de frames à metadata
        metadata["frame_paths"] = frame_paths
        metadata["frame_count"] = len(frame_paths)
        
        return frame_paths, metadata
        
    except Exception as e:
        logger.error(f"Erro ao extrair frames: {str(e)}")
        raise

def create_video_from_frames(
    frame_paths: List[str],
    output_path: Optional[str] = None,
    fps: float = 30.0,
    audio_path: Optional[str] = None,
    codec: str = "mp4v"
) -> str:
    """
    Cria um vídeo a partir de uma sequência de imagens.
    
    Args:
        frame_paths: Lista de caminhos para as imagens
        output_path: Caminho para salvar o vídeo
        fps: Frames por segundo
        audio_path: Caminho para arquivo de áudio a adicionar
        codec: Codec de vídeo (mp4v, avc1, etc.)
        
    Returns:
        Caminho para o vídeo criado
    """
    try:
        logger.info(f"Criando vídeo a partir de {len(frame_paths)} imagens")
        
        # Verificar se há frames
        if not frame_paths:
            raise ValueError("Nenhuma imagem fornecida")
            
        # Definir caminho de saída
        if output_path is None:
            output_dir = tempfile.gettempdir()
            temp_output_path = os.path.join(output_dir, f"video_{int(time.time())}.mp4")
        else:
            temp_output_path = output_path
            
        # Ler primeira imagem para obter dimensões
        frame = cv2.imread(frame_paths[0])
        height, width, _ = frame.shape
        
        # Configurar writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        # Escrever frames
        for frame_path in tqdm(frame_paths, desc="Criando vídeo"):
            frame = cv2.imread(frame_path)
            writer.write(frame)
            
        # Liberar recursos
        writer.release()
        
        # Adicionar áudio se fornecido
        if audio_path:
            logger.info(f"Adicionando áudio: {audio_path}")
            
            # Definir caminho para vídeo final
            if output_path is None:
                output_dir = tempfile.gettempdir()
                final_output_path = os.path.join(output_dir, f"video_with_audio_{int(time.time())}.mp4")
            else:
                # Criar um caminho temporário
                final_output_path = output_path
                temp_output_path = output_path + ".temp.mp4"
                
            # Adicionar áudio com ffmpeg
            command = [
                "ffmpeg", 
                "-i", temp_output_path, 
                "-i", audio_path, 
                "-c:v", "copy", 
                "-c:a", "aac", 
                "-shortest",
                final_output_path,
                "-y"
            ]
            
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Limpar arquivo temporário
            if temp_output_path != final_output_path:
                os.remove(temp_output_path)
                
            output_path = final_output_path
        else:
            output_path = temp_output_path
            
        logger.info(f"Vídeo criado: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Erro ao criar vídeo: {str(e)}")
        raise

def video_to_base64(
    video_path: str,
    max_size_mb: Optional[float] = None,
    resize_if_needed: bool = True
) -> str:
    """
    Converte um vídeo para string base64.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        max_size_mb: Tamanho máximo em MB (opcional)
        resize_if_needed: Se True, redimensiona para reduzir tamanho
        
    Returns:
        String base64 do vídeo
    """
    try:
        logger.info(f"Convertendo vídeo para base64: {video_path}")
        
        # Verificar tamanho
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # Redimensionar se necessário
        temp_file = None
        path_to_encode = video_path
        
        if max_size_mb is not None and file_size_mb > max_size_mb and resize_if_needed:
            logger.info(f"Vídeo muito grande ({file_size_mb:.2f} MB). Redimensionando...")
            
            # Criar arquivo temporário
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.close()
            
            # Redimensionar com ffmpeg
            scale_factor = (max_size_mb / file_size_mb) ** 0.5
            
            # Obter dimensões originais
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Calcular novas dimensões
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Garantir que as dimensões são pares
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            # Redimensionar
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"scale={new_width}:{new_height}",
                "-c:v", "libx264",
                "-crf", "28",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "128k",
                temp_file.name,
                "-y"
            ]
            
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            path_to_encode = temp_file.name
            
        # Ler arquivo
        with open(path_to_encode, "rb") as f:
            video_bytes = f.read()
            
        # Converter para base64
        base64_str = base64.b64encode(video_bytes).decode("utf-8")
        
        # Adicionar prefixo
        result = f"data:video/mp4;base64,{base64_str}"
        
        # Limpar arquivo temporário
        if temp_file:
            os.unlink(temp_file.name)
            
        logger.info(f"Vídeo convertido para base64: {len(result)} caracteres")
        
        return result
        
    except Exception as e:
        logger.error(f"Erro ao converter vídeo para base64: {str(e)}")
        raise
