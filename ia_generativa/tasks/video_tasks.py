"""
Video Tasks - Tarefas Celery relacionadas a processamento de vídeo.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import tempfile
import json
import base64
import cv2
import numpy as np

from tasks.core import celery_app, task_logger, store_large_result

from models.video.video_processor import VideoProcessor
from models.video.video_understanding import VideoUnderstandingManager
from models.video.video_generator import VideoGenerator
from preprocessors.video.video_loader import (
    load_video_from_base64,
    video_to_base64,
    extract_audio_from_video
)

logger = logging.getLogger(__name__)

# Inicializar gerenciadores
video_processor = VideoProcessor()
video_understanding = VideoUnderstandingManager()
video_generator = VideoGenerator()

@celery_app.task(name="video.process", bind=True)
@task_logger
def process_video(
    self,
    video_data: str,
    operations: List[Dict[str, Any]],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    preserve_audio: bool = True
) -> Dict[str, Any]:
    """
    Processa um vídeo aplicando uma lista de operações.
    
    Args:
        video_data: Vídeo em formato base64
        operations: Lista de operações a serem aplicadas
        start_time: Tempo inicial para processamento (segundos)
        end_time: Tempo final para processamento (segundos)
        preserve_audio: Se True, preserva o áudio do vídeo original
        
    Returns:
        Dicionário com o vídeo processado e metadados
    """
    try:
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar vídeo temporariamente
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.mp4")
        
        # Decodificar vídeo de base64
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Processar vídeo
        output_path = video_processor.process_video(
            video_path=input_path,
            operations=operations,
            start_time=start_time,
            end_time=end_time,
            preserve_audio=preserve_audio
        )
        
        # Obter informações do vídeo processado
        cap = cv2.VideoCapture(output_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Converter para base64
        video_base64 = video_to_base64(output_path)
        
        # Limpar arquivos temporários
        try:
            os.remove(input_path)
            if output_path != input_path:
                os.remove(output_path)
        except:
            pass
            
        # Se o vídeo resultante for muito grande, armazenar em arquivo
        if len(video_base64) > 10 * 1024 * 1024:  # Maior que 10MB
            result = store_large_result(
                {
                    "video": video_base64,
                    "format": "mp4",
                    "duration": duration,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "operations_applied": len(operations)
                },
                self.request.id
            )
            
            # Adicionar informações resumidas
            result.update({
                "format": "mp4",
                "duration": duration,
                "width": width,
                "height": height,
                "fps": fps,
                "operations_applied": len(operations),
                "video_size_estimate": f"{len(video_base64)/(1024*1024):.1f} MB"
            })
            
            return result
        else:
            return {
                "video": video_base64,
                "format": "mp4",
                "duration": duration,
                "width": width,
                "height": height,
                "fps": fps,
                "operations_applied": len(operations)
            }
            
    except Exception as e:
        logger.error(f"Erro ao processar vídeo: {str(e)}")
        raise

@celery_app.task(name="video.analyze", bind=True)
@task_logger
def analyze_video(
    self,
    video_data: str,
    analyze_objects: bool = True,
    classify_content: bool = True,
    generate_caption: bool = True,
    frame_interval: int = 10
) -> Dict[str, Any]:
    """
    Realiza uma análise completa do conteúdo de um vídeo.
    
    Args:
        video_data: Vídeo em formato base64
        analyze_objects: Se True, detecta objetos
        classify_content: Se True, classifica o conteúdo
        generate_caption: Se True, gera descrição
        frame_interval: Intervalo entre frames
        
    Returns:
        Dicionário com os resultados da análise
    """
    try:
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar vídeo temporariamente
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.mp4")
        
        # Decodificar vídeo de base64
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Analisar vídeo
        analysis_result = video_understanding.analyze_video_content(
            video_path=input_path,
            analyze_objects=analyze_objects,
            classify_content=classify_content,
            generate_caption=generate_caption,
            frame_interval=frame_interval
        )
        
        # Limpar arquivo temporário
        try:
            os.remove(input_path)
        except:
            pass
            
        return analysis_result
        
    except Exception as e:
        logger.error(f"Erro ao analisar vídeo: {str(e)}")
        raise

@celery_app.task(name="video.create_slideshow", bind=True)
@task_logger
def create_slideshow(
    self,
    images: List[str],
    fps: float = 1.0,
    transition_frames: int = 30,
    transition_type: str = "fade",
    audio: Optional[str] = None,
    duration_per_image: Optional[float] = None,
    text_overlay: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Cria um slideshow a partir de imagens.
    
    Args:
        images: Lista de imagens em formato base64
        fps: Frames por segundo
        transition_frames: Número de frames para transição
        transition_type: Tipo de transição (fade, wipe)
        audio: Áudio opcional em formato base64
        duration_per_image: Duração em segundos para cada imagem
        text_overlay: Lista de textos para sobrepor em cada imagem
        
    Returns:
        Dicionário com o slideshow gerado e metadados
    """
    try:
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar imagens temporariamente
        image_paths = []
        
        for i, img_b64 in enumerate(images):
            # Decodificar imagem de base64
            if "base64," in img_b64:
                img_b64 = img_b64.split("base64,")[1]
                
            img_bytes = base64.b64decode(img_b64)
            
            # Salvar imagem
            img_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
                
            image_paths.append(img_path)
            
        # Salvar áudio temporariamente (se fornecido)
        audio_path = None
        if audio:
            # Decodificar áudio de base64
            if "base64," in audio:
                audio = audio.split("base64,")[1]
                
            audio_bytes = base64.b64decode(audio)
            
            # Salvar áudio
            audio_path = os.path.join(temp_dir, f"audio_{int(time.time())}.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
                
        # Criar slideshow
        output_path = os.path.join(temp_dir, f"output_video_{int(time.time())}.mp4")
        
        slideshow_path = video_generator.create_slideshow(
            image_paths=image_paths,
            output_path=output_path,
            fps=fps,
            transition_frames=transition_frames,
            transition_type=transition_type,
            audio_path=audio_path,
            duration_per_image=duration_per_image,
            text_overlay=text_overlay
        )
        
        # Obter informações do slideshow
        cap = cv2.VideoCapture(slideshow_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Converter para base64
        video_base64 = video_to_base64(slideshow_path)
        
        # Limpar arquivos temporários
        try:
            for path in image_paths:
                os.remove(path)
            if audio_path:
                os.remove(audio_path)
            os.remove(slideshow_path)
        except:
            pass
            
        return {
            "video": video_base64,
            "format": "mp4",
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "image_count": len(images)
        }
        
    except Exception as e:
        logger.error(f"Erro ao criar slideshow: {str(e)}")
        raise

@celery_app.task(name="video.extract_audio", bind=True)
@task_logger
def extract_audio_from_video_task(
    self,
    video_data: str,
    output_format: str = "mp3",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Extrai a trilha de áudio de um vídeo.
    
    Args:
        video_data: Vídeo em formato base64
        output_format: Formato do áudio de saída
        start_time: Tempo inicial para extração (segundos)
        end_time: Tempo final para extração (segundos)
        
    Returns:
        Dicionário com o áudio extraído e metadados
    """
    try:
        # Converter de vídeo para áudio
        audio_b64 = video_to_audio(
            video_data=video_data,
            output_format=output_format,
            start_time=start_time,
            end_time=end_time
        )
        
        # Para evitar dependência circular
        from orchestrator.converters import video_to_audio
        
        # Carregar áudio para obter duração
        audio_array, sr = load_audio_from_base64(audio_b64)
        duration = len(audio_array) / sr
        
        return {
            "audio": audio_b64,
            "format": output_format,
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"Erro ao extrair áudio de vídeo: {str(e)}")
        raise
