"""
Endpoints da API relacionados a processamento de vídeo.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, File, UploadFile, Form
from fastapi.responses import Response
import time
import json
import logging
from typing import List, Dict, Optional, Union, Any
import os
import io
import base64
import tempfile
import cv2
import numpy as np
import subprocess
from PIL import Image

from models.video.video_processor import VideoProcessor
from models.video.video_understanding import VideoUnderstandingManager
from models.video.video_generator import VideoGenerator
from preprocessors.video.video_loader import (
    load_video_from_base64,
    video_to_base64,
    extract_audio_from_video
)
from api.v1.schemas.video_schemas import (
    VideoProcessRequest,
    VideoTrimRequest,
    VideoConcatenateRequest,
    VideoAddTextRequest,
    VideoSlideshowRequest,
    VideoAnalysisRequest,
    VideoMontageRequest,
    VideoExtractAudioRequest,
    VideoExtractSceneRequest,
    VideoSpeedChangeRequest,
    VideoTimelapseRequest,
    VideoResponse,
    AudioResponse,
    VideoAnalysisResponse,
    SceneDetectionResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/video",
    tags=["video"],
    responses={404: {"description": "Not found"}},
)

# Instanciar gerenciadores
video_processor = VideoProcessor()
video_understanding = VideoUnderstandingManager()
video_generator = VideoGenerator()

@router.post("/process", response_model=VideoResponse)
async def process_video(request: VideoProcessRequest):
    """
    Processa um vídeo aplicando uma lista de operações.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para processamento de vídeo com {len(request.operations)} operações")
        
        # Salvar vídeo temporariamente
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.{request.format}")
        
        # Decodificar vídeo de base64
        video_data = request.video
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Converter operações para formato esperado pelo processador
        operations = []
        for op in request.operations:
            # Converter para dicionário e incluir apenas os campos não-nulos
            op_dict = op.dict(exclude_none=True)
            operations.append(op_dict)
            
        # Processar vídeo
        output_path = video_processor.process_video(
            video_path=input_path,
            operations=operations,
            start_time=request.start_time,
            end_time=request.end_time,
            sample_rate=request.sample_rate,
            preserve_audio=request.preserve_audio
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
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return VideoResponse(
            video=video_base64,
            format=request.output_format,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro ao processar vídeo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao processar vídeo: {str(e)}")
        
@router.post("/trim", response_model=VideoResponse)
async def trim_video(request: VideoTrimRequest):
    """
    Recorta um vídeo entre dois pontos de tempo.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para recorte de vídeo: {request.start_time}s - {request.end_time}s")
        
        # Salvar vídeo temporariamente
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.{request.format}")
        
        # Decodificar vídeo de base64
        video_data = request.video
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Recortar vídeo
        output_path = video_processor.trim_video(
            video_path=input_path,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        # Obter informações do vídeo recortado
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
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return VideoResponse(
            video=video_base64,
            format=request.output_format,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro ao recortar vídeo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao recortar vídeo: {str(e)}")
        
@router.post("/concatenate", response_model=VideoResponse)
async def concatenate_videos(request: VideoConcatenateRequest):
    """
    Concatena múltiplos vídeos.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para concatenar {len(request.videos)} vídeos")
        
        # Salvar vídeos temporariamente
        temp_dir = tempfile.gettempdir()
        input_paths = []
        
        for i, video_data in enumerate(request.videos):
            input_path = os.path.join(temp_dir, f"input_video_{i}_{int(time.time())}.{request.format}")
            
            # Decodificar vídeo de base64
            if "base64," in video_data:
                video_data = video_data.split("base64,")[1]
                
            with open(input_path, "wb") as f:
                f.write(base64.b64decode(video_data))
                
            input_paths.append(input_path)
            
        # Concatenar vídeos
        output_path = video_processor.concatenate_videos(
            video_paths=input_paths,
            crossfade_duration=request.crossfade_duration
        )
        
        # Obter informações do vídeo concatenado
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
            for path in input_paths:
                os.remove(path)
            if output_path not in input_paths:
                os.remove(output_path)
        except:
            pass
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return VideoResponse(
            video=video_base64,
            format=request.output_format,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro ao concatenar vídeos: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao concatenar vídeos: {str(e)}")
        
@router.post("/add-text", response_model=VideoResponse)
async def add_text_to_video(request: VideoAddTextRequest):
    """
    Adiciona texto a um vídeo.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para adicionar texto a vídeo: {request.text}")
        
        # Salvar vídeo temporariamente
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.{request.format}")
        
        # Decodificar vídeo de base64
        video_data = request.video
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Adicionar texto ao vídeo
        output_path = video_generator.add_text_to_video(
            video_path=input_path,
            text=request.text,
            position=request.position,
            font_size=request.font_size,
            color=request.color,
            background=request.background,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        # Obter informações do vídeo resultante
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
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return VideoResponse(
            video=video_base64,
            format=request.output_format,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro ao adicionar texto ao vídeo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar texto ao vídeo: {str(e)}")
        
@router.post("/slideshow", response_model=VideoResponse)
async def create_slideshow(request: VideoSlideshowRequest):
    """
    Cria um slideshow a partir de imagens.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para criar slideshow com {len(request.images)} imagens")
        
        # Salvar imagens temporariamente
        temp_dir = tempfile.gettempdir()
        image_paths = []
        
        for i, image_data in enumerate(request.images):
            image_path = os.path.join(temp_dir, f"image_{i}_{int(time.time())}.jpg")
            
            # Decodificar imagem de base64
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
                
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_data))
                
            image_paths.append(image_path)
            
        # Salvar áudio temporariamente, se fornecido
        audio_path = None
        if request.audio:
            audio_data = request.audio
            if "base64," in audio_data:
                audio_data = audio_data.split("base64,")[1]
                
            audio_path = os.path.join(temp_dir, f"audio_{int(time.time())}.mp3")
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_data))
                
        # Criar slideshow
        output_path = video_generator.create_slideshow(
            image_paths=image_paths,
            fps=request.fps,
            transition_frames=request.transition_frames,
            transition_type=request.transition_type,
            audio_path=audio_path,
            duration_per_image=request.duration_per_image,
            text_overlay=request.text_overlay
        )
        
        # Obter informações do slideshow
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
            for path in image_paths:
                os.remove(path)
            if audio_path:
                os.remove(audio_path)
            os.remove(output_path)
        except:
            pass
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return VideoResponse(
            video=video_base64,
            format=request.output_format,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro ao criar slideshow: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao criar slideshow: {str(e)}")
        
@router.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """
    Realiza uma análise completa do conteúdo de um vídeo.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para análise de vídeo")
        
        # Salvar vídeo temporariamente
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.{request.format}")
        
        # Decodificar vídeo de base64
        video_data = request.video
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Analisar vídeo
        analysis_result = video_understanding.analyze_video_content(
            video_path=input_path,
            analyze_objects=request.analyze_objects,
            classify_content=request.classify_content,
            generate_caption=request.generate_caption,
            frame_interval=request.frame_interval
        )
        
        # Limpar arquivos temporários
        try:
            os.remove(input_path)
        except:
            pass
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        # Adicionar tempo de processamento
        analysis_result["processing_time"] = processing_time
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Erro ao analisar vídeo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao analisar vídeo: {str(e)}")
        
@router.post("/montage", response_model=VideoResponse)
async def create_montage(request: VideoMontageRequest):
    """
    Cria uma montagem de múltiplos vídeos.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para criar montagem com {len(request.videos)} vídeos")
        
        # Salvar vídeos temporariamente
        temp_dir = tempfile.gettempdir()
        input_paths = []
        
        for i, video_data in enumerate(request.videos):
            input_path = os.path.join(temp_dir, f"input_video_{i}_{int(time.time())}.{request.format}")
            
            # Decodificar vídeo de base64
            if "base64," in video_data:
                video_data = video_data.split("base64,")[1]
                
            with open(input_path, "wb") as f:
                f.write(base64.b64decode(video_data))
                
            input_paths.append(input_path)
            
        # Criar montagem
        output_path = video_generator.create_montage(
            video_paths=input_paths,
            layout=request.layout,
            grid_size=request.grid_size,
            output_size=request.output_size,
            include_audio=request.include_audio
        )
        
        # Obter informações da montagem
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
            for path in input_paths:
                os.remove(path)
            os.remove(output_path)
        except:
            pass
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return VideoResponse(
            video=video_base64,
            format=request.output_format,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro ao criar montagem: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao criar montagem: {str(e)}")
        
@router.post("/extract-audio", response_model=AudioResponse)
async def extract_audio(request: VideoExtractAudioRequest):
    """
    Extrai a trilha de áudio de um vídeo.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para extrair áudio de vídeo")
        
        # Salvar vídeo temporariamente
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.{request.format}")
        
        # Decodificar vídeo de base64
        video_data = request.video
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Extrair áudio
        output_path = os.path.join(temp_dir, f"extracted_audio_{int(time.time())}.{request.output_format}")
        
        audio_path = extract_audio_from_video(
            video_path=input_path,
            output_path=output_path,
            format=request.output_format,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        # Obter duração do áudio
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # Converter de ms para segundos
        
        # Converter para base64
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            
        # Adicionar prefixo para data URI
        audio_base64 = f"data:audio/{request.output_format};base64,{audio_base64}"
        
        # Limpar arquivos temporários
        try:
            os.remove(input_path)
            os.remove(audio_path)
        except:
            pass
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return AudioResponse(
            audio=audio_base64,
            format=request.output_format,
            duration=duration
        )
        
    except Exception as e:
        logger.error(f"Erro ao extrair áudio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao extrair áudio: {str(e)}")
        
@router.post("/detect-scenes", response_model=SceneDetectionResponse)
async def detect_scenes(request: VideoExtractSceneRequest):
    """
    Detecta transições entre cenas em um vídeo.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para detectar cenas em vídeo")
        
        # Salvar vídeo temporariamente
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.{request.format}")
        
        # Decodificar vídeo de base64
        video_data = request.video
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Detectar cenas
        scene_transitions = video_processor.extract_scene_transitions(
            video_path=input_path,
            threshold=request.threshold,
            min_scene_length=request.min_scene_length
        )
        
        # Obter duração do vídeo
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Limpar arquivos temporários
        try:
            os.remove(input_path)
        except:
            pass
            
        return SceneDetectionResponse(
            scenes=scene_transitions,
            duration=duration,
            total_scenes=len(scene_transitions)
        )
        
    except Exception as e:
        logger.error(f"Erro ao detectar cenas: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao detectar cenas: {str(e)}")
        
@router.post("/change-speed", response_model=VideoResponse)
async def change_video_speed(request: VideoSpeedChangeRequest):
    """
    Altera a velocidade de um vídeo.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para alterar velocidade de vídeo: fator {request.speed_factor}")
        
        # Salvar vídeo temporariamente
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.{request.format}")
        
        # Decodificar vídeo de base64
        video_data = request.video
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Alterar velocidade
        output_path = video_processor.change_video_speed(
            video_path=input_path,
            speed_factor=request.speed_factor,
            preserve_audio_pitch=request.preserve_audio_pitch
        )
        
        # Obter informações do vídeo resultante
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
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return VideoResponse(
            video=video_base64,
            format=request.output_format,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro ao alterar velocidade do vídeo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao alterar velocidade do vídeo: {str(e)}")
        
@router.post("/timelapse", response_model=VideoResponse)
async def create_timelapse(request: VideoTimelapseRequest):
    """
    Cria um timelapse a partir de um vídeo.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para criar timelapse: fator {request.speed_factor}")
        
        # Salvar vídeo temporariamente
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, f"input_video_{int(time.time())}.{request.format}")
        
        # Decodificar vídeo de base64
        video_data = request.video
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Criar timelapse
        output_path = video_processor.create_timelapse(
            video_path=input_path,
            speed_factor=request.speed_factor,
            frame_interval=request.frame_interval
        )
        
        # Obter informações do vídeo resultante
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
            
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return VideoResponse(
            video=video_base64,
            format=request.output_format,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro ao criar timelapse: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao criar timelapse: {str(e)}")

@router.get("/models/understanding/info")
async def get_understanding_model_info():
    """
    Retorna informações sobre os modelos de análise de vídeo carregados.
    """
    try:
        status = {
            "video_classifier": video_understanding.video_classifier is not None,
            "object_detector": video_understanding.object_detector is not None,
            "video_captioner": video_understanding.video_captioner is not None,
        }
        
        return {
            "status": status,
            "device": video_understanding.device,
            "frame_sample_rate": video_understanding.frame_sample_rate
        }
    except Exception as e:
        logger.error(f"Erro ao obter informações dos modelos: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações dos modelos: {str(e)}")

@router.post("/models/unload")
async def unload_video_models():
    """
    Descarrega todos os modelos de vídeo da memória.
    """
    try:
        video_understanding.unload_models()
        return {"status": "success", "message": "Modelos de vídeo descarregados com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao descarregar modelos de vídeo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao descarregar modelos: {str(e)}")
