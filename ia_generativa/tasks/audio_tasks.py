"""
Audio Tasks - Tarefas Celery relacionadas a processamento de áudio.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import tempfile
import json
import base64
import numpy as np

from tasks.core import celery_app, task_logger, store_large_result

from models.audio.speech_recognition import SpeechRecognitionManager
from models.audio.speech_synthesis import SpeechSynthesisManager
from models.audio.music_generation import MusicGenerationManager
from preprocessors.audio.audio_processor import load_audio_from_base64, audio_to_base64, normalize_audio
from preprocessors.audio.audio_features import extract_audio_features

logger = logging.getLogger(__name__)

# Inicializar gerenciadores
speech_recognition = SpeechRecognitionManager()
speech_synthesis = SpeechSynthesisManager()
music_generation = MusicGenerationManager()

@celery_app.task(name="audio.transcribe", bind=True)
@task_logger
def transcribe_audio(
    self,
    audio_data: str,
    language: Optional[str] = None,
    task: str = "transcribe",
    return_timestamps: bool = False
) -> Dict[str, Any]:
    """
    Transcreve áudio para texto.
    
    Args:
        audio_data: Áudio em formato base64
        language: Código do idioma para transcrição
        task: Tarefa a ser realizada ("transcribe" ou "translate")
        return_timestamps: Se True, inclui timestamps na transcrição
        
    Returns:
        Dicionário com o texto transcrito e metadados
    """
    try:
        # Carregar áudio
        audio_array, sr = load_audio_from_base64(
            audio_data,
            target_sr=16000,  # Taxa de amostragem padrão para Whisper
            mono=True
        )
        
        # Transcrever áudio
        result = speech_recognition.transcribe(
            audio=audio_array,
            sampling_rate=16000,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            return_all_segments=True
        )
        
        # Retornar resultado
        return {
            "text": result["text"],
            "language": result["language"],
            "duration": result["duration"],
            "segments": result.get("segments"),
            "task": task
        }
        
    except Exception as e:
        logger.error(f"Erro ao transcrever áudio: {str(e)}")
        raise

@celery_app.task(name="audio.synthesize", bind=True)
@task_logger
def synthesize_speech(
    self,
    text: str,
    voice_preset: Optional[str] = None,
    language: Optional[str] = None,
    speaker_id: Optional[int] = None,
    output_format: str = "wav"
) -> Dict[str, Any]:
    """
    Sintetiza fala a partir de texto.
    
    Args:
        text: Texto para sintetizar
        voice_preset: Preset de voz (específico para Bark)
        language: Código do idioma
        speaker_id: ID do falante (específico para VITS)
        output_format: Formato do áudio de saída
        
    Returns:
        Dicionário com o áudio sintetizado e metadados
    """
    try:
        # Determinar taxa de amostragem
        sampling_rate = 24000  # Padrão para Bark
        if speech_synthesis.model_type == "vits":
            sampling_rate = 22050  # Padrão para VITS
            
        # Sintetizar fala
        audio_array = speech_synthesis.synthesize(
            text=text,
            voice_preset=voice_preset,
            language=language,
            speaker_id=speaker_id,
            sampling_rate=sampling_rate
        )
        
        # Converter para base64
        audio_b64 = audio_to_base64(
            audio=audio_array,
            sr=sampling_rate,
            format=output_format,
            normalize=True
        )
        
        # Calcular duração
        duration = len(audio_array) / sampling_rate
        
        return {
            "audio": audio_b64,
            "format": output_format,
            "duration": duration,
            "model": speech_synthesis.model_type,
            "voice": voice_preset if voice_preset else f"speaker_{speaker_id}"
        }
        
    except Exception as e:
        logger.error(f"Erro ao sintetizar fala: {str(e)}")
        raise

@celery_app.task(name="audio.generate_music", bind=True)
@task_logger
def generate_music(
    self,
    prompt: str,
    duration: float = 10.0,
    output_format: str = "wav",
    **kwargs
) -> Dict[str, Any]:
    """
    Gera música a partir de uma descrição textual.
    
    Args:
        prompt: Descrição textual da música
        duration: Duração em segundos
        output_format: Formato do áudio de saída
        **kwargs: Parâmetros adicionais
        
    Returns:
        Dicionário com a música gerada e metadados
    """
    try:
        # Extrair parâmetros adicionais
        top_k = kwargs.get("top_k", 250)
        temperature = kwargs.get("temperature", 1.0)
        guidance_scale = kwargs.get("guidance_scale", 3.0)
        
        # Carregar áudio de referência se fornecido
        audio_conditioned = None
        audio_conditioned_sr = None
        
        if "reference_audio" in kwargs and kwargs["reference_audio"]:
            audio_conditioned, audio_conditioned_sr = load_audio_from_base64(
                kwargs["reference_audio"],
                target_sr=32000,  # Taxa padrão para MusicGen
                mono=True
            )
            
        # Gerar música
        audio_array = music_generation.generate_music(
            prompt=prompt,
            duration=duration,
            sampling_rate=32000,
            top_k=top_k,
            temperature=temperature,
            classifier_free_guidance=guidance_scale,
            audio_conditioned=audio_conditioned,
            audio_conditioned_sr=audio_conditioned_sr
        )
        
        # Converter para base64
        audio_b64 = audio_to_base64(
            audio=audio_array,
            sr=32000,
            format=output_format,
            normalize=True
        )
        
        # Calcular duração real
        real_duration = len(audio_array) / 32000
        
        return {
            "audio": audio_b64,
            "format": output_format,
            "duration": real_duration,
            "model": music_generation.model_name,
            "prompt": prompt
        }
        
    except Exception as e:
        logger.error(f"Erro ao gerar música: {str(e)}")
        raise

@celery_app.task(name="audio.process", bind=True)
@task_logger
def process_audio(
    self,
    audio_data: str,
    operations: List[Dict[str, Any]],
    output_format: str = "wav"
) -> Dict[str, Any]:
    """
    Processa um áudio com várias operações.
    
    Args:
        audio_data: Áudio em formato base64
        operations: Lista de operações a serem aplicadas
        output_format: Formato do áudio de saída
        
    Returns:
        Dicionário com o áudio processado e metadados
    """
    try:
        # Carregar áudio
        audio_array, sr = load_audio_from_base64(
            audio_data,
            normalize=False  # Não normalizar aqui, pois pode ser uma operação solicitada
        )
        
        # Processar cada operação na ordem
        for operation in operations:
            op_type = operation.get("type", "")
            params = operation.get("parameters", {})
            
            if op_type == "normalize":
                # Normalizar volume
                target_db = params.get("target_db", -3.0)
                headroom_db = params.get("headroom_db", 1.0)
                
                audio_array = normalize_audio(
                    audio_array,
                    target_db=target_db,
                    headroom_db=headroom_db
                )
                
            elif op_type == "trim":
                # Recortar trecho de áudio
                start_time = params.get("start_time", 0.0)
                end_time = params.get("end_time")
                
                # Converter tempo para amostras
                start_sample = int(start_time * sr)
                
                if end_time is not None:
                    end_sample = int(end_time * sr)
                    audio_array = audio_array[start_sample:end_sample]
                else:
                    audio_array = audio_array[start_sample:]
                    
            elif op_type == "noise_reduction":
                # Redução de ruído (simplificado)
                import scipy.signal as signal
                
                # Aplicar filtro de mediana para reduzir ruído impulsivo
                audio_array = signal.medfilt(audio_array, kernel_size=3)
                
            elif op_type == "change_speed":
                # Alterar velocidade
                import librosa
                
                speed_factor = params.get("factor", 1.0)
                
                # Alterar velocidade sem afetar pitch
                audio_array = librosa.effects.time_stretch(audio_array, rate=speed_factor)
                
            elif op_type == "change_pitch":
                # Alterar pitch
                import librosa
                
                n_steps = params.get("steps", 0)
                
                # Alterar pitch sem afetar tempo
                if n_steps != 0:
                    audio_array = librosa.effects.pitch_shift(
                        audio_array, 
                        sr=sr, 
                        n_steps=n_steps
                    )
                    
            elif op_type == "filter":
                # Aplicar filtro de frequência
                from scipy import signal
                
                filter_type = params.get("filter_type", "lowpass")
                cutoff_freq = params.get("cutoff_freq", 1000)
                order = params.get("order", 5)
                
                # Normalizar frequência de corte
                nyquist = 0.5 * sr
                normal_cutoff = cutoff_freq / nyquist
                
                # Criar e aplicar filtro
                if filter_type == "lowpass":
                    b, a = signal.butter(order, normal_cutoff, btype='low')
                elif filter_type == "highpass":
                    b, a = signal.butter(order, normal_cutoff, btype='high')
                elif filter_type == "bandpass":
                    # Para bandpass, precisamos de duas frequências
                    high_cutoff = params.get("high_cutoff", 2000)
                    normal_high = high_cutoff / nyquist
                    b, a = signal.butter(order, [normal_cutoff, normal_high], btype='band')
                else:
                    logger.warning(f"Tipo de filtro não suportado: {filter_type}")
                    continue
                    
                audio_array = signal.filtfilt(b, a, audio_array)
                
            else:
                logger.warning(f"Operação desconhecida: {op_type}")
                
        # Converter para base64
        audio_b64 = audio_to_base64(
            audio=audio_array,
            sr=sr,
            format=output_format,
            normalize=True  # Normalizar na saída para evitar clipping
        )
        
        return {
            "audio": audio_b64,
            "format": output_format,
            "duration": len(audio_array) / sr,
            "operations_applied": [op.get("type") for op in operations]
        }
        
    except Exception as e:
        logger.error(f"Erro ao processar áudio: {str(e)}")
        raise

@celery_app.task(name="audio.extract_features", bind=True)
@task_logger
def extract_audio_features_task(
    self,
    audio_data: str,
    features: List[str]
) -> Dict[str, Any]:
    """
    Extrai features de um áudio.
    
    Args:
        audio_data: Áudio em formato base64
        features: Lista de features a extrair
        
    Returns:
        Dicionário com as features extraídas
    """
    try:
        # Carregar áudio
        audio_array, sr = load_audio_from_base64(
            audio_data,
            normalize=True
        )
        
        # Extrair features solicitadas
        features_dict = extract_audio_features(
            audio=audio_array,
            sr=sr,
            features=features
        )
        
        # Converter arrays numpy para listas
        result = {}
        for name, feature in features_dict.items():
            if isinstance(feature, np.ndarray):
                if feature.ndim == 1:
                    result[name] = feature.tolist()
                else:
                    # Para features 2D, simplificar pegando a média
                    result[name] = feature.mean(axis=1).tolist()
            elif isinstance(feature, (int, float)):
                result[name] = feature
                
        return {
            "features": result,
            "duration": len(audio_array) / sr
        }
        
    except Exception as e:
        logger.error(f"Erro ao extrair features: {str(e)}")
        raise
