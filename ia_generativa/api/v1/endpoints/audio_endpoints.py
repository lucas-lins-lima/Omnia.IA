"""
Endpoints da API relacionados a processamento de áudio.
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
import numpy as np

from models.audio.speech_recognition import SpeechRecognitionManager
from models.audio.speech_synthesis import SpeechSynthesisManager
from models.audio.music_generation import MusicGenerationManager
from preprocessors.audio.audio_processor import (
    load_audio_from_base64,
    normalize_audio,
    convert_audio_format,
    audio_to_base64
)
from preprocessors.audio.audio_features import extract_audio_features
from api.v1.schemas.audio_schemas import (
    TranscriptionRequest,
    TextToSpeechRequest,
    MusicGenerationRequest,
    AudioProcessingRequest,
    AudioFeatureExtractionRequest,
    TranscriptionResponse,
    SpeechSynthesisResponse,
    MusicGenerationResponse,
    AudioProcessingResponse,
    AudioFeatureExtractionResponse,
    AudioModelInfoResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/audio",
    tags=["audio"],
    responses={404: {"description": "Not found"}},
)

# Instanciar gerenciadores
speech_recognition_manager = SpeechRecognitionManager()
speech_synthesis_manager = SpeechSynthesisManager()
music_generation_manager = MusicGenerationManager()

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcreve áudio para texto.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para transcrição de áudio")
        
        # Carregar áudio do base64
        audio_array, sr = load_audio_from_base64(
            request.audio,
            format=request.format,
            target_sr=16000,  # Taxa de amostragem padrão para Whisper
            mono=True
        )
        
        # Configurar idioma
        language = request.language
        if language == "auto":
            language = None  # Whisper detectará automaticamente
            
        # Transcrever áudio
        result = speech_recognition_manager.transcribe(
            audio=audio_array,
            sampling_rate=16000,
            language=language,
            task=request.task,
            return_timestamps=request.return_timestamps,
            return_all_segments=True
        )
        
        # Calcular tempo de processamento
        processing_time = time.time() - start_time
        
        return TranscriptionResponse(
            text=result["text"],
            language=result["language"],
            duration=result["duration"],
            segments=result.get("segments"),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro ao transcrever áudio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao transcrever áudio: {str(e)}")
        
@router.post("/synthesize", response_model=SpeechSynthesisResponse)
async def synthesize_speech(request: TextToSpeechRequest):
    """
    Sintetiza fala a partir de texto.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para síntese de fala: {len(request.text)} caracteres")
        
        # Configurar a voz e idioma
        voice_preset = request.voice_preset
        language = request.language
        speaker_id = request.speaker_id
        
        # Determinar taxa de amostragem
        sampling_rate = 24000  # Padrão para Bark
        if request.model.value == "vits":
            sampling_rate = 22050  # Padrão para VITS
            
        # Sintetizar fala
        audio_array = speech_synthesis_manager.synthesize(
            text=request.text,
            voice_preset=voice_preset,
            language=language,
            speaker_id=speaker_id,
            sampling_rate=sampling_rate
        )
        
        # Converter para base64
        audio_b64 = audio_to_base64(
            audio=audio_array,
            sr=sampling_rate,
            format=request.output_format,
            normalize=True
        )
        
        # Calcular duração
        duration = len(audio_array) / sampling_rate
        
        return SpeechSynthesisResponse(
            audio=audio_b64,
            format=request.output_format,
            duration=duration,
            model=request.model.value,
            voice=voice_preset if voice_preset else f"speaker_{speaker_id}"
        )
        
    except Exception as e:
        logger.error(f"Erro ao sintetizar fala: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao sintetizar fala: {str(e)}")
        
@router.post("/generate-music", response_model=MusicGenerationResponse)
async def generate_music(request: MusicGenerationRequest):
    """
    Gera música a partir de uma descrição textual.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para geração de música: {request.prompt[:50]}...")
        
        # Carregar áudio de referência se fornecido
        audio_conditioned = None
        audio_conditioned_sr = None
        
        if request.reference_audio:
            audio_conditioned, audio_conditioned_sr = load_audio_from_base64(
                request.reference_audio,
                target_sr=32000,  # Taxa padrão para MusicGen
                mono=True
            )
            
        # Gerar música
        audio_array = music_generation_manager.generate_music(
            prompt=request.prompt,
            duration=request.duration,
            sampling_rate=32000,
            top_k=request.top_k,
            temperature=request.temperature,
            classifier_free_guidance=request.guidance_scale,
            audio_conditioned=audio_conditioned,
            audio_conditioned_sr=audio_conditioned_sr
        )
        
        # Converter para base64
        audio_b64 = audio_to_base64(
            audio=audio_array,
            sr=32000,
            format=request.output_format,
            normalize=True
        )
        
        # Calcular duração real
        duration = len(audio_array) / 32000
        
        return MusicGenerationResponse(
            audio=audio_b64,
            format=request.output_format,
            duration=duration,
            model=request.model.value,
            prompt=request.prompt
        )
        
    except Exception as e:
        logger.error(f"Erro ao gerar música: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao gerar música: {str(e)}")
        
@router.post("/process", response_model=AudioProcessingResponse)
async def process_audio(request: AudioProcessingRequest):
    """
    Processa um áudio com várias operações.
    """
    try:
        logger.info(f"Recebida requisição para processamento de áudio com operações: {request.operations}")
        
        # Carregar áudio do base64
        audio_array, sr = load_audio_from_base64(
            request.audio,
            format=request.format,
            normalize=False  # Não normalizar aqui, pois pode ser uma operação solicitada
        )
        
        # Processar cada operação na ordem
        for operation in request.operations:
            if operation == "normalize":
                target_db = request.parameters.get("target_db", -3.0)
                headroom_db = request.parameters.get("headroom_db", 1.0)
                audio_array = normalize_audio(
                    audio_array,
                    target_db=target_db,
                    headroom_db=headroom_db
                )
                
            elif operation == "trim_silence":
                import librosa
                
                threshold = request.parameters.get("threshold", 0.01)
                top_db = request.parameters.get("top_db", 60)
                
                # Remover silêncio do início e fim
                audio_array, _ = librosa.effects.trim(
                    audio_array,
                    top_db=top_db,
                    frame_length=2048,
                    hop_length=512
                )
                
            elif operation == "noise_reduction":
                # Implementar redução de ruído básica
                # Isto é uma simplificação; uma implementação real seria mais complexa
                import scipy.signal as signal
                
                smoothing_factor = request.parameters.get("smoothing_factor", 0.1)
                
                # Aplicar filtro de mediana para reduzir ruído impulsivo
                if len(audio_array) > 0:
                    audio_array = signal.medfilt(audio_array, kernel_size=3)
                
            elif operation == "change_tempo":
                import librosa
                
                tempo_factor = request.parameters.get("factor", 1.0)
                
                # Mudar o tempo sem afetar o pitch
                audio_array = librosa.effects.time_stretch(audio_array, rate=tempo_factor)
                
            elif operation == "change_pitch":
                import librosa
                
                n_steps = request.parameters.get("steps", 0)
                
                # Mudar o pitch sem afetar o tempo
                if n_steps != 0:
                    audio_array = librosa.effects.pitch_shift(
                        audio_array, 
                        sr=sr, 
                        n_steps=n_steps
                    )
                    
            elif operation == "low_pass_filter":
                from scipy import signal
                
                cutoff_freq = request.parameters.get("cutoff_freq", 1000)
                order = request.parameters.get("order", 5)
                
                # Normalizar frequência de corte
                nyquist = 0.5 * sr
                normal_cutoff = cutoff_freq / nyquist
                
                # Criar e aplicar filtro
                b, a = signal.butter(order, normal_cutoff, btype='low')
                audio_array = signal.filtfilt(b, a, audio_array)
                
            elif operation == "high_pass_filter":
                from scipy import signal
                
                cutoff_freq = request.parameters.get("cutoff_freq", 200)
                order = request.parameters.get("order", 5)
                
                # Normalizar frequência de corte
                nyquist = 0.5 * sr
                normal_cutoff = cutoff_freq / nyquist
                
                # Criar e aplicar filtro
                b, a = signal.butter(order, normal_cutoff, btype='high')
                audio_array = signal.filtfilt(b, a, audio_array)
                
            elif operation == "reverb":
                # Implementar reverb simples
                # Isto é uma simplificação; uma implementação real seria mais complexa
                from scipy import signal
                
                decay = request.parameters.get("decay", 0.5)
                room_scale = request.parameters.get("room_scale", 0.8)
                
                # Tamanho da resposta ao impulso
                reverb_length = int(sr * room_scale)
                
                # Criar resposta ao impulso exponencial decrescente
                impulse_response = np.exp(-decay * np.arange(reverb_length) / sr)
                
                # Normalizar
                impulse_response = impulse_response / np.sum(impulse_response)
                
                # Aplicar convolução
                audio_array = signal.fftconvolve(audio_array, impulse_response, mode='full')
                
                # Normalizar o resultado
                audio_array = normalize_audio(audio_array)
                
        # Converter para base64
        audio_b64 = audio_to_base64(
            audio=audio_array,
            sr=sr,
            format=request.output_format,
            normalize=True  # Normalizar na saída para evitar clipping
        )
        
        return AudioProcessingResponse(
            audio=audio_b64,
            format=request.output_format,
            duration=len(audio_array) / sr,
            operations_applied=request.operations
        )
        
    except Exception as e:
        logger.error(f"Erro ao processar áudio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao processar áudio: {str(e)}")
        
@router.post("/extract-features", response_model=AudioFeatureExtractionResponse)
async def extract_audio_features_endpoint(request: AudioFeatureExtractionRequest):
    """
    Extrai features de um áudio.
    """
    try:
        logger.info(f"Recebida requisição para extração de features: {request.features}")
        
        # Carregar áudio do base64
        audio_array, sr = load_audio_from_base64(
            request.audio,
            format=request.format,
            normalize=True
        )
        
        # Extrair features solicitadas
        features_dict = extract_audio_features(
            audio=audio_array,
            sr=sr,
            features=request.features
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
                
        return AudioFeatureExtractionResponse(
            features=result,
            duration=len(audio_array) / sr
        )
        
    except Exception as e:
        logger.error(f"Erro ao extrair features: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao extrair features: {str(e)}")
        
@router.get("/models/speech-recognition/info", response_model=AudioModelInfoResponse)
async def get_speech_recognition_model_info():
    """
    Retorna informações sobre o modelo de reconhecimento de fala carregado.
    """
    try:
        model_info = speech_recognition_manager.get_model_info()
        return AudioModelInfoResponse(
            model_name=model_info["model_name"],
            model_type="speech-to-text",
            status=model_info["status"],
            device=model_info["device"],
            memory_used_gb=model_info.get("memory_used_gb"),
            additional_info={
                "flash_attention": model_info.get("flash_attention")
            }
        )
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo ASR: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações do modelo: {str(e)}")
        
@router.get("/models/speech-synthesis/info", response_model=AudioModelInfoResponse)
async def get_speech_synthesis_model_info():
    """
    Retorna informações sobre o modelo de síntese de fala carregado.
    """
    try:
        model_info = speech_synthesis_manager.get_model_info()
        return AudioModelInfoResponse(
            model_name=model_info["model_name"],
            model_type=model_info["model_type"],
            status=model_info["status"],
            device=model_info["device"],
            memory_used_gb=model_info.get("memory_used_gb"),
            additional_info={
                "voices_available": model_info.get("voices_available", 0)
            }
        )
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo TTS: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações do modelo: {str(e)}")
        
@router.get("/models/music-generation/info", response_model=AudioModelInfoResponse)
async def get_music_generation_model_info():
    """
    Retorna informações sobre o modelo de geração de música carregado.
    """
    try:
        model_info = music_generation_manager.get_model_info()
        return AudioModelInfoResponse(
            model_name=model_info["model_name"],
            model_type="music-generation",
            status=model_info["status"],
            device=model_info["device"],
            memory_used_gb=model_info.get("memory_used_gb")
        )
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo de geração de música: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações do modelo: {str(e)}")

@router.post("/models/unload")
async def unload_audio_models():
    """
    Descarrega todos os modelos de áudio da memória.
    """
    try:
        speech_recognition_manager.unload_model()
        speech_synthesis_manager.unload_model()
        music_generation_manager.unload_model()
        return {"status": "success", "message": "Modelos de áudio descarregados com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao descarregar modelos de áudio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao descarregar modelos: {str(e)}")

@router.get("/voices")
async def get_available_voices():
    """
    Retorna as vozes disponíveis para síntese de fala.
    """
    try:
        if speech_synthesis_manager.model is None:
            # Carregar modelo temporariamente para obter vozes
            speech_synthesis_manager.load_model()
            
        voices = speech_synthesis_manager.get_available_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Erro ao obter vozes disponíveis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter vozes: {str(e)}")
