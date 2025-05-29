"""
Schemas para validação de dados de áudio na API.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Dict, Optional, Union, Any, Tuple
from enum import Enum
import base64
import re

class AudioFormat(str, Enum):
    """Formatos de áudio suportados."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    
class SpeechModel(str, Enum):
    """Modelos de ASR suportados."""
    WHISPER_TINY = "openai/whisper-tiny"
    WHISPER_BASE = "openai/whisper-base"
    WHISPER_SMALL = "openai/whisper-small"
    WHISPER_MEDIUM = "openai/whisper-medium"
    WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    
class TTSModel(str, Enum):
    """Modelos de TTS suportados."""
    BARK_SMALL = "suno/bark-small"
    BARK = "suno/bark"
    VITS = "vits"
    
class MusicModel(str, Enum):
    """Modelos de geração de música suportados."""
    MUSICGEN_SMALL = "facebook/musicgen-small"
    MUSICGEN_MEDIUM = "facebook/musicgen-medium"
    MUSICGEN_LARGE = "facebook/musicgen-large"
    
class TranscriptionRequest(BaseModel):
    """Requisição para transcrição de áudio."""
    audio: str = Field(..., description="Áudio em formato base64")
    format: AudioFormat = Field(AudioFormat.WAV, description="Formato do áudio")
    model: SpeechModel = Field(SpeechModel.WHISPER_SMALL, description="Modelo ASR a ser usado")
    language: Optional[str] = Field(None, description="Código do idioma (ex: 'pt', 'en', 'auto')")
    return_timestamps: bool = Field(False, description="Se True, inclui timestamps na transcrição")
    task: str = Field("transcribe", description="Tarefa ('transcribe' ou 'translate')")
    
    @validator('audio')
    def validate_base64_audio(cls, v):
        """Valida que a string é um base64 válido de áudio."""
        # Verificar se começa com data:audio ou é base64 puro
        if not (v.startswith('data:audio/') or re.match(r'^[A-Za-z0-9+/=]+$', v)):
            raise ValueError("Formato de áudio base64 inválido")
        return v
    
    @validator('task')
    def validate_task(cls, v):
        """Valida que a tarefa é válida."""
        if v not in ["transcribe", "translate"]:
            raise ValueError("Tarefa deve ser 'transcribe' ou 'translate'")
        return v
    
class TextToSpeechRequest(BaseModel):
    """Requisição para síntese de fala."""
    text: str = Field(..., description="Texto para converter em fala")
    model: TTSModel = Field(TTSModel.BARK_SMALL, description="Modelo TTS a ser usado")
    voice_preset: Optional[str] = Field("v2/en_speaker_6", description="Preset de voz (para Bark)")
    speaker_id: Optional[int] = Field(0, description="ID do falante (para VITS)")
    language: Optional[str] = Field(None, description="Código do idioma")
    output_format: AudioFormat = Field(AudioFormat.WAV, description="Formato do áudio de saída")
    
    @validator('text')
    def text_not_empty(cls, v):
        """Valida que o texto não está vazio."""
        if not v.strip():
            raise ValueError("Texto não pode estar vazio")
        return v
    
class MusicGenerationRequest(BaseModel):
    """Requisição para geração de música."""
    prompt: str = Field(..., description="Descrição textual da música a ser gerada")
    model: MusicModel = Field(MusicModel.MUSICGEN_SMALL, description="Modelo de geração de música")
    duration: float = Field(10.0, description="Duração da música em segundos", ge=1.0, le=30.0)
    top_k: int = Field(250, description="Parâmetro de filtragem top-k", ge=1)
    temperature: float = Field(1.0, description="Temperatura para geração", ge=0.1, le=2.0)
    guidance_scale: float = Field(3.0, description="Escala de orientação", ge=1.0, le=15.0)
    output_format: AudioFormat = Field(AudioFormat.WAV, description="Formato do áudio de saída")
    reference_audio: Optional[str] = Field(None, description="Áudio de referência em base64 (opcional)")
    
    @validator('prompt')
    def prompt_not_empty(cls, v):
        """Valida que o prompt não está vazio."""
        if not v.strip():
            raise ValueError("Prompt não pode estar vazio")
        return v
    
class AudioProcessingRequest(BaseModel):
    """Requisição para processamento de áudio."""
    audio: str = Field(..., description="Áudio em formato base64")
    format: AudioFormat = Field(AudioFormat.WAV, description="Formato do áudio")
    operations: List[str] = Field(..., description="Lista de operações a serem aplicadas")
    parameters: Dict[str, Any] = Field({}, description="Parâmetros para as operações")
    output_format: AudioFormat = Field(AudioFormat.WAV, description="Formato do áudio de saída")
    
    @validator('operations')
    def validate_operations(cls, v):
        """Valida que as operações são válidas."""
        valid_operations = [
            "normalize", "trim_silence", "noise_reduction", 
            "change_tempo", "change_pitch", "low_pass_filter", 
            "high_pass_filter", "reverb"
        ]
        
        for op in v:
            if op not in valid_operations:
                raise ValueError(f"Operação não suportada: {op}")
                
        return v
    
class AudioFeatureExtractionRequest(BaseModel):
    """Requisição para extração de features de áudio."""
    audio: str = Field(..., description="Áudio em formato base64")
    format: AudioFormat = Field(AudioFormat.WAV, description="Formato do áudio")
    features: List[str] = Field(..., description="Features a serem extraídas")
    
    @validator('features')
    def validate_features(cls, v):
        """Valida que as features são válidas."""
        valid_features = [
            "mfcc", "mel_spectrogram", "chroma", "onset_strength",
            "tempo", "pitch", "spectral_contrast", "tonnetz"
        ]
        
        for feature in v:
            if feature not in valid_features:
                raise ValueError(f"Feature não suportada: {feature}")
                
        return v
    
class TranscriptionResponse(BaseModel):
    """Resposta de transcrição de áudio."""
    text: str = Field(..., description="Texto transcrito")
    language: str = Field(..., description="Idioma detectado ou usado")
    duration: float = Field(..., description="Duração do áudio em segundos")
    segments: Optional[List[Dict[str, Any]]] = Field(None, description="Segmentos individuais com timestamps")
    processing_time: float = Field(..., description="Tempo de processamento em segundos")
    
class SpeechSynthesisResponse(BaseModel):
    """Resposta de síntese de fala."""
    audio: str = Field(..., description="Áudio sintetizado em formato base64")
    format: str = Field(..., description="Formato do áudio")
    duration: float = Field(..., description="Duração do áudio em segundos")
    model: str = Field(..., description="Modelo usado")
    voice: str = Field(..., description="Voz usada")
    
class MusicGenerationResponse(BaseModel):
    """Resposta de geração de música."""
    audio: str = Field(..., description="Música gerada em formato base64")
    format: str = Field(..., description="Formato do áudio")
    duration: float = Field(..., description="Duração da música em segundos")
    model: str = Field(..., description="Modelo usado")
    prompt: str = Field(..., description="Prompt utilizado")
    
class AudioProcessingResponse(BaseModel):
    """Resposta de processamento de áudio."""
    audio: str = Field(..., description="Áudio processado em formato base64")
    format: str = Field(..., description="Formato do áudio")
    duration: float = Field(..., description="Duração do áudio em segundos")
    operations_applied: List[str] = Field(..., description="Operações aplicadas")
    
class AudioFeatureExtractionResponse(BaseModel):
    """Resposta de extração de features de áudio."""
    features: Dict[str, Union[List[float], float]] = Field(..., description="Features extraídas")
    duration: float = Field(..., description="Duração do áudio em segundos")
    
class AudioModelInfoResponse(BaseModel):
    """Informações sobre um modelo de áudio carregado."""
    model_name: str = Field(..., description="Nome do modelo")
    model_type: Optional[str] = Field(None, description="Tipo do modelo")
    status: str = Field(..., description="Status do modelo (carregado ou não)")
    device: str = Field(..., description="Dispositivo onde o modelo está carregado")
    memory_used_gb: Optional[float] = Field(None, description="Memória usada pelo modelo (GB)")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Informações adicionais")
