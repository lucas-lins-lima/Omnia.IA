"""
Schemas para validação de dados de vídeo na API.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Dict, Optional, Union, Any, Tuple
from enum import Enum
import base64
import re

class VideoFormat(str, Enum):
    """Formatos de vídeo suportados."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"
    
class VideoOperation(str, Enum):
    """Operações disponíveis para processamento de vídeo."""
    RESIZE = "resize"
    FILTER = "filter"
    ADJUST = "adjust"
    TEXT = "text"
    EFFECT = "effect"
    
class FilterType(str, Enum):
    """Tipos de filtros de cor disponíveis."""
    GRAYSCALE = "grayscale"
    SEPIA = "sepia"
    NEGATIVE = "negative"
    BLUE = "blue"
    WARM = "warm"
    COOL = "cool"
    
class EffectType(str, Enum):
    """Tipos de efeitos disponíveis."""
    BLUR = "blur"
    EDGE = "edge"
    EMBOSS = "emboss"
    SHARPEN = "sharpen"
    PENCIL = "pencil"
    CARTOON = "cartoon"
    
class OperationParameters(BaseModel):
    """Parâmetros para uma operação de vídeo."""
    type: VideoOperation = Field(..., description="Tipo de operação")
    
    # Parâmetros para resize
    width: Optional[int] = Field(None, description="Nova largura para redimensionamento")
    height: Optional[int] = Field(None, description="Nova altura para redimensionamento")
    scale: Optional[float] = Field(None, description="Fator de escala para redimensionamento")
    
    # Parâmetros para filter
    filter_type: Optional[FilterType] = Field(None, description="Tipo de filtro de cor")
    intensity: Optional[float] = Field(0.8, description="Intensidade do filtro (0.0 a 1.0)")
    
    # Parâmetros para adjust
    brightness: Optional[float] = Field(0.0, description="Ajuste de brilho (-1.0 a 1.0)")
    contrast: Optional[float] = Field(0.0, description="Ajuste de contraste (-1.0 a 1.0)")
    saturation: Optional[float] = Field(0.0, description="Ajuste de saturação (-1.0 a 1.0)")
    
    # Parâmetros para text
    text: Optional[str] = Field(None, description="Texto a ser adicionado")
    position: Optional[Tuple[int, int]] = Field(None, description="Posição (x, y) do texto")
    font_size: Optional[float] = Field(1.0, description="Tamanho da fonte")
    color: Optional[Tuple[int, int, int]] = Field((255, 255, 255), description="Cor do texto (BGR)")
    background: Optional[bool] = Field(False, description="Se True, adiciona fundo atrás do texto")
    
    # Parâmetros para effect
    effect_type: Optional[EffectType] = Field(None, description="Tipo de efeito")
    params: Optional[Dict[str, Any]] = Field({}, description="Parâmetros adicionais para o efeito")
    
    @validator('intensity', 'brightness', 'contrast', 'saturation')
    def validate_range(cls, v, values):
        """Valida que os valores estão no intervalo correto."""
        if v is not None:
            if values.get('type') == VideoOperation.FILTER and 'intensity' in values:
                if v < 0.0 or v > 1.0:
                    raise ValueError("Intensity deve estar entre 0.0 e 1.0")
            if values.get('type') == VideoOperation.ADJUST:
                if v < -1.0 or v > 1.0:
                    raise ValueError("Brightness/contrast/saturation devem estar entre -1.0 e 1.0")
        return v
    
    @validator('text')
    def validate_text(cls, v, values):
        """Valida que o texto não está vazio para operação de texto."""
        if values.get('type') == VideoOperation.TEXT and (v is None or not v.strip()):
            raise ValueError("Texto não pode estar vazio para operação de texto")
        return v
    
class VideoProcessRequest(BaseModel):
    """Requisição para processamento de vídeo."""
    video: str = Field(..., description="Vídeo em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo")
    operations: List[OperationParameters] = Field(..., description="Lista de operações a serem aplicadas")
    output_format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo de saída")
    start_time: Optional[float] = Field(None, description="Tempo inicial para processamento (segundos)")
    end_time: Optional[float] = Field(None, description="Tempo final para processamento (segundos)")
    sample_rate: Optional[int] = Field(None, description="Taxa de amostragem (processar 1 a cada N frames)")
    preserve_audio: bool = Field(True, description="Se True, preserva o áudio do vídeo original")
    
    @validator('video')
    def validate_base64_video(cls, v):
        """Valida que a string é um base64 válido de vídeo."""
        # Verificar se começa com data:video ou é base64 puro
        if not (v.startswith('data:video/') or re.match(r'^[A-Za-z0-9+/=]+$', v)):
            raise ValueError("Formato de vídeo base64 inválido")
        return v
    
    @validator('operations')
    def validate_operations(cls, v):
        """Valida que há pelo menos uma operação."""
        if not v:
            raise ValueError("É necessário fornecer pelo menos uma operação")
        return v
    
class VideoTrimRequest(BaseModel):
    """Requisição para recorte de vídeo."""
    video: str = Field(..., description="Vídeo em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo")
    start_time: float = Field(..., description="Tempo inicial do recorte (segundos)")
    end_time: float = Field(..., description="Tempo final do recorte (segundos)")
    output_format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo de saída")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        """Valida que end_time é maior que start_time."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError("end_time deve ser maior que start_time")
        return v
    
class VideoConcatenateRequest(BaseModel):
    """Requisição para concatenação de vídeos."""
    videos: List[str] = Field(..., description="Lista de vídeos em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato dos vídeos")
    output_format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo de saída")
    crossfade_duration: Optional[float] = Field(None, description="Duração da transição em segundos")
    
    @validator('videos')
    def validate_videos(cls, v):
        """Valida que há pelo menos dois vídeos."""
        if len(v) < 2:
            raise ValueError("É necessário fornecer pelo menos dois vídeos")
        return v
    
class VideoAddTextRequest(BaseModel):
    """Requisição para adicionar texto a um vídeo."""
    video: str = Field(..., description="Vídeo em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo")
    text: str = Field(..., description="Texto a ser adicionado")
    position: str = Field("bottom", description="Posição do texto (top, center, bottom)")
    font_size: float = Field(1.0, description="Tamanho da fonte")
    color: Tuple[int, int, int] = Field((255, 255, 255), description="Cor do texto (BGR)")
    background: bool = Field(True, description="Se True, adiciona fundo atrás do texto")
    start_time: Optional[float] = Field(None, description="Tempo inicial para mostrar o texto (segundos)")
    end_time: Optional[float] = Field(None, description="Tempo final para mostrar o texto (segundos)")
    output_format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo de saída")
    
    @validator('text')
    def validate_text(cls, v):
        """Valida que o texto não está vazio."""
        if not v.strip():
            raise ValueError("Texto não pode estar vazio")
        return v
    
class VideoSlideshowRequest(BaseModel):
    """Requisição para criação de slideshow."""
    images: List[str] = Field(..., description="Lista de imagens em formato base64")
    fps: float = Field(1.0, description="Frames por segundo")
    transition_frames: int = Field(30, description="Número de frames para transição")
    transition_type: str = Field("fade", description="Tipo de transição (fade, wipe)")
    audio: Optional[str] = Field(None, description="Áudio opcional em formato base64")
    duration_per_image: Optional[float] = Field(None, description="Duração em segundos para cada imagem")
    text_overlay: Optional[List[str]] = Field(None, description="Lista de textos para sobrepor em cada imagem")
    output_format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo de saída")
    
    @validator('images')
    def validate_images(cls, v):
        """Valida que há pelo menos uma imagem."""
        if not v:
            raise ValueError("É necessário fornecer pelo menos uma imagem")
        return v
    
class VideoAnalysisRequest(BaseModel):
    """Requisição para análise de vídeo."""
    video: str = Field(..., description="Vídeo em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo")
    analyze_objects: bool = Field(True, description="Se True, detecta objetos")
    classify_content: bool = Field(True, description="Se True, classifica o conteúdo")
    generate_caption: bool = Field(True, description="Se True, gera descrição")
    frame_interval: Optional[int] = Field(10, description="Intervalo entre frames")
    
class VideoMontageRequest(BaseModel):
    """Requisição para criação de montagem de vídeos."""
        videos: List[str] = Field(..., description="Lista de vídeos em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato dos vídeos")
    layout: str = Field("grid", description="Tipo de layout (grid, horizontal, vertical)")
    grid_size: Optional[Tuple[int, int]] = Field(None, description="Tamanho da grade (rows, cols)")
    output_size: Optional[Tuple[int, int]] = Field(None, description="Tamanho do vídeo de saída (width, height)")
    include_audio: bool = Field(True, description="Se True, inclui áudio do primeiro vídeo")
    output_format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo de saída")
    
    @validator('videos')
    def validate_videos(cls, v):
        """Valida que há pelo menos um vídeo."""
        if not v:
            raise ValueError("É necessário fornecer pelo menos um vídeo")
        return v
    
    @validator('layout')
    def validate_layout(cls, v):
        """Valida que o layout é válido."""
        if v not in ["grid", "horizontal", "vertical"]:
            raise ValueError("Layout deve ser 'grid', 'horizontal' ou 'vertical'")
        return v
    
class VideoExtractAudioRequest(BaseModel):
    """Requisição para extração de áudio de um vídeo."""
    video: str = Field(..., description="Vídeo em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo")
    output_format: str = Field("mp3", description="Formato do áudio de saída")
    start_time: Optional[float] = Field(None, description="Tempo inicial para extração (segundos)")
    end_time: Optional[float] = Field(None, description="Tempo final para extração (segundos)")
    
    @validator('output_format')
    def validate_output_format(cls, v):
        """Valida que o formato de saída é válido."""
        if v.lower() not in ["mp3", "wav", "ogg", "aac"]:
            raise ValueError("Formato de saída deve ser 'mp3', 'wav', 'ogg' ou 'aac'")
        return v.lower()
    
class VideoExtractSceneRequest(BaseModel):
    """Requisição para detecção de cenas em um vídeo."""
    video: str = Field(..., description="Vídeo em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo")
    threshold: float = Field(30.0, description="Limiar de diferença para detectar transição")
    min_scene_length: float = Field(1.0, description="Duração mínima de uma cena em segundos")
    
class VideoSpeedChangeRequest(BaseModel):
    """Requisição para alteração de velocidade de um vídeo."""
    video: str = Field(..., description="Vídeo em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo")
    speed_factor: float = Field(..., description="Fator de velocidade (>1 para acelerar, <1 para desacelerar)")
    preserve_audio_pitch: bool = Field(True, description="Se True, preserva o tom do áudio")
    output_format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo de saída")
    
    @validator('speed_factor')
    def validate_speed_factor(cls, v):
        """Valida que o fator de velocidade é positivo."""
        if v <= 0:
            raise ValueError("speed_factor deve ser positivo")
        return v
    
class VideoTimelapseRequest(BaseModel):
    """Requisição para criação de timelapse."""
    video: str = Field(..., description="Vídeo em formato base64")
    format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo")
    speed_factor: float = Field(10.0, description="Fator de aceleração do timelapse")
    frame_interval: int = Field(1, description="Intervalo de frames a considerar")
    output_format: VideoFormat = Field(VideoFormat.MP4, description="Formato do vídeo de saída")
    
    @validator('speed_factor')
    def validate_speed_factor(cls, v):
        """Valida que o fator de velocidade é positivo."""
        if v <= 0:
            raise ValueError("speed_factor deve ser positivo")
        return v
    
    @validator('frame_interval')
    def validate_frame_interval(cls, v):
        """Valida que o intervalo de frames é positivo."""
        if v <= 0:
            raise ValueError("frame_interval deve ser positivo")
        return v
    
class VideoResponse(BaseModel):
    """Resposta contendo um vídeo processado."""
    video: str = Field(..., description="Vídeo processado em formato base64")
    format: str = Field(..., description="Formato do vídeo")
    duration: float = Field(..., description="Duração do vídeo em segundos")
    width: int = Field(..., description="Largura do vídeo em pixels")
    height: int = Field(..., description="Altura do vídeo em pixels")
    fps: float = Field(..., description="Frames por segundo")
    processing_time: float = Field(..., description="Tempo de processamento em segundos")
    
class AudioResponse(BaseModel):
    """Resposta contendo áudio extraído de um vídeo."""
    audio: str = Field(..., description="Áudio extraído em formato base64")
    format: str = Field(..., description="Formato do áudio")
    duration: float = Field(..., description="Duração do áudio em segundos")
    
class VideoAnalysisResponse(BaseModel):
    """Resposta contendo análise de vídeo."""
    video_info: Dict[str, Any] = Field(..., description="Informações básicas do vídeo")
    content_classification: Optional[List[Dict[str, Any]]] = Field(None, description="Classificação de conteúdo")
    object_detection: Optional[Dict[str, Any]] = Field(None, description="Detecção de objetos")
    video_description: Optional[str] = Field(None, description="Descrição geral do vídeo")
    processing_time: float = Field(..., description="Tempo de processamento em segundos")
    
class SceneDetectionResponse(BaseModel):
    """Resposta contendo transições de cena detectadas."""
    scenes: List[Dict[str, Any]] = Field(..., description="Lista de cenas detectadas")
    duration: float = Field(..., description="Duração do vídeo em segundos")
    total_scenes: int = Field(..., description="Número total de cenas detectadas")
