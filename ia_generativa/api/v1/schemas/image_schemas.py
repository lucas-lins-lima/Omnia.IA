"""
Schemas para validação de dados de imagem na API.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Dict, Optional, Union, Any, Tuple
from enum import Enum
import base64
import re

class ImageFormat(str, Enum):
    """Formatos de imagem suportados."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    
class ImageSize(BaseModel):
    """Tamanho da imagem."""
    width: int = Field(..., description="Largura da imagem em pixels", ge=64, le=2048)
    height: int = Field(..., description="Altura da imagem em pixels", ge=64, le=2048)
    
    @validator('width', 'height')
    def must_be_multiple_of_8(cls, v):
        """Valida que as dimensões são múltiplos de 8."""
        if v % 8 != 0:
            raise ValueError("Dimensões devem ser múltiplos de 8")
        return v
    
class GenerationModel(str, Enum):
    """Modelos de geração de imagem suportados."""
    STABLE_DIFFUSION_1_5 = "runwayml/stable-diffusion-v1-5"
    STABLE_DIFFUSION_2_1 = "stabilityai/stable-diffusion-2-1"
    SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
    KANDINSKY = "kandinsky-community/kandinsky-2-1"
    
class TextToImageRequest(BaseModel):
    """Requisição para geração de imagem a partir de texto."""
    prompt: str = Field(..., description="Descrição textual da imagem desejada")
    negative_prompt: Optional[str] = Field(None, description="Elementos a serem evitados na imagem")
    model: GenerationModel = Field(GenerationModel.STABLE_DIFFUSION_1_5, description="Modelo a ser usado")
    size: ImageSize = Field(default_factory=lambda: ImageSize(width=512, height=512))
    num_inference_steps: int = Field(25, description="Número de passos de inferência", ge=1, le=150)
    guidance_scale: float = Field(7.5, description="Escala de orientação do prompt", ge=1.0, le=20.0)
    num_images: int = Field(1, description="Número de imagens a serem geradas", ge=1, le=4)
    seed: Optional[int] = Field(None, description="Semente para geração determinística")
    output_format: ImageFormat = Field(ImageFormat.PNG, description="Formato da imagem de saída")
    
    @validator('prompt')
    def prompt_not_empty(cls, v):
        """Valida que o prompt não está vazio."""
        if not v.strip():
            raise ValueError("Prompt não pode estar vazio")
        return v
    
class ImageToImageRequest(BaseModel):
    """Requisição para transformação de imagem existente."""
    prompt: str = Field(..., description="Descrição textual da transformação desejada")
    negative_prompt: Optional[str] = Field(None, description="Elementos a serem evitados na imagem")
    model: GenerationModel = Field(GenerationModel.STABLE_DIFFUSION_1_5, description="Modelo a ser usado")
    image: str = Field(..., description="Imagem base em formato base64")
    strength: float = Field(0.8, description="Intensidade da transformação", ge=0.0, le=1.0)
    num_inference_steps: int = Field(25, description="Número de passos de inferência", ge=1, le=150)
    guidance_scale: float = Field(7.5, description="Escala de orientação do prompt", ge=1.0, le=20.0)
    num_images: int = Field(1, description="Número de imagens a serem geradas", ge=1, le=4)
    seed: Optional[int] = Field(None, description="Semente para geração determinística")
    resize_to_original: bool = Field(True, description="Se True, mantém as dimensões originais")
    output_format: ImageFormat = Field(ImageFormat.PNG, description="Formato da imagem de saída")
    
    @validator('image')
    def validate_base64_image(cls, v):
        """Valida que a string é um base64 válido de imagem."""
        # Verificar se começa com data:image ou é base64 puro
        if not (v.startswith('data:image/') or re.match(r'^[A-Za-z0-9+/=]+$', v)):
            raise ValueError("Formato de imagem base64 inválido")
        return v
    
class InpaintingRequest(BaseModel):
    """Requisição para inpainting (preenchimento de áreas) em uma imagem."""
    prompt: str = Field(..., description="Descrição textual do conteúdo a ser gerado")
    negative_prompt: Optional[str] = Field(None, description="Elementos a serem evitados na imagem")
    model: GenerationModel = Field(GenerationModel.STABLE_DIFFUSION_1_5, description="Modelo a ser usado")
    image: str = Field(..., description="Imagem base em formato base64")
    mask: str = Field(..., description="Máscara indicando áreas a serem preenchidas (base64)")
    num_inference_steps: int = Field(25, description="Número de passos de inferência", ge=1, le=150)
    guidance_scale: float = Field(7.5, description="Escala de orientação do prompt", ge=1.0, le=20.0)
    num_images: int = Field(1, description="Número de imagens a serem geradas", ge=1, le=4)
    seed: Optional[int] = Field(None, description="Semente para geração determinística")
    output_format: ImageFormat = Field(ImageFormat.PNG, description="Formato da imagem de saída")
    
class ImageCaptionRequest(BaseModel):
    """Requisição para gerar legenda para uma imagem."""
    image: str = Field(..., description="Imagem em formato base64")
    max_length: int = Field(30, description="Comprimento máximo da legenda em tokens", ge=10, le=100)
    conditional_prompt: Optional[str] = Field(None, description="Prompt condicional para guiar a geração")
    
class ImageClassificationRequest(BaseModel):
    """Requisição para classificar uma imagem."""
    image: str = Field(..., description="Imagem em formato base64")
    candidate_labels: List[str] = Field(..., description="Lista de categorias candidatas")
    
    @validator('candidate_labels')
    def validate_candidate_labels(cls, v):
        """Valida que há pelo menos uma categoria candidata."""
        if not v:
            raise ValueError("É necessário fornecer pelo menos uma categoria candidata")
        return v
    
class ImageSimilarityRequest(BaseModel):
    """Requisição para comparar imagem com textos."""
    image: str = Field(..., description="Imagem em formato base64")
    texts: List[str] = Field(..., description="Lista de textos para comparar")
    
    @validator('texts')
    def validate_texts(cls, v):
        """Valida que há pelo menos um texto para comparar."""
        if not v:
            raise ValueError("É necessário fornecer pelo menos um texto para comparar")
        return v
    
class GeneratedImageResponse(BaseModel):
    """Resposta contendo imagem(ns) gerada(s)."""
    images: List[str] = Field(..., description="Lista de imagens geradas em formato base64")
    seed: int = Field(..., description="Semente usada para geração")
    prompt: str = Field(..., description="Prompt utilizado")
    generation_time: float = Field(..., description="Tempo de geração em segundos")
    model: str = Field(..., description="Modelo utilizado")
    
class ImageCaptionResponse(BaseModel):
    """Resposta contendo legenda gerada para uma imagem."""
    caption: str = Field(..., description="Legenda gerada")
    
class ImageClassificationResponse(BaseModel):
    """Resposta contendo resultados de classificação de imagem."""
    classifications: Dict[str, float] = Field(..., description="Mapeamento de categorias para scores")
    
class ImageSimilarityResponse(BaseModel):
    """Resposta contendo scores de similaridade entre imagem e textos."""
    similarities: Dict[str, float] = Field(..., description="Mapeamento de textos para scores de similaridade")
    
class ModelInfoResponse(BaseModel):
    """Informações sobre um modelo carregado."""
    model_name: str = Field(..., description="Nome do modelo")
    device: str = Field(..., description="Dispositivo onde o modelo está carregado")
    dtype: str = Field(..., description="Tipo de dados do modelo")
    pipelines_loaded: List[str] = Field(..., description="Pipelines carregados")
    memory_used_gb: Optional[float] = Field(None, description="Memória usada pelo modelo (GB)")
    optimizations: Optional[Dict[str, Any]] = Field(None, description="Otimizações ativadas")
