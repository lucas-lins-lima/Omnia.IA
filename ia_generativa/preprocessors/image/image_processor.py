"""
Image Processor - Funções para processamento e manipulação de imagens.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import logging
import io
import base64

logger = logging.getLogger(__name__)

def load_image_from_path(image_path: str) -> Image.Image:
    """
    Carrega uma imagem a partir de um caminho de arquivo.
    
    Args:
        image_path: Caminho para o arquivo de imagem
        
    Returns:
        Objeto PIL.Image carregado
    """
    try:
        image = Image.open(image_path).convert("RGB")
        logger.info(f"Imagem carregada: {image_path}, tamanho: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Erro ao carregar imagem de {image_path}: {str(e)}")
        raise
        
def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Carrega uma imagem a partir de bytes.
    
    Args:
        image_bytes: Bytes da imagem
        
    Returns:
        Objeto PIL.Image carregado
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info(f"Imagem carregada de bytes, tamanho: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Erro ao carregar imagem de bytes: {str(e)}")
        raise
        
def load_image_from_base64(base64_string: str) -> Image.Image:
    """
    Carrega uma imagem a partir de uma string base64.
    
    Args:
        base64_string: String base64 da imagem (pode incluir ou não o prefixo data:image/...)
        
    Returns:
        Objeto PIL.Image carregado
    """
    try:
        # Remover prefixo se presente
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        # Decodificar string base64
        image_bytes = base64.b64decode(base64_string)
        
        # Carregar imagem dos bytes
        image = load_image_from_bytes(image_bytes)
        
        return image
    except Exception as e:
        logger.error(f"Erro ao carregar imagem de base64: {str(e)}")
        raise
        
def resize_image(
    image: Image.Image,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect_ratio: bool = True,
    resize_method: str = "lanczos"
) -> Image.Image:
    """
    Redimensiona uma imagem para as dimensões especificadas.
    
    Args:
        image: Imagem a ser redimensionada
        width: Largura alvo (se None, calculada a partir da altura)
        height: Altura alvo (se None, calculada a partir da largura)
        keep_aspect_ratio: Se True, mantém a proporção original
        resize_method: Método de redimensionamento ('lanczos', 'bilinear', 'bicubic', 'nearest')
        
    Returns:
        Imagem redimensionada
    """
    if width is None and height is None:
        return image
        
    orig_width, orig_height = image.size
    
    # Determinar dimensões de destino
    if keep_aspect_ratio:
        if width is None:
            # Calcular largura a partir da altura
            aspect_ratio = orig_width / orig_height
            width = int(height * aspect_ratio)
        elif height is None:
            # Calcular altura a partir da largura
            aspect_ratio = orig_height / orig_width
            height = int(width * aspect_ratio)
        else:
            # Ambas as dimensões especificadas, ajustar a menor para manter proporção
            target_ratio = width / height
            orig_ratio = orig_width / orig_height
            
            if target_ratio > orig_ratio:
                # Limitar pela altura
                width = int(height * orig_ratio)
            else:
                # Limitar pela largura
                height = int(width / orig_ratio)
    
    # Mapear método de redimensionamento para filtro PIL
    resize_filters = {
        "lanczos": Image.LANCZOS,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "nearest": Image.NEAREST
    }
    
    filter_method = resize_filters.get(resize_method.lower(), Image.LANCZOS)
    
    # Redimensionar
    resized_image = image.resize((width, height), filter_method)
    logger.info(f"Imagem redimensionada de {(orig_width, orig_height)} para {(width, height)}")
    
    return resized_image
    
def pad_image_to_square(
    image: Image.Image,
    target_size: Optional[int] = None,
    pad_color: Union[int, Tuple[int, int, int]] = (0, 0, 0)
) -> Image.Image:
    """
    Adiciona padding a uma imagem para torná-la quadrada.
    
    Args:
        image: Imagem a ser processada
        target_size: Tamanho final desejado (se None, usa o maior lado da imagem)
        pad_color: Cor de fundo para o padding
        
    Returns:
        Imagem quadrada com padding
    """
    width, height = image.size
    
    # Determinar tamanho alvo
    if target_size is None:
        target_size = max(width, height)
        
    # Criar imagem quadrada com a cor de fundo
    result = Image.new("RGB", (target_size, target_size), pad_color)
    
    # Calcular posição para colar a imagem original (centralizada)
    paste_x = (target_size - width) // 2
    paste_y = (target_size - height) // 2
    
    # Colar imagem original
    result.paste(image, (paste_x, paste_y))
    
    logger.info(f"Imagem preenchida para {target_size}x{target_size}")
    return result
    
def crop_to_center(
    image: Image.Image,
    crop_width: int,
    crop_height: int
) -> Image.Image:
    """
    Recorta uma imagem para as dimensões especificadas, centralizada.
    
    Args:
        image: Imagem a ser recortada
        crop_width: Largura do recorte
        crop_height: Altura do recorte
        
    Returns:
        Imagem recortada
    """
    width, height = image.size
    
    # Verificar se o recorte é maior que a imagem
    if crop_width > width or crop_height > height:
        logger.warning("Dimensões de recorte maiores que a imagem original")
        return pad_image_to_square(image, max(crop_width, crop_height))
        
    # Calcular coordenadas de recorte
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    # Recortar
    cropped = image.crop((left, top, right, bottom))
    logger.info(f"Imagem recortada para {crop_width}x{crop_height}")
    
    return cropped
    
def normalize_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    make_square: bool = True,
    keep_aspect_ratio: bool = True,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Normaliza uma imagem para uso com modelos de IA.
    
    Args:
        image: Imagem a ser normalizada
        target_size: Tamanho alvo (largura, altura)
        make_square: Se True, garante que a imagem seja quadrada
        keep_aspect_ratio: Se True, mantém a proporção original
        pad_color: Cor de fundo para padding
        
    Returns:
        Imagem normalizada
    """
    # Determinar dimensões alvo
    target_width, target_height = target_size
    
    if make_square:
        # Garantir que largura e altura alvo são iguais
        target_size = max(target_width, target_height)
        target_width = target_height = target_size
        
        if keep_aspect_ratio:
            # Redimensionar para caber no quadrado, mantendo proporção
            width, height = image.size
            if width > height:
                new_height = int(height * target_size / width)
                new_width = target_size
            else:
                new_width = int(width * target_size / height)
                new_height = target_size
                
            image = resize_image(image, new_width, new_height)
            
            # Adicionar padding para tornar quadrado
            image = pad_image_to_square(image, target_size, pad_color)
        else:
            # Simplesmente redimensionar para o tamanho quadrado
            image = resize_image(image, target_size, target_size, keep_aspect_ratio=False)
    else:
        # Não é necessário ser quadrado, apenas redimensionar
        image = resize_image(image, target_width, target_height, keep_aspect_ratio)
        
    return image
    
def convert_to_base64(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 90
) -> str:
    """
    Converte uma imagem PIL para uma string base64.
    
    Args:
        image: Imagem a ser convertida
        format: Formato de saída (JPEG, PNG, etc.)
        quality: Qualidade da imagem (para formatos com compressão)
        
    Returns:
        String base64 da imagem
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"
    
def create_inpainting_mask(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    feather: int = 0
) -> Image.Image:
    """
    Cria uma máscara para inpainting a partir de coordenadas e dimensões.
    
    Args:
        image: Imagem de referência para dimensões
        x, y: Coordenadas do canto superior esquerdo da área a ser preenchida
        width, height: Dimensões da área a ser preenchida
        feather: Quantidade de suavização nas bordas (em pixels)
        
    Returns:
        Imagem em escala de cinza (máscara) com a área selecionada em branco
    """
    img_width, img_height = image.size
    
    # Criar máscara preta (sem área a ser preenchida)
    mask = Image.new("L", (img_width, img_height), 0)
    
    # Desenhar retângulo branco na área a ser preenchida
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([(x, y), (x + width, y + height)], fill=255)
    
    # Aplicar suavização nas bordas
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
        
    return mask
