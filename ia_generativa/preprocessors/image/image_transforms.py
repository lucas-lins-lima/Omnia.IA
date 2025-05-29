"""
Image Transforms - Transformações específicas para preparar imagens para modelos de IA.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger(__name__)

def adjust_brightness(
    image: Image.Image, 
    factor: float
) -> Image.Image:
    """
    Ajusta o brilho de uma imagem.
    
    Args:
        image: Imagem a ser ajustada
        factor: Fator de ajuste (0 = preto, 1 = original, >1 = mais brilhante)
        
    Returns:
        Imagem com brilho ajustado
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)
    
def adjust_contrast(
    image: Image.Image, 
    factor: float
) -> Image.Image:
    """
    Ajusta o contraste de uma imagem.
    
    Args:
        image: Imagem a ser ajustada
        factor: Fator de ajuste (0 = cinza, 1 = original, >1 = mais contraste)
        
    Returns:
        Imagem com contraste ajustado
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)
    
def adjust_saturation(
    image: Image.Image, 
    factor: float
) -> Image.Image:
    """
    Ajusta a saturação de uma imagem.
    
    Args:
        image: Imagem a ser ajustada
        factor: Fator de ajuste (0 = preto e branco, 1 = original, >1 = mais saturação)
        
    Returns:
        Imagem com saturação ajustada
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)
    
def adjust_sharpness(
    image: Image.Image, 
    factor: float
) -> Image.Image:
    """
    Ajusta a nitidez de uma imagem.
    
    Args:
        image: Imagem a ser ajustada
        factor: Fator de ajuste (0 = desfocado, 1 = original, >1 = mais nítido)
        
    Returns:
        Imagem com nitidez ajustada
    """
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)
    
def apply_blur(
    image: Image.Image, 
    radius: float
) -> Image.Image:
    """
    Aplica desfoque gaussiano a uma imagem.
    
    Args:
        image: Imagem a ser desfocada
        radius: Raio do desfoque (maior = mais desfocado)
        
    Returns:
        Imagem desfocada
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
def apply_edge_enhance(
    image: Image.Image,
    factor: float = 1.0
) -> Image.Image:
    """
    Aplica realce de bordas a uma imagem.
    
    Args:
        image: Imagem a ser processada
        factor: Intensidade do efeito
        
    Returns:
        Imagem com bordas realçadas
    """
    if factor <= 0:
        return image
        
    if factor == 1.0:
        return image.filter(ImageFilter.EDGE_ENHANCE)
    elif factor > 1.0:
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    else:
        # Para fatores entre 0 e 1, misturar com a imagem original
        enhanced = image.filter(ImageFilter.EDGE_ENHANCE)
        return Image.blend(image, enhanced, factor)
        
def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Converte uma imagem para escala de cinza.
    
    Args:
        image: Imagem a ser convertida
        
    Returns:
        Imagem em escala de cinza
    """
    return image.convert("L").convert("RGB")
    
def apply_depth_map(
    image: Image.Image,
    depth_model_name: str = "Intel/dpt-large"
) -> Image.Image:
    """
    Gera um mapa de profundidade a partir de uma imagem.
    
    Args:
        image: Imagem de entrada
        depth_model_name: Nome do modelo de estimativa de profundidade
        
    Returns:
        Mapa de profundidade como imagem em escala de cinza
    """
    try:
        from transformers import pipeline
        
        # Carregar pipeline de estimativa de profundidade
        depth_estimator = pipeline("depth-estimation", model=depth_model_name)
        
        # Gerar mapa de profundidade
        result = depth_estimator(image)
        depth_map = result["depth"]
        
        # Normalizar e converter para PIL
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)
        depth_image = Image.fromarray(depth_map)
        
        return depth_image
    except Exception as e:
        logger.error(f"Erro ao gerar mapa de profundidade: {str(e)}")
        # Fallback para uma versão simples
        return convert_to_grayscale(image)
        
def apply_canny_edges(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200
) -> Image.Image:
    """
    Aplica detector de bordas Canny a uma imagem.
    
    Args:
        image: Imagem de entrada
        low_threshold: Limite inferior para detecção de bordas
        high_threshold: Limite superior para detecção de bordas
        
    Returns:
        Imagem com bordas destacadas
    """
    try:
        import cv2
        import numpy as np
        
        # Converter para formato numpy/OpenCV
        img_array = np.array(image)
        
        # Converter para escala de cinza se necessário
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Aplicar Canny
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Converter de volta para PIL
        return Image.fromarray(edges)
    except Exception as e:
        logger.error(f"Erro ao aplicar detector Canny: {str(e)}")
        # Fallback para um filtro de bordas simples
        return image.filter(ImageFilter.FIND_EDGES)
    
def apply_controlnet_preprocessor(
    image: Image.Image,
    preprocessor_type: str,
    **kwargs
) -> Image.Image:
    """
    Aplica um pré-processador específico para uso com ControlNet.
    
    Args:
        image: Imagem de entrada
        preprocessor_type: Tipo de pré-processador ('canny', 'depth', 'normal', 'pose', etc.)
        **kwargs: Argumentos específicos para o pré-processador
        
    Returns:
        Imagem processada pronta para uso com ControlNet
    """
    preprocessor_map = {
        "canny": apply_canny_edges,
        "depth": apply_depth_map,
        "grayscale": convert_to_grayscale,
        "blur": apply_blur,
        "edge_enhance": apply_edge_enhance
    }
    
    if preprocessor_type not in preprocessor_map:
        logger.warning(f"Pré-processador não implementado: {preprocessor_type}")
        return image
        
    preprocessor_func = preprocessor_map[preprocessor_type]
    return preprocessor_func(image, **kwargs)
