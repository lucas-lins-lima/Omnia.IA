"""
Frame Processor - Funções para processamento de frames de vídeo.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

def resize_frame(
    frame: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
    interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """
    Redimensiona um frame.
    
    Args:
        frame: Frame a ser redimensionado
        width: Nova largura (None para manter proporção)
        height: Nova altura (None para manter proporção)
        scale: Fator de escala (alternativa a width/height)
        interpolation: Método de interpolação
        
    Returns:
        Frame redimensionado
    """
    try:
        h, w = frame.shape[:2]
        
        if scale is not None:
            width = int(w * scale)
            height = int(h * scale)
        elif width is not None and height is None:
            # Manter proporção
            height = int(h * (width / w))
        elif height is not None and width is None:
            # Manter proporção
            width = int(w * (height / h))
            
        if width is not None and height is not None:
            resized = cv2.resize(frame, (width, height), interpolation=interpolation)
            return resized
        else:
            return frame
            
    except Exception as e:
        logger.error(f"Erro ao redimensionar frame: {str(e)}")
        return frame

def apply_color_filter(
    frame: np.ndarray,
    filter_type: str,
    intensity: float = 1.0
) -> np.ndarray:
    """
    Aplica um filtro de cor a um frame.
    
    Args:
        frame: Frame para aplicar o filtro
        filter_type: Tipo de filtro (grayscale, sepia, etc.)
        intensity: Intensidade do filtro (0.0 a 1.0)
        
    Returns:
        Frame com filtro aplicado
    """
    try:
        result = frame.copy()
        
        if filter_type == "grayscale":
            # Converter para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Misturar com o original pela intensidade
            if intensity < 1.0:
                result = cv2.addWeighted(gray_3c, intensity, frame, 1.0 - intensity, 0)
            else:
                result = gray_3c
                
        elif filter_type == "sepia":
            # Matriz de transformação para sepia
            sepia_kernel = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ])
            
            # Aplicar transformação
            sepia = cv2.transform(frame, sepia_kernel)
            
            # Misturar com o original pela intensidade
            if intensity < 1.0:
                result = cv2.addWeighted(sepia, intensity, frame, 1.0 - intensity, 0)
            else:
                result = sepia
                
        elif filter_type == "negative":
            # Inverter cores
            negative = 255 - frame
            
            # Misturar com o original pela intensidade
            if intensity < 1.0:
                result = cv2.addWeighted(negative, intensity, frame, 1.0 - intensity, 0)
            else:
                result = negative
                
        elif filter_type == "blue":
            # Aumentar componente azul
            blue = frame.copy()
            blue[:, :, 0] = np.clip(blue[:, :, 0] * (1 + intensity), 0, 255).astype(np.uint8)
            
            # Misturar com o original
            result = cv2.addWeighted(blue, intensity, frame, 1.0 - intensity, 0)
            
        elif filter_type == "warm":
            # Aumentar componentes vermelho e verde
            warm = frame.copy()
            warm[:, :, 1] = np.clip(warm[:, :, 1] * (1 + 0.5 * intensity), 0, 255).astype(np.uint8)
            warm[:, :, 2] = np.clip(warm[:, :, 2] * (1 + intensity), 0, 255).astype(np.uint8)
            
            # Misturar com o original
            result = cv2.addWeighted(warm, intensity, frame, 1.0 - intensity, 0)
            
        elif filter_type == "cool":
            # Aumentar componente azul e diminuir vermelho
            cool = frame.copy()
            cool[:, :, 0] = np.clip(cool[:, :, 0] * (1 + intensity), 0, 255).astype(np.uint8)
            cool[:, :, 2] = np.clip(cool[:, :, 2] * (1 - 0.5 * intensity), 0, 255).astype(np.uint8)
            
            # Misturar com o original
            result = cv2.addWeighted(cool, intensity, frame, 1.0 - intensity, 0)
            
        else:
            logger.warning(f"Filtro desconhecido: {filter_type}")
            
        return result
        
    except Exception as e:
        logger.error(f"Erro ao aplicar filtro de cor: {str(e)}")
        return frame

def adjust_frame(
    frame: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0
) -> np.ndarray:
    """
    Ajusta propriedades de um frame.
    
    Args:
        frame: Frame para ajustar
        brightness: Ajuste de brilho (-1.0 a 1.0)
        contrast: Ajuste de contraste (-1.0 a 1.0)
        saturation: Ajuste de saturação (-1.0 a 1.0)
        
    Returns:
        Frame ajustado
    """
    try:
        result = frame.copy()
        
        # Ajustar brilho
        if brightness != 0:
            if brightness > 0:
                # Aumentar brilho
                brightness_factor = 1 + brightness
                result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
            else:
                # Diminuir brilho
                brightness_factor = 1 + brightness
                result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
                
        # Ajustar contraste
        if contrast != 0:
            if contrast > 0:
                # Aumentar contraste
                contrast_factor = 1 + contrast
                mean = np.mean(result)
                result = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=(1-contrast_factor)*mean)
            else:
                # Diminuir contraste
                contrast_factor = 1 + contrast
                mean = np.mean(result)
                result = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=(1-contrast_factor)*mean)
                
        # Ajustar saturação
        if saturation != 0:
            # Converter para HSV
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Ajustar canal S
            if saturation > 0:
                hsv[:, :, 1] = hsv[:, :, 1] * (1 + saturation)
            else:
                hsv[:, :, 1] = hsv[:, :, 1] * (1 + saturation)
                
            # Limitar valores
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Converter de volta para BGR
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
        return result
        
    except Exception as e:
        logger.error(f"Erro ao ajustar frame: {str(e)}")
        return frame

def add_text_to_frame(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    background: bool = False,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_opacity: float = 0.5
) -> np.ndarray:
    """
    Adiciona texto a um frame.
    
    Args:
        frame: Frame para adicionar texto
        text: Texto a ser adicionado
        position: Posição (x, y) do texto
        font_size: Tamanho da fonte
        color: Cor do texto (BGR)
        thickness: Espessura do texto
        background: Se True, adiciona fundo atrás do texto
        bg_color: Cor do fundo
        bg_opacity: Opacidade do fundo
        
    Returns:
        Frame com texto adicionado
    """
    try:
        result = frame.copy()
        
        # Definir fonte
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calcular tamanho do texto
        text_size, _ = cv2.getTextSize(text, font, font_size, thickness)
        text_w, text_h = text_size
        
        # Adicionar fundo
        if background:
            x, y = position
            padding = 10
            
            # Criar overlay para o fundo
            overlay = result.copy()
            cv2.rectangle(
                overlay, 
                (x - padding, y - text_h - padding), 
                (x + text_w + padding, y + padding), 
                bg_color, 
                -1
            )
            
            # Misturar com a imagem original
            cv2.addWeighted(overlay, bg_opacity, result, 1 - bg_opacity, 0, result)
            
        # Adicionar texto
        cv2.putText(
            result, 
            text, 
            position, 
            font, 
            font_size, 
            color, 
            thickness
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Erro ao adicionar texto ao frame: {str(e)}")
        return frame

def overlay_image_on_frame(
    frame: np.ndarray,
    overlay: np.ndarray,
    position: Tuple[int, int],
    scale: Optional[float] = None,
    opacity: float = 1.0
) -> np.ndarray:
    """
    Sobrepõe uma imagem em um frame.
    
    Args:
        frame: Frame base
        overlay: Imagem para sobrepor
        position: Posição (x, y) para colocar a imagem
        scale: Fator de escala para a imagem (opcional)
        opacity: Opacidade da imagem sobreposta
        
    Returns:
        Frame com imagem sobreposta
    """
    try:
        result = frame.copy()
        
        # Redimensionar overlay se necessário
        if scale is not None and scale != 1.0:
            overlay_height, overlay_width = overlay.shape[:2]
            overlay = cv2.resize(
                overlay, 
                (int(overlay_width * scale), int(overlay_height * scale))
            )
            
        # Obter dimensões
        overlay_height, overlay_width = overlay.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        
        # Calcular região para overlay
        x, y = position
        
        # Verificar se está dentro dos limites
        if x >= frame_width or y >= frame_height:
            return result
            
        # Calcular largura e altura efetivas
        w = min(overlay_width, frame_width - x)
        h = min(overlay_height, frame_height - y)
        
        # Verificar se overlay tem canal alpha
        if overlay.shape[2] == 4:
            # Extrair canal alpha
            alpha = overlay[:h, :w, 3] / 255.0 * opacity
            alpha = np.dstack([alpha, alpha, alpha])
            
            # Extrair canais BGR
            overlay_bgr = overlay[:h, :w, :3]
            
            # Combinar imagens
            result[y:y+h, x:x+w] = result[y:y+h, x:x+w] * (1 - alpha) + overlay_bgr * alpha
        else:
            # Sem canal alpha, usar opacidade global
            result[y:y+h, x:x+w] = cv2.addWeighted(
                result[y:y+h, x:x+w], 
                1 - opacity,
                overlay[:h, :w], 
                opacity, 
                0
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Erro ao sobrepor imagem: {str(e)}")
        return frame

def apply_effect_to_frame(
    frame: np.ndarray,
    effect: str,
    params: Dict[str, Any] = {}
) -> np.ndarray:
    """
    Aplica um efeito especial a um frame.
    
    Args:
        frame: Frame para aplicar o efeito
        effect: Nome do efeito
        params: Parâmetros adicionais para o efeito
        
    Returns:
        Frame com efeito aplicado
    """
    try:
        result = frame.copy()
        
        if effect == "blur":
            # Aplicar desfoque gaussiano
            kernel_size = params.get("kernel_size", 15)
            
            # Garantir que o kernel é ímpar
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
            
        elif effect == "edge":
            # Detectar bordas com Canny
            threshold1 = params.get("threshold1", 100)
            threshold2 = params.get("threshold2", 200)
            
            edges = cv2.Canny(result, threshold1, threshold2)
            
            # Converter para 3 canais
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        elif effect == "emboss":
            # Kernel para efeito de relevo
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            
            # Aplicar filtro
            result = cv2.filter2D(result, -1, kernel)
            
        elif effect == "sharpen":
            # Kernel para nitidez
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            
            # Aplicar filtro
            result = cv2.filter2D(result, -1, kernel)
            
        elif effect == "pencil":
            # Efeito de desenho a lápis
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            
            # Inverter
            inv = 255 - gray
            
            # Aplicar blur
            blur = cv2.GaussianBlur(inv, (7, 7), 0)
            
            # Dividir cinza pelo blur invertido
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            
            # Converter para 3 canais
            result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            
        elif effect == "cartoon":
            # Efeito de cartoon
            
            # Reduzir ruído
            smooth = cv2.bilateralFilter(result, 9, 75, 75)
            
            # Converter para escala de cinza
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            
            # Detectar bordas
            edges = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 8
            )
            
            # Converter bordas para BGR
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Combinar bordas com imagem suavizada
            result = cv2.bitwise_and(smooth, edges)
            
        else:
            logger.warning(f"Efeito desconhecido: {effect}")
            
        return result
        
    except Exception as e:
        logger.error(f"Erro ao aplicar efeito ao frame: {str(e)}")
        return frame

def detect_and_annotate_faces(
    frame: np.ndarray,
    draw_rectangles: bool = True,
    blur_faces: bool = False,
    rectangle_color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Detecta e anota faces em um frame.
    
    Args:
        frame: Frame para processar
        draw_rectangles: Se True, desenha retângulos ao redor das faces
        blur_faces: Se True, desfoca as faces detectadas
        rectangle_color: Cor dos retângulos
        thickness: Espessura das linhas dos retângulos
        
    Returns:
        Frame com faces anotadas/desfocadas
    """
    try:
        result = frame.copy()
        
        # Carregar classificador Haar Cascade para detecção de faces
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Processar cada face
        for (x, y, w, h) in faces:
            if draw_rectangles:
                # Desenhar retângulo
                cv2.rectangle(result, (x, y), (x+w, y+h), rectangle_color, thickness)
                
            if blur_faces:
                # Desfocar região da face
                face_roi = result[y:y+h, x:x+w]
                
                # Aplicar desfoque gaussiano
                blurred = cv2.GaussianBlur(face_roi, (31, 31), 0)
                
                # Substituir região
                result[y:y+h, x:x+w] = blurred
                
        return result
        
    except Exception as e:
        logger.error(f"Erro ao detectar faces: {str(e)}")
        return frame
