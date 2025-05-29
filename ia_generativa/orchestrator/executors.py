"""
Executors - Funções executoras específicas por modalidade.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import tempfile
import time
import base64
import json
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Executores para TEXT

def text_summarize(
    text: str,
    max_length: int = 200,
    min_length: int = 50,
    **kwargs
) -> str:
    """
    Gera um resumo de um texto.
    
    Args:
        text: Texto para resumir
        max_length: Comprimento máximo do resumo
        min_length: Comprimento mínimo do resumo
        **kwargs: Parâmetros adicionais
        
    Returns:
        Texto resumido
    """
    try:
        logger.info(f"Resumindo texto de {len(text)} caracteres")
        
        # Importar sob demanda para evitar dependência circular
        from models.text.llm_manager import LLMManager
        
        # Instanciar gerenciador
        llm_manager = LLMManager()
        
        # Preparar prompt
        prompt = f"""Resuma o seguinte texto em no máximo {max_length} caracteres, 
        mantendo as informações mais importantes. O resumo deve ter pelo menos {min_length} caracteres.
        
        Texto:
        {text}
        
        Resumo:"""
        
        # Gerar resumo
        summary = llm_manager.generate_text(
            prompt=prompt,
            max_new_tokens=max_length,
            temperature=0.3,
            top_p=0.9
        )
        
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Erro ao resumir texto: {str(e)}")
        raise
        
def text_translate(
    text: str,
    target_language: str = "en",
    **kwargs
) -> str:
    """
    Traduz um texto para outro idioma.
    
    Args:
        text: Texto para traduzir
        target_language: Idioma alvo (código de 2 letras)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Texto traduzido
    """
    try:
        logger.info(f"Traduzindo texto para {target_language}")
        
        # Importar sob demanda para evitar dependência circular
        from models.text.llm_manager import LLMManager
        
        # Instanciar gerenciador
        llm_manager = LLMManager()
        
        # Mapear códigos para nomes completos
        language_map = {
            "en": "English",
            "pt": "Portuguese",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi"
        }
        
        # Obter nome completo do idioma
        target_language_full = language_map.get(target_language, target_language)
        
        # Preparar prompt
        prompt = f"""Translate the following text to {target_language_full}.
        
        Text:
        {text}
        
        Translation:"""
        
        # Gerar tradução
        translation = llm_manager.generate_text(
            prompt=prompt,
            max_new_tokens=len(text) * 2,  # Estimativa generosa
            temperature=0.1,
            top_p=0.9
        )
        
        return translation.strip()
        
    except Exception as e:
        logger.error(f"Erro ao traduzir texto: {str(e)}")
        raise
        
def text_analyze(
    text: str,
    analysis_type: str = "sentiment",
    **kwargs
) -> Dict[str, Any]:
    """
    Analisa um texto (sentimento, entidades, etc).
    
    Args:
        text: Texto para analisar
        analysis_type: Tipo de análise (sentiment, entities, keywords)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Resultado da análise
    """
    try:
        logger.info(f"Analisando texto: tipo={analysis_type}")
        
        # Importar sob demanda para evitar dependência circular
        from models.text.llm_manager import LLMManager
        
        # Instanciar gerenciador
        llm_manager = LLMManager()
        
        if analysis_type == "sentiment":
            # Análise de sentimento
            prompt = f"""Analyze the sentiment of the following text and respond with a JSON object containing:
            - "sentiment": "positive", "negative", or "neutral"
            - "confidence": a number between 0 and 1
            - "explanation": brief explanation for the classification
            
            Text:
            {text}
            
            JSON response:"""
            
        elif analysis_type == "entities":
            # Extração de entidades
            prompt = f"""Extract named entities from the following text and respond with a JSON array where each item is an object containing:
            - "entity": the extracted entity text
            - "type": the entity type (person, organization, location, date, etc.)
            - "start": starting character position
            - "end": ending character position
            
            Text:
            {text}
            
            JSON response:"""
            
        elif analysis_type == "keywords":
            # Extração de palavras-chave
            prompt = f"""Extract the main keywords from the following text and respond with a JSON array where each item is an object containing:
            - "keyword": the extracted keyword or phrase
            - "relevance": a number between 0 and 1 indicating relevance
            
            Text:
            {text}
            
            JSON response:"""
            
        else:
            raise ValueError(f"Tipo de análise não suportado: {analysis_type}")
            
        # Gerar análise
        result_text = llm_manager.generate_text(
            prompt=prompt,
            max_new_tokens=1000,
            temperature=0.1,
            top_p=0.9
        )
        
        # Extrair JSON da resposta
        try:
            # Tentar extrair bloco JSON, se houver
            if "```json" in result_text:
                json_str = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                json_str = result_text.split("```")[1].strip()
            else:
                json_str = result_text.strip()
                
            # Analisar JSON
            result = json.loads(json_str)
            return result
            
        except Exception as json_error:
            logger.error(f"Erro ao analisar JSON da resposta: {str(json_error)}")
            return {"error": "Falha ao analisar resultado", "text": result_text}
            
    except Exception as e:
        logger.error(f"Erro ao analisar texto: {str(e)}")
        raise
        
# Executores para IMAGE

def image_classify(
    image_data: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Classifica o conteúdo de uma imagem.
    
    Args:
        image_data: Imagem em formato base64
        **kwargs: Parâmetros adicionais
        
    Returns:
        Resultados da classificação
    """
    try:
        logger.info(f"Classificando imagem")
        
        # Importar sob demanda para evitar dependência circular
        from models.image.image_understanding import ImageUnderstandingManager
        
        # Instanciar gerenciador
        image_understanding = ImageUnderstandingManager()
        
        # Decodificar imagem de base64
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
            
        image_bytes = base64.b64decode(image_data)
        
        # Carregar imagem
        image = Image.open(io.BytesIO(image_bytes))
        
        # Classificar imagem
        candidate_labels = kwargs.get("candidate_labels", [
            "paisagem", "retrato", "animal", "comida", "veículo", 
            "esporte", "arquitetura", "texto", "arte", "tecnologia"
        ])
        
        # Carregar modelo CLIP se necessário
        if image_understanding.clip_model is None:
            image_understanding.load_clip_model()
            
        # Calcular similaridade com labels candidatos
        result = image_understanding.classify_image(image, candidate_labels)
        
        return result
        
    except Exception as e:
        logger.error(f"Erro ao classificar imagem: {str(e)}")
        raise
        
def image_edit(
    image_data: str,
    edit_type: str = "filter",
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Aplica edições a uma imagem.
    
    Args:
        image_data: Imagem em formato base64
        edit_type: Tipo de edição (filter, crop, resize, etc.)
        parameters: Parâmetros específicos para a edição
        **kwargs: Parâmetros adicionais
        
    Returns:
        Imagem editada em formato base64
    """
    try:
        logger.info(f"Editando imagem: tipo={edit_type}")
        
        # Validar parâmetros
        parameters = parameters or {}
        
        # Decodificar imagem de base64
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
            
        image_bytes = base64.b64decode(image_data)
        
        # Carregar imagem com OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Aplicar edição apropriada
        if edit_type == "filter":
            # Aplicar filtro de cor
            filter_type = parameters.get("filter_type", "grayscale")
            intensity = parameters.get("intensity", 1.0)
            
            # Importar função apropriada
            from preprocessors.video.frame_processor import apply_color_filter
            
            result_img = apply_color_filter(img, filter_type, intensity)
            
        elif edit_type == "resize":
            # Redimensionar imagem
            width = parameters.get("width")
            height = parameters.get("height")
            scale = parameters.get("scale")
            
            # Importar função apropriada
            from preprocessors.video.frame_processor import resize_frame
            
            result_img = resize_frame(img, width, height, scale)
            
        elif edit_type == "crop":
            # Recortar imagem
            x = parameters.get("x", 0)
            y = parameters.get("y", 0)
            width = parameters.get("width", img.shape[1] - x)
            height = parameters.get("height", img.shape[0] - y)
            
            result_img = img[y:y+height, x:x+width]
            
        elif edit_type == "rotate":
            # Rotacionar imagem
            angle = parameters.get("angle", 90)
            
            # Obter dimensões
            h, w = img.shape[:2]
            
            # Calcular matriz de rotação
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Aplicar rotação
            result_img = cv2.warpAffine(img, M, (w, h))
            
        elif edit_type == "adjust":
            # Ajustar propriedades da imagem
            brightness = parameters.get("brightness", 0.0)
            contrast = parameters.get("contrast", 0.0)
            saturation = parameters.get("saturation", 0.0)
            
            # Importar função apropriada
            from preprocessors.video.frame_processor import adjust_frame
            
            result_img = adjust_frame(img, brightness, contrast, saturation)
            
        elif edit_type == "effect":
            # Aplicar efeito especial
            effect_type = parameters.get("effect_type", "blur")
            effect_params = parameters.get("params", {})
            
            # Importar função apropriada
            from preprocessors.video.frame_processor import apply_effect_to_frame
            
            result_img = apply_effect_to_frame(img, effect_type, effect_params)
            
        else:
            raise ValueError(f"Tipo de edição não suportado: {edit_type}")
            
        # Converter resultado de volta para base64
        from preprocessors.image.image_processor import convert_to_base64
        
        # Converter para PIL Image
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_img_rgb)
        
        # Converter para base64
        result_b64 = convert_to_base64(pil_img, format="JPEG")
        
        return result_b64
        
    except Exception as e:
        logger.error(f"Erro ao editar imagem: {str(e)}")
        raise

# Executores para AUDIO

def audio_edit(
    audio_data: str,
    edit_type: str = "normalize",
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Aplica edições a um áudio.
    
    Args:
        audio_data: Áudio em formato base64
        edit_type: Tipo de edição (normalize, trim, filter, etc.)
        parameters: Parâmetros específicos para a edição
        **kwargs: Parâmetros adicionais
        
            Returns:
        Áudio editado em formato base64
    """
    try:
        logger.info(f"Editando áudio: tipo={edit_type}")
        
        # Validar parâmetros
        parameters = parameters or {}
        
        # Importar funções necessárias
        from preprocessors.audio.audio_processor import (
            load_audio_from_base64,
            audio_to_base64,
            normalize_audio
        )
        
        # Carregar áudio
        audio_array, sr = load_audio_from_base64(
            audio_data,
            normalize=False  # Não normalizar aqui, pode ser uma operação solicitada
        )
        
        # Determinar formato de saída
        if "base64," in audio_data and "audio/" in audio_data:
            format = audio_data.split("audio/")[1].split(";")[0]
        else:
            format = "wav"  # Default
            
        # Aplicar edição apropriada
        if edit_type == "normalize":
            # Normalizar volume
            target_db = parameters.get("target_db", -3.0)
            headroom_db = parameters.get("headroom_db", 1.0)
            
            result_audio = normalize_audio(
                audio_array,
                target_db=target_db,
                headroom_db=headroom_db
            )
            
        elif edit_type == "trim":
            # Recortar trecho de áudio
            start_time = parameters.get("start_time", 0.0)
            end_time = parameters.get("end_time")
            
            # Converter tempo para amostras
            start_sample = int(start_time * sr)
            
            if end_time is not None:
                end_sample = int(end_time * sr)
                result_audio = audio_array[start_sample:end_sample]
            else:
                result_audio = audio_array[start_sample:]
                
        elif edit_type == "noise_reduction":
            # Redução de ruído (simplificado)
            import scipy.signal as signal
            
            # Aplicar filtro de mediana para reduzir ruído impulsivo
            result_audio = signal.medfilt(audio_array, kernel_size=3)
            
        elif edit_type == "change_speed":
            # Alterar velocidade
            import librosa
            
            speed_factor = parameters.get("factor", 1.0)
            
            # Alterar velocidade sem afetar pitch
            result_audio = librosa.effects.time_stretch(audio_array, rate=speed_factor)
            
        elif edit_type == "change_pitch":
            # Alterar pitch
            import librosa
            
            n_steps = parameters.get("steps", 0)
            
            # Alterar pitch sem afetar tempo
            if n_steps != 0:
                result_audio = librosa.effects.pitch_shift(
                    audio_array, 
                    sr=sr, 
                    n_steps=n_steps
                )
            else:
                result_audio = audio_array
                
        elif edit_type == "filter":
            # Aplicar filtro de frequência
            from scipy import signal
            
            filter_type = parameters.get("filter_type", "lowpass")
            cutoff_freq = parameters.get("cutoff_freq", 1000)
            order = parameters.get("order", 5)
            
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
                high_cutoff = parameters.get("high_cutoff", 2000)
                normal_high = high_cutoff / nyquist
                b, a = signal.butter(order, [normal_cutoff, normal_high], btype='band')
            else:
                raise ValueError(f"Tipo de filtro não suportado: {filter_type}")
                
            result_audio = signal.filtfilt(b, a, audio_array)
            
        else:
            raise ValueError(f"Tipo de edição não suportado: {edit_type}")
            
        # Converter resultado de volta para base64
        result_b64 = audio_to_base64(
            result_audio,
            sr=sr,
            format=format,
            normalize=True  # Normalizar na saída para evitar clipping
        )
        
        return result_b64
        
    except Exception as e:
        logger.error(f"Erro ao editar áudio: {str(e)}")
        raise
        
def audio_analyze(
    audio_data: str,
    analysis_type: str = "features",
    **kwargs
) -> Dict[str, Any]:
    """
    Analisa características de um áudio.
    
    Args:
        audio_data: Áudio em formato base64
        analysis_type: Tipo de análise (features, speech_detection, etc.)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Resultados da análise
    """
    try:
        logger.info(f"Analisando áudio: tipo={analysis_type}")
        
        # Importar funções necessárias
        from preprocessors.audio.audio_processor import load_audio_from_base64
        
        # Carregar áudio
        audio_array, sr = load_audio_from_base64(audio_data)
        
        if analysis_type == "features":
            # Extrair características básicas
            from preprocessors.audio.audio_features import extract_audio_features
            
            # Definir features a extrair
            features_to_extract = kwargs.get("features", [
                "mfcc", "chroma", "spectral_contrast", "tonnetz"
            ])
            
            # Extrair features
            features_dict = extract_audio_features(
                audio_array,
                sr=sr,
                features=features_to_extract
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
                    
            return result
            
        elif analysis_type == "speech_detection":
            # Detectar segmentos de fala
            from preprocessors.audio.audio_features import detect_speech_segments
            
            # Parâmetros
            threshold = kwargs.get("threshold", 0.1)
            
            # Detectar segmentos
            segments = detect_speech_segments(
                audio_array,
                sr=sr,
                threshold=threshold
            )
            
            return {"segments": segments}
            
        elif analysis_type == "tempo":
            # Estimar tempo/BPM
            from preprocessors.audio.audio_features import extract_tempo
            
            # Estimar tempo
            tempo, _ = extract_tempo(audio_array, sr=sr)
            
            return {"tempo": float(tempo)}
            
        else:
            raise ValueError(f"Tipo de análise não suportado: {analysis_type}")
            
    except Exception as e:
        logger.error(f"Erro ao analisar áudio: {str(e)}")
        raise

# Executores para VIDEO

def video_edit(
    video_data: str,
    edit_type: str = "process",
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Aplica edições a um vídeo.
    
    Args:
        video_data: Vídeo em formato base64
        edit_type: Tipo de edição (process, trim, speed, etc.)
        parameters: Parâmetros específicos para a edição
        **kwargs: Parâmetros adicionais
        
    Returns:
        Vídeo editado em formato base64
    """
    try:
        logger.info(f"Editando vídeo: tipo={edit_type}")
        
        # Validar parâmetros
        parameters = parameters or {}
        
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar vídeo temporariamente
        video_path = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
        
        # Decodificar vídeo de base64
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Importar módulo sob demanda
        from models.video.video_processor import VideoProcessor
        
        # Instanciar processador
        video_processor = VideoProcessor()
        
        if edit_type == "process":
            # Aplicar operações de processamento
            operations = parameters.get("operations", [])
            
            # Processar vídeo
            output_path = video_processor.process_video(
                video_path=video_path,
                operations=operations,
                preserve_audio=parameters.get("preserve_audio", True)
            )
            
        elif edit_type == "trim":
            # Recortar vídeo
            start_time = parameters.get("start_time", 0.0)
            end_time = parameters.get("end_time")
            
            # Recortar vídeo
            output_path = video_processor.trim_video(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time
            )
            
        elif edit_type == "speed":
            # Alterar velocidade
            speed_factor = parameters.get("speed_factor", 1.0)
            preserve_audio_pitch = parameters.get("preserve_audio_pitch", True)
            
            # Alterar velocidade
            output_path = video_processor.change_video_speed(
                video_path=video_path,
                speed_factor=speed_factor,
                preserve_audio_pitch=preserve_audio_pitch
            )
            
        elif edit_type == "timelapse":
            # Criar timelapse
            speed_factor = parameters.get("speed_factor", 10.0)
            frame_interval = parameters.get("frame_interval", 1)
            
            # Criar timelapse
            output_path = video_processor.create_timelapse(
                video_path=video_path,
                speed_factor=speed_factor,
                frame_interval=frame_interval
            )
            
        else:
            raise ValueError(f"Tipo de edição não suportado: {edit_type}")
            
        # Converter para base64
        from preprocessors.video.video_loader import video_to_base64
        
        video_b64 = video_to_base64(output_path)
        
        # Limpar arquivos temporários
        try:
            os.remove(video_path)
            if output_path != video_path:
                os.remove(output_path)
        except:
            pass
            
        return video_b64
        
    except Exception as e:
        logger.error(f"Erro ao editar vídeo: {str(e)}")
        raise
        
def video_analyze(
    video_data: str,
    analysis_type: str = "content",
    **kwargs
) -> Dict[str, Any]:
    """
    Analisa o conteúdo de um vídeo.
    
    Args:
        video_data: Vídeo em formato base64
        analysis_type: Tipo de análise (content, scenes, objects, etc.)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Resultados da análise
    """
    try:
        logger.info(f"Analisando vídeo: tipo={analysis_type}")
        
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar vídeo temporariamente
        video_path = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
        
        # Decodificar vídeo de base64
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        if analysis_type == "content":
            # Análise completa de conteúdo
            from models.video.video_understanding import VideoUnderstandingManager
            
            # Instanciar gerenciador
            video_understanding = VideoUnderstandingManager()
            
            # Analisar vídeo
            result = video_understanding.analyze_video_content(
                video_path=video_path,
                analyze_objects=kwargs.get("analyze_objects", True),
                classify_content=kwargs.get("classify_content", True),
                generate_caption=kwargs.get("generate_caption", True),
                frame_interval=kwargs.get("frame_interval", 10)
            )
            
        elif analysis_type == "scenes":
            # Detecção de cenas
            from models.video.video_processor import VideoProcessor
            
            # Instanciar processador
            video_processor = VideoProcessor()
            
            # Detectar cenas
            threshold = kwargs.get("threshold", 30.0)
            min_scene_length = kwargs.get("min_scene_length", 1.0)
            
            scene_transitions = video_processor.extract_scene_transitions(
                video_path=video_path,
                threshold=threshold,
                min_scene_length=min_scene_length
            )
            
            # Obter informações do vídeo
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            result = {
                "scenes": scene_transitions,
                "duration": duration,
                "total_scenes": len(scene_transitions)
            }
            
        elif analysis_type == "objects":
            # Detecção de objetos
            from models.video.video_understanding import VideoUnderstandingManager
            
            # Instanciar gerenciador
            video_understanding = VideoUnderstandingManager()
            
            # Detectar objetos
            result = video_understanding.detect_objects_in_video(
                video_path=video_path,
                confidence_threshold=kwargs.get("confidence_threshold", 0.3),
                frame_interval=kwargs.get("frame_interval", 10)
            )
            
        else:
            raise ValueError(f"Tipo de análise não suportado: {analysis_type}")
            
        # Limpar arquivo temporário
        try:
            os.remove(video_path)
        except:
            pass
            
        return result
        
    except Exception as e:
        logger.error(f"Erro ao analisar vídeo: {str(e)}")
        raise
