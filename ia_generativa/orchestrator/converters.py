"""
Converters - Funções para converter entre diferentes tipos de mídia.
"""

import os
import tempfile
import base64
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import cv2
import json
import subprocess
from PIL import Image
import io

# Importar funções de outros módulos
from preprocessors.video.video_loader import (
    extract_audio_from_video,
    extract_frames_from_video,
    load_video_from_base64
)
from preprocessors.audio.audio_processor import (
    load_audio_from_base64,
    audio_to_base64
)
from models.audio.speech_recognition import SpeechRecognitionManager
from models.audio.speech_synthesis import SpeechSynthesisManager
from models.image.image_understanding import ImageUnderstandingManager

logger = logging.getLogger(__name__)

# Instanciar gerenciadores necessários
speech_recognition = SpeechRecognitionManager()
speech_synthesis = SpeechSynthesisManager()
image_understanding = ImageUnderstandingManager()

def text_to_audio(
    text: str,
    voice_preset: str = "v2/en_speaker_6",
    sampling_rate: int = 24000,
    output_format: str = "wav",
    **kwargs
) -> str:
    """
    Converte texto em áudio utilizando síntese de fala.
    
    Args:
        text: Texto para converter em fala
        voice_preset: Preset de voz
        sampling_rate: Taxa de amostragem
        output_format: Formato do áudio de saída
        **kwargs: Parâmetros adicionais
        
    Returns:
        String base64 do áudio
    """
    try:
        logger.info(f"Convertendo texto para áudio: {len(text)} caracteres")
        
        # Sintetizar fala
        audio_array = speech_synthesis.synthesize(
            text=text,
            voice_preset=voice_preset,
            sampling_rate=sampling_rate
        )
        
        # Converter para base64
        audio_b64 = audio_to_base64(
            audio=audio_array,
            sr=sampling_rate,
            format=output_format,
            normalize=True
        )
        
        return audio_b64
        
    except Exception as e:
        logger.error(f"Erro ao converter texto para áudio: {str(e)}")
        raise
        
def audio_to_text(
    audio_data: str,
    language: Optional[str] = None,
    task: str = "transcribe",
    **kwargs
) -> str:
    """
    Converte áudio em texto utilizando reconhecimento de fala.
    
    Args:
        audio_data: Áudio em formato base64
        language: Código do idioma (opcional)
        task: Tarefa (transcribe ou translate)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Texto transcrito
    """
    try:
        logger.info(f"Convertendo áudio para texto")
        
        # Carregar áudio
        audio_array, sr = load_audio_from_base64(
            audio_data,
            target_sr=16000,  # Taxa esperada pelo Whisper
            mono=True
        )
        
        # Transcrever
        result = speech_recognition.transcribe(
            audio=audio_array,
            sampling_rate=16000,
            language=language,
            task=task
        )
        
        return result["text"]
        
    except Exception as e:
        logger.error(f"Erro ao converter áudio para texto: {str(e)}")
        raise
        
def image_to_text(
    image_data: str,
    mode: str = "caption",
    **kwargs
) -> str:
    """
    Extrai texto de uma imagem (captioning ou OCR).
    
    Args:
        image_data: Imagem em formato base64
        mode: Modo de extração (caption ou ocr)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Texto extraído
    """
    try:
        logger.info(f"Extraindo texto de imagem usando modo: {mode}")
        
        # Decodificar imagem de base64
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
            
        image_bytes = base64.b64decode(image_data)
        
        # Carregar imagem
        image = Image.open(io.BytesIO(image_bytes))
        
        if mode == "caption":
            # Usar modelo de captioning
            if image_understanding.video_captioner is None:
                image_understanding.load_video_captioner()
                
            # Gerar legenda
            caption = image_understanding.video_captioner(image)[0]["generated_text"]
            return caption
            
        elif mode == "ocr":
            # Importar biblioteca de OCR sob demanda
            try:
                import pytesseract
                
                # Converter para formato compatível com pytesseract
                image_np = np.array(image)
                
                # Extrair texto
                text = pytesseract.image_to_string(image_np)
                return text
                
            except ImportError:
                logger.error("pytesseract não encontrado. Instale com: pip install pytesseract")
                return "Erro: OCR não disponível (pytesseract não instalado)"
        else:
            raise ValueError(f"Modo de extração desconhecido: {mode}")
            
    except Exception as e:
        logger.error(f"Erro ao extrair texto de imagem: {str(e)}")
        raise
        
def text_to_image(
    text: str,
    width: int = 512,
    height: int = 512,
    model: str = "stable-diffusion",
    **kwargs
) -> str:
    """
    Gera uma imagem a partir de um texto utilizando modelos de difusão.
    
    Args:
        text: Prompt descritivo para geração da imagem
        width: Largura da imagem
        height: Altura da imagem
        model: Modelo a ser usado
        **kwargs: Parâmetros adicionais
        
    Returns:
        Imagem em formato base64
    """
    try:
        logger.info(f"Gerando imagem a partir de texto: {text[:50]}...")
        
        # Importar sob demanda para evitar dependência circular
        from models.image.diffusion_manager import DiffusionManager
        
        # Instanciar gerenciador
        diffusion_manager = DiffusionManager()
        
        # Gerar imagem
        images = diffusion_manager.generate_image(
            prompt=text,
            width=width,
            height=height,
            num_images=1
        )
        
        # Converter para base64
        from preprocessors.image.image_processor import convert_to_base64
        image_b64 = convert_to_base64(images[0], format="JPEG")
        
        return image_b64
        
    except Exception as e:
        logger.error(f"Erro ao gerar imagem a partir de texto: {str(e)}")
        raise
        
def video_to_audio(
    video_data: str,
    output_format: str = "mp3",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    **kwargs
) -> str:
    """
    Extrai o áudio de um vídeo.
    
    Args:
        video_data: Vídeo em formato base64
        output_format: Formato do áudio de saída
        start_time: Tempo inicial para extração (segundos)
        end_time: Tempo final para extração (segundos)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Áudio em formato base64
    """
    try:
        logger.info(f"Extraindo áudio de vídeo")
        
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar vídeo temporariamente
        video_path = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
        
        # Decodificar vídeo de base64
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Extrair áudio
        audio_path = extract_audio_from_video(
            video_path=video_path,
            format=output_format,
            start_time=start_time,
            end_time=end_time
        )
        
        # Ler arquivo de áudio
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        # Converter para base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Adicionar prefixo
        audio_b64 = f"data:audio/{output_format};base64,{audio_b64}"
        
        # Limpar arquivos temporários
        try:
            os.remove(video_path)
            os.remove(audio_path)
        except:
            pass
            
        return audio_b64
        
    except Exception as e:
        logger.error(f"Erro ao extrair áudio de vídeo: {str(e)}")
        raise
        
def video_to_text(
    video_data: str,
    mode: str = "transcribe",
    language: Optional[str] = None,
    **kwargs
) -> str:
    """
    Extrai texto de um vídeo (transcrição ou caption).
    
    Args:
        video_data: Vídeo em formato base64
        mode: Modo de extração (transcribe ou caption)
        language: Código do idioma para transcrição
        **kwargs: Parâmetros adicionais
        
    Returns:
        Texto extraído
    """
    try:
        logger.info(f"Extraindo texto de vídeo usando modo: {mode}")
        
        if mode == "transcribe":
            # Extrair áudio e transcrever
            audio_b64 = video_to_audio(
                video_data=video_data,
                output_format="wav"
            )
            
            # Transcrever áudio
            text = audio_to_text(
                audio_data=audio_b64,
                language=language
            )
            
            return text
            
        elif mode == "caption":
            # Extrair frames e gerar caption
            
            # Criar diretório temporário
            temp_dir = tempfile.gettempdir()
            
            # Salvar vídeo temporariamente
            video_path = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
            
            # Decodificar vídeo de base64
            if "base64," in video_data:
                video_data = video_data.split("base64,")[1]
                
            with open(video_path, "wb") as f:
                f.write(base64.b64decode(video_data))
                
            # Importar sob demanda
            from models.video.video_understanding import VideoUnderstandingManager
            
            # Instanciar gerenciador
            video_understanding = VideoUnderstandingManager()
            
            # Gerar caption
            caption_result = video_understanding.generate_video_caption(
                video_path=video_path,
                num_keyframes=5
            )
            
            # Limpar arquivo temporário
            try:
                os.remove(video_path)
            except:
                pass
                
            return caption_result["overall_description"]
            
        else:
            raise ValueError(f"Modo de extração desconhecido: {mode}")
            
    except Exception as e:
        logger.error(f"Erro ao extrair texto de vídeo: {str(e)}")
        raise
        
def video_to_images(
    video_data: str,
    frame_interval: int = 30,
    max_frames: Optional[int] = 10,
    output_format: str = "jpeg",
    **kwargs
) -> List[str]:
    """
    Extrai frames de um vídeo.
    
    Args:
        video_data: Vídeo em formato base64
        frame_interval: Intervalo entre frames
        max_frames: Número máximo de frames
        output_format: Formato das imagens
        **kwargs: Parâmetros adicionais
        
    Returns:
        Lista de imagens em formato base64
    """
    try:
        logger.info(f"Extraindo frames de vídeo: intervalo={frame_interval}, max={max_frames}")
        
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar vídeo temporariamente
        video_path = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
        
        # Decodificar vídeo de base64
        if "base64," in video_data:
            video_data = video_data.split("base64,")[1]
            
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(video_data))
            
        # Carregar vídeo e extrair frames
        frames, metadata = load_video_from_base64(
            video_data,
            frame_interval=frame_interval,
            max_frames=max_frames
        )
        
        # Converter frames para base64
        frame_base64 = []
        
        for frame in frames:
            # Converter para PIL Image
            pil_image = Image.fromarray(frame)
            
            # Converter para base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format=output_format.upper())
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Adicionar prefixo
            img_b64 = f"data:image/{output_format};base64,{img_str}"
            frame_base64.append(img_b64)
            
        # Limpar arquivo temporário
        try:
            os.remove(video_path)
        except:
            pass
            
        return frame_base64
        
    except Exception as e:
        logger.error(f"Erro ao extrair frames de vídeo: {str(e)}")
        raise
        
def images_to_video(
    images: List[str],
    fps: float = 30.0,
    output_format: str = "mp4",
    audio_data: Optional[str] = None,
    **kwargs
) -> str:
    """
    Cria um vídeo a partir de uma sequência de imagens.
    
    Args:
        images: Lista de imagens em formato base64
        fps: Frames por segundo
        output_format: Formato do vídeo de saída
        audio_data: Áudio opcional em formato base64
        **kwargs: Parâmetros adicionais
        
    Returns:
        Vídeo em formato base64
    """
    try:
        logger.info(f"Criando vídeo a partir de {len(images)} imagens")
        
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar imagens temporariamente
        frame_paths = []
        
        for i, img_b64 in enumerate(images):
            # Decodificar imagem de base64
            if "base64," in img_b64:
                img_b64 = img_b64.split("base64,")[1]
                
            img_bytes = base64.b64decode(img_b64)
            
            # Salvar imagem
            img_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
                
            frame_paths.append(img_path)
            
        # Salvar áudio temporariamente (se fornecido)
        audio_path = None
        if audio_data:
            # Decodificar áudio de base64
            if "base64," in audio_data:
                audio_data = audio_data.split("base64,")[1]
                
            audio_bytes = base64.b64decode(audio_data)
            
            # Salvar áudio
            audio_path = os.path.join(temp_dir, f"audio_{int(time.time())}.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
                
        # Criar vídeo a partir dos frames
        from preprocessors.video.video_loader import create_video_from_frames
        
        output_path = os.path.join(temp_dir, f"output_video_{int(time.time())}.{output_format}")
        
        video_path = create_video_from_frames(
            frame_paths=frame_paths,
            output_path=output_path,
            fps=fps,
            audio_path=audio_path
        )
        
        # Ler arquivo de vídeo
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            
        # Converter para base64
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        
        # Adicionar prefixo
        video_b64 = f"data:video/{output_format};base64,{video_b64}"
        
        # Limpar arquivos temporários
        try:
            for path in frame_paths:
                os.remove(path)
            if audio_path:
                os.remove(audio_path)
            os.remove(video_path)
        except:
            pass
            
        return video_b64
        
    except Exception as e:
        logger.error(f"Erro ao criar vídeo a partir de imagens: {str(e)}")
        raise
        
def pdf_to_text(
    pdf_data: str,
    mode: str = "text",
    **kwargs
) -> Union[str, List[str]]:
    """
    Extrai texto de um PDF.
    
    Args:
        pdf_data: PDF em formato base64
        mode: Modo de extração (text ou pages)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Texto extraído ou lista de páginas
    """
    try:
        logger.info(f"Extraindo texto de PDF usando modo: {mode}")
        
        # Importar bibliotecas sob demanda
        try:
            import PyPDF2
        except ImportError:
            logger.error("PyPDF2 não encontrado. Instale com: pip install PyPDF2")
            return "Erro: Extração de PDF não disponível (PyPDF2 não instalado)"
            
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar PDF temporariamente
        pdf_path = os.path.join(temp_dir, f"temp_pdf_{int(time.time())}.pdf")
        
        # Decodificar PDF de base64
        if "base64," in pdf_data:
            pdf_data = pdf_data.split("base64,")[1]
            
        with open(pdf_path, "wb") as f:
            f.write(base64.b64decode(pdf_data))
            
        # Abrir PDF
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            
            if mode == "text":
                # Extrair todo o texto
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                    
                return text.strip()
                
            elif mode == "pages":
                # Extrair texto por página
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text().strip())
                    
                return pages
                
            else:
                raise ValueError(f"Modo de extração desconhecido: {mode}")
                
    except Exception as e:
        logger.error(f"Erro ao extrair texto de PDF: {str(e)}")
        raise
        
def spreadsheet_to_text(
    spreadsheet_data: str,
    format: str = "csv",
    **kwargs
) -> str:
    """
    Extrai dados de uma planilha como texto.
    
    Args:
        spreadsheet_data: Planilha em formato base64
        format: Formato da planilha (csv ou xlsx)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Dados da planilha como texto
    """
    try:
        logger.info(f"Extraindo dados de planilha {format}")
        
        # Criar diretório temporário
        temp_dir = tempfile.gettempdir()
        
        # Salvar planilha temporariamente
        spreadsheet_path = os.path.join(temp_dir, f"temp_spreadsheet_{int(time.time())}.{format}")
        
        # Decodificar planilha de base64
        if "base64," in spreadsheet_data:
            spreadsheet_data = spreadsheet_data.split("base64,")[1]
            
        with open(spreadsheet_path, "wb") as f:
            f.write(base64.b64decode(spreadsheet_data))
            
        if format == "csv":
            # Importar biblioteca sob demanda
            try:
                import pandas as pd
            except ImportError:
                logger.error("pandas não encontrado. Instale com: pip install pandas")
                return "Erro: Processamento de planilha não disponível (pandas não instalado)"
                
            # Ler CSV
            df = pd.read_csv(spreadsheet_path)
            
            # Converter para texto
            return df.to_string(index=False)
            
        elif format == "xlsx":
            # Importar biblioteca sob demanda
            try:
                import pandas as pd
            except ImportError:
                logger.error("pandas não encontrado. Instale com: pip install pandas")
                return "Erro: Processamento de planilha não disponível (pandas não instalado)"
                
            # Ler XLSX
            df = pd.read_excel(spreadsheet_path)
            
            # Converter para texto
            return df.to_string(index=False)
            
        else:
            raise ValueError(f"Formato de planilha não suportado: {format}")
            
    except Exception as e:
        logger.error(f"Erro ao extrair dados de planilha: {str(e)}")
        raise
