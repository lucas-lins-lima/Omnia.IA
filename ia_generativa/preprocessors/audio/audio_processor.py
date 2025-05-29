"""
Audio Processor - Funções para processamento e manipulação de áudio.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import io
import base64
import tempfile
import soundfile as sf
import librosa
import pydub
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def load_audio_file(
    audio_path: str,
    target_sr: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Carrega um arquivo de áudio.
    
    Args:
        audio_path: Caminho para o arquivo de áudio
        target_sr: Taxa de amostragem alvo (se None, mantém a original)
        mono: Se True, converte para mono
        normalize: Se True, normaliza o áudio entre -1 e 1
        
    Returns:
        Tupla (array de áudio, taxa de amostragem)
    """
    try:
        logger.info(f"Carregando arquivo de áudio: {audio_path}")
        
        # Detectar formato a partir da extensão
        _, extension = os.path.splitext(audio_path)
        extension = extension.lower().strip(".")
        
        # Para formatos suportados diretamente pelo librosa
        if extension in ["wav", "mp3", "ogg", "flac", "m4a"]:
            # Carregar com librosa (para mp3, ogg, etc.)
            y, sr = librosa.load(
                audio_path, 
                sr=target_sr,
                mono=mono
            )
            
            logger.info(f"Áudio carregado: {len(y) / sr:.2f}s, {sr}Hz, {y.shape} shape")
            
            # Normalizar se solicitado
            if normalize and y.size > 0:
                y = normalize_audio(y)
                
            return y, sr
            
        # Para outros formatos, usar pydub
        else:
            audio = AudioSegment.from_file(audio_path)
            
            # Converter para mono se necessário
            if mono and audio.channels > 1:
                audio = audio.set_channels(1)
                
            # Resample se necessário
            orig_sr = audio.frame_rate
            if target_sr is not None and orig_sr != target_sr:
                audio = audio.set_frame_rate(target_sr)
                
            # Converter para numpy array
            y = np.array(audio.get_array_of_samples()).astype(np.float32)
            
            # Normalizar para [-1, 1]
            if normalize:
                y = y / (1 << (8 * audio.sample_width - 1))
                
            sr = target_sr or orig_sr
            
            logger.info(f"Áudio carregado (pydub): {len(y) / sr:.2f}s, {sr}Hz, {y.shape} shape")
            
            return y, sr
            
    except Exception as e:
        logger.error(f"Erro ao carregar áudio: {str(e)}")
        raise
        
def load_audio_from_bytes(
    audio_bytes: bytes,
    format: str = "wav",
    target_sr: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Carrega áudio a partir de bytes.
    
    Args:
        audio_bytes: Bytes do áudio
        format: Formato do áudio (wav, mp3, etc.)
        target_sr: Taxa de amostragem alvo (se None, mantém a original)
        mono: Se True, converte para mono
        normalize: Se True, normaliza o áudio entre -1 e 1
        
    Returns:
        Tupla (array de áudio, taxa de amostragem)
    """
    try:
        # Salvar em arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(audio_bytes)
            
        # Carregar do arquivo temporário
        y, sr = load_audio_file(
            tmp_path,
            target_sr=target_sr,
            mono=mono,
            normalize=normalize
        )
        
        # Remover arquivo temporário
        os.unlink(tmp_path)
        
        return y, sr
        
    except Exception as e:
        logger.error(f"Erro ao carregar áudio de bytes: {str(e)}")
        raise
        
def load_audio_from_base64(
    base64_string: str,
    format: str = "wav",
    target_sr: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Carrega áudio a partir de uma string base64.
    
    Args:
        base64_string: String base64 do áudio
        format: Formato do áudio (wav, mp3, etc.)
        target_sr: Taxa de amostragem alvo (se None, mantém a original)
        mono: Se True, converte para mono
        normalize: Se True, normaliza o áudio entre -1 e 1
        
    Returns:
        Tupla (array de áudio, taxa de amostragem)
    """
    try:
        # Remover prefixo se presente
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        # Decodificar string base64
        audio_bytes = base64.b64decode(base64_string)
        
        # Carregar dos bytes
        return load_audio_from_bytes(
            audio_bytes,
            format=format,
            target_sr=target_sr,
            mono=mono,
            normalize=normalize
        )
        
    except Exception as e:
        logger.error(f"Erro ao carregar áudio de base64: {str(e)}")
        raise
        
def normalize_audio(
    audio: np.ndarray,
    target_db: float = -3.0,
    headroom_db: float = 1.0
) -> np.ndarray:
    """
    Normaliza o volume do áudio.
    
    Args:
        audio: Array de áudio
        target_db: Volume alvo em dB (negativo)
        headroom_db: Margem de segurança em dB
        
    Returns:
        Array de áudio normalizado
    """
    # Verificar se o áudio não é silêncio
    if np.max(np.abs(audio)) < 1e-6:
        return audio
        
    # Calcular RMS atual
    rms = np.sqrt(np.mean(audio**2))
    
    # Converter para dB
    current_db = 20 * np.log10(rms)
    
    # Calcular ganho necessário
    gain_db = target_db - current_db - headroom_db
    
    # Aplicar ganho
    gain_linear = 10 ** (gain_db / 20)
    normalized_audio = audio * gain_linear
    
    # Limitar para evitar clipping
    if np.max(np.abs(normalized_audio)) > 1.0:
        normalized_audio = normalized_audio / np.max(np.abs(normalized_audio))
        
    return normalized_audio
    
def convert_audio_format(
    audio: np.ndarray,
    sr: int,
    output_path: str,
    format: str = "wav",
    sample_width: int = 2,
    normalize: bool = True
) -> str:
    """
    Converte e salva áudio em um formato específico.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        output_path: Caminho para salvar o arquivo
        format: Formato de saída (wav, mp3, etc.)
        sample_width: Largura da amostra em bytes (qualidade)
        normalize: Se True, normaliza o áudio antes de salvar
        
    Returns:
        Caminho do arquivo salvo
    """
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Normalizar se solicitado
        if normalize:
            audio = normalize_audio(audio)
            
        # Para wav, usar soundfile diretamente
        if format.lower() == "wav":
            sf.write(output_path, audio, sr, subtype=f'PCM_{sample_width*8}')
        else:
            # Para outros formatos, usar pydub
            # Converter para array de inteiros primeiro
            bit_depth = sample_width * 8
            max_value = 2 ** (bit_depth - 1) - 1
            audio_int = (audio * max_value).astype(np.int16 if sample_width == 2 else np.int32)
            
            # Criar segmento de áudio
            segment = AudioSegment(
                audio_int.tobytes(),
                frame_rate=sr,
                sample_width=sample_width,
                channels=1 if audio.ndim == 1 else audio.shape[1]
            )
            
            # Exportar no formato desejado
            segment.export(output_path, format=format)
            
        logger.info(f"Áudio convertido e salvo em: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Erro ao converter áudio: {str(e)}")
        raise
        
def audio_to_base64(
    audio: np.ndarray,
    sr: int,
    format: str = "wav",
    sample_width: int = 2,
    normalize: bool = True
) -> str:
    """
    Converte áudio para string base64.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        format: Formato (wav, mp3, etc.)
        sample_width: Largura da amostra em bytes
        normalize: Se True, normaliza o áudio
        
    Returns:
        String base64 do áudio
    """
    try:
        # Criar buffer temporário
        with io.BytesIO() as buffer:
            # Normalizar se solicitado
            if normalize:
                audio = normalize_audio(audio)
                
            # Para wav, usar soundfile diretamente
            if format.lower() == "wav":
                sf.write(buffer, audio, sr, format=format, subtype=f'PCM_{sample_width*8}')
            else:
                # Para outros formatos, usar pydub
                # Converter para array de inteiros primeiro
                bit_depth = sample_width * 8
                max_value = 2 ** (bit_depth - 1) - 1
                audio_int = (audio * max_value).astype(np.int16 if sample_width == 2 else np.int32)
                
                # Criar segmento de áudio
                segment = AudioSegment(
                    audio_int.tobytes(),
                    frame_rate=sr,
                    sample_width=sample_width,
                    channels=1 if audio.ndim == 1 else audio.shape[1]
                )
                
                # Exportar no formato desejado
                segment.export(buffer, format=format)
                
            # Obter bytes e converter para base64
            buffer.seek(0)
            audio_bytes = buffer.getvalue()
            base64_str = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Adicionar prefixo com tipo MIME
            mime_type = f"audio/{format}"
            if format.lower() == "mp3":
                mime_type = "audio/mpeg"
                
            return f"data:{mime_type};base64,{base64_str}"
            
    except Exception as e:
        logger.error(f"Erro ao converter áudio para base64: {str(e)}")
        raise
        
def split_audio(
    audio: np.ndarray,
    sr: int,
    max_duration_s: float = 30.0,
    min_duration_s: float = 1.0,
    overlap_s: float = 0.5
) -> List[np.ndarray]:
    """
    Divide um áudio em segmentos menores.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        max_duration_s: Duração máxima de cada segmento em segundos
        min_duration_s: Duração mínima de um segmento válido em segundos
        overlap_s: Sobreposição entre segmentos em segundos
        
    Returns:
        Lista de arrays de áudio
    """
    # Calcular parâmetros em amostras
    max_samples = int(max_duration_s * sr)
    min_samples = int(min_duration_s * sr)
    overlap_samples = int(overlap_s * sr)
    
    # Se o áudio é menor que a duração mínima, retornar como está
    if len(audio) < min_samples:
        return [audio]
        
    # Se o áudio é menor que a duração máxima, retornar como está
    if len(audio) <= max_samples:
        return [audio]
        
    # Dividir em segmentos
    segments = []
    start = 0
    
    while start < len(audio):
        end = min(start + max_samples, len(audio))
        segment = audio[start:end]
        
        # Adicionar apenas se o segmento for grande o suficiente
        if len(segment) >= min_samples:
            segments.append(segment)
            
        # Avançar para o próximo segmento com sobreposição
        start = end - overlap_samples
        
    logger.info(f"Áudio dividido em {len(segments)} segmentos")
    return segments
