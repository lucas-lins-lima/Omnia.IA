"""
Audio Utils - Funções utilitárias para manipulação de áudio.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import io
import base64
import tempfile
import time
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import librosa.display

logger = logging.getLogger(__name__)

def create_spectrogram_image(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: int = 0,
    fmax: Optional[int] = None,
    title: str = "Spectrogram",
    width: int = 10,
    height: int = 4,
    cmap: str = 'viridis'
) -> np.ndarray:
    """
    Cria uma imagem do espectrograma de um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        n_fft: Tamanho da FFT
        hop_length: Deslocamento entre janelas consecutivas
        fmin: Frequência mínima
        fmax: Frequência máxima
        title: Título do gráfico
        width: Largura da imagem em polegadas
        height: Altura da imagem em polegadas
        cmap: Mapa de cores
        
    Returns:
        Array NumPy com a imagem do espectrograma
    """
    try:
        plt.figure(figsize=(width, height))
        
        # Calcular espectrograma
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        
        # Plotar
        librosa.display.specshow(
            D,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='log',
            fmin=fmin,
            fmax=fmax,
            cmap=cmap
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        # Converter para array
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Carregar como imagem PIL e converter para array
        from PIL import Image
        image = Image.open(buffer)
        image_array = np.array(image)
        
        plt.close()
        
        return image_array
        
    except Exception as e:
        logger.error(f"Erro ao criar imagem de espectrograma: {str(e)}")
        plt.close()
        raise
        
def create_waveform_image(
    audio: np.ndarray,
    sr: int,
    title: str = "Waveform",
    width: int = 10,
    height: int = 3,
    color: str = 'blue'
) -> np.ndarray:
    """
    Cria uma imagem da forma de onda de um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        title: Título do gráfico
        width: Largura da imagem em polegadas
        height: Altura da imagem em polegadas
        color: Cor da forma de onda
        
    Returns:
        Array NumPy com a imagem da forma de onda
    """
    try:
        plt.figure(figsize=(width, height))
        
        # Criar eixo de tempo
        time = np.linspace(0, len(audio) / sr, len(audio))
        
        # Plotar forma de onda
        plt.plot(time, audio, color=color, alpha=0.8)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        
        # Converter para array
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Carregar como imagem PIL e converter para array
        from PIL import Image
        image = Image.open(buffer)
        image_array = np.array(image)
        
        plt.close()
        
        return image_array
        
    except Exception as e:
        logger.error(f"Erro ao criar imagem de forma de onda: {str(e)}")
        plt.close()
        raise
        
def detect_audio_features(
    audio: np.ndarray,
    sr: int
) -> Dict[str, Any]:
    """
    Detecta características básicas de um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        
    Returns:
        Dicionário com características detectadas
    """
    try:
        import librosa
        import numpy as np
        
        # Duração em segundos
        duration = len(audio) / sr
        
        # Estatísticas básicas
        min_amp = float(np.min(audio))
        max_amp = float(np.max(audio))
        mean_amp = float(np.mean(audio))
        rms = float(np.sqrt(np.mean(audio**2)))
        
        # Converter para dB
        db_rms = 20 * np.log10(rms) if rms > 0 else -100
        
        # Estimar BPM
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        except Exception:
            tempo = None
            
        # Detectar silêncio
        non_silent = librosa.effects.split(
            audio,
            top_db=40,
            frame_length=2048,
            hop_length=512
        )
        
        silence_percentage = 100 - (sum(end - start for start, end in non_silent) / len(audio) * 100)
        
        return {
            "duration": duration,
            "sampling_rate": sr,
            "min_amplitude": min_amp,
            "max_amplitude": max_amp,
            "mean_amplitude": mean_amp,
            "rms_amplitude": rms,
            "db_rms": db_rms,
            "estimated_tempo": tempo,
            "silence_percentage": silence_percentage
        }
        
    except Exception as e:
        logger.error(f"Erro ao detectar características de áudio: {str(e)}")
        raise
        
def mix_audio(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr1: int,
    sr2: int,
    weight1: float = 0.5,
    weight2: float = 0.5,
    target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Mistura dois áudios com pesos diferentes.
    
    Args:
        audio1: Primeiro array de áudio
        audio2: Segundo array de áudio
        sr1: Taxa de amostragem do primeiro áudio
        sr2: Taxa de amostragem do segundo áudio
        weight1: Peso para o primeiro áudio (0-1)
        weight2: Peso para o segundo áudio (0-1)
        target_sr: Taxa de amostragem alvo (se None, usa a maior das duas)
        
    Returns:
        Tupla (array de áudio misturado, taxa de amostragem)
    """
    try:
        import librosa
        
        # Determinar taxa de amostragem alvo
        output_sr = target_sr or max(sr1, sr2)
        
        # Resample se necessário
        if sr1 != output_sr:
            audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=output_sr)
        if sr2 != output_sr:
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=output_sr)
            
        # Determinar comprimento do resultado
        length = min(len(audio1), len(audio2))
        
        # Cortar para mesmo comprimento
        audio1 = audio1[:length]
        audio2 = audio2[:length]
        
        # Normalizar pesos
        total_weight = weight1 + weight2
        weight1 = weight1 / total_weight
        weight2 = weight2 / total_weight
        
        # Misturar áudios
        mixed = (audio1 * weight1) + (audio2 * weight2)
        
        # Normalizar
        if np.max(np.abs(mixed)) > 1.0:
            mixed = mixed / np.max(np.abs(mixed))
            
        return mixed, output_sr
        
    except Exception as e:
        logger.error(f"Erro ao misturar áudios: {str(e)}")
        raise
