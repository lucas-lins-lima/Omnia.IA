"""
Audio Features - Funções para extração de features de áudio.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import librosa
import librosa.display

logger = logging.getLogger(__name__)

def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extrai coeficientes MFCCs de um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        n_mfcc: Número de coeficientes MFCC
        n_fft: Tamanho da FFT
        hop_length: Deslocamento entre janelas consecutivas
        
    Returns:
        Array com coeficientes MFCC
    """
    try:
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        logger.info(f"MFCCs extraídos com forma: {mfccs.shape}")
        return mfccs
    except Exception as e:
        logger.error(f"Erro ao extrair MFCCs: {str(e)}")
        raise
        
def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: int = 0,
    fmax: Optional[int] = None
) -> np.ndarray:
    """
    Extrai espectrograma mel de um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        n_fft: Tamanho da FFT
        hop_length: Deslocamento entre janelas consecutivas
        n_mels: Número de bandas mel
        fmin: Frequência mínima
        fmax: Frequência máxima
        
    Returns:
        Array com espectrograma mel
    """
    try:
        # Extrair espectrograma mel
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
        
        # Converter para dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        logger.info(f"Espectrograma mel extraído com forma: {mel_spec_db.shape}")
        return mel_spec_db
    except Exception as e:
        logger.error(f"Erro ao extrair espectrograma mel: {str(e)}")
        raise
        
def extract_chroma(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_chroma: int = 12
) -> np.ndarray:
    """
    Extrai features de chroma de um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        n_fft: Tamanho da FFT
        hop_length: Deslocamento entre janelas consecutivas
        n_chroma: Número de bins de chroma
        
    Returns:
        Array com features de chroma
    """
    try:
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_chroma=n_chroma
        )
        
        logger.info(f"Features de chroma extraídas com forma: {chroma.shape}")
        return chroma
    except Exception as e:
        logger.error(f"Erro ao extrair features de chroma: {str(e)}")
        raise
        
def extract_onset_strength(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512
) -> np.ndarray:
    """
    Calcula a força de onset de um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        hop_length: Deslocamento entre janelas consecutivas
        
    Returns:
        Array com força de onset
    """
    try:
                onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=hop_length
        )
        
        logger.info(f"Força de onset calculada com forma: {onset_env.shape}")
        return onset_env
    except Exception as e:
        logger.error(f"Erro ao calcular força de onset: {str(e)}")
        raise
        
def extract_tempo(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512
) -> Tuple[float, np.ndarray]:
    """
    Estima o tempo (BPM) de um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        hop_length: Deslocamento entre janelas consecutivas
        
    Returns:
        Tupla (tempo estimado, array de click track)
    """
    try:
        # Calcular onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio, 
            sr=sr,
            hop_length=hop_length
        )
        
        # Estimar tempo
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length
        )
        
        # Gerar click track
        click_track = librosa.clicks(
            frames=beats,
            sr=sr,
            hop_length=hop_length
        )
        
        logger.info(f"Tempo estimado: {tempo:.2f} BPM")
        return tempo, click_track
    except Exception as e:
        logger.error(f"Erro ao estimar tempo: {str(e)}")
        raise
        
def extract_pitch(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 2000.0
) -> np.ndarray:
    """
    Estima o pitch (F0) de um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        hop_length: Deslocamento entre janelas consecutivas
        fmin: Frequência mínima
        fmax: Frequência máxima
        
    Returns:
        Array com estimativas de pitch
    """
    try:
        # Usar algoritmo pYIN para estimativa de pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length
        )
        
        logger.info(f"Pitch extraído com forma: {f0.shape}")
        return f0
    except Exception as e:
        logger.error(f"Erro ao extrair pitch: {str(e)}")
        raise
        
def detect_speech_segments(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    frame_length: int = 2048,
    threshold: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Detecta segmentos de fala em um áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        hop_length: Deslocamento entre janelas consecutivas
        frame_length: Tamanho da janela
        threshold: Limiar para detecção de fala
        
    Returns:
        Lista de tuplas (tempo_início, tempo_fim) para cada segmento de fala
    """
    try:
        # Calcular energia RMS
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        ).squeeze()
        
        # Normalizar
        rms_norm = rms / np.max(rms) if np.max(rms) > 0 else rms
        
        # Encontrar segmentos acima do threshold
        is_speech = rms_norm > threshold
        
        # Encontrar transições
        transitions = np.diff(is_speech.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1
        
        # Ajustar se o áudio começa ou termina durante fala
        if is_speech[0]:
            starts = np.concatenate([[0], starts])
        if is_speech[-1]:
            ends = np.concatenate([ends, [len(is_speech)]])
            
        # Converter para tempos em segundos
        segments = [
            (start * hop_length / sr, end * hop_length / sr)
            for start, end in zip(starts, ends)
        ]
        
        logger.info(f"Detectados {len(segments)} segmentos de fala")
        return segments
    except Exception as e:
        logger.error(f"Erro ao detectar segmentos de fala: {str(e)}")
        raise
        
def extract_audio_features(
    audio: np.ndarray,
    sr: int,
    features: List[str] = ["mfcc", "chroma", "spectral_contrast", "tonnetz"]
) -> Dict[str, np.ndarray]:
    """
    Extrai múltiplas features de áudio.
    
    Args:
        audio: Array de áudio
        sr: Taxa de amostragem
        features: Lista de features a extrair
        
    Returns:
        Dicionário com as features extraídas
    """
    result = {}
    
    try:
        for feature in features:
            if feature == "mfcc":
                result["mfcc"] = extract_mfcc(audio, sr)
            elif feature == "mel_spectrogram":
                result["mel_spectrogram"] = extract_mel_spectrogram(audio, sr)
            elif feature == "chroma":
                result["chroma"] = extract_chroma(audio, sr)
            elif feature == "onset_strength":
                result["onset_strength"] = extract_onset_strength(audio, sr)
            elif feature == "tempo":
                result["tempo"], result["click_track"] = extract_tempo(audio, sr)
            elif feature == "pitch":
                result["pitch"] = extract_pitch(audio, sr)
            elif feature == "spectral_contrast":
                result["spectral_contrast"] = librosa.feature.spectral_contrast(
                    y=audio, sr=sr
                )
            elif feature == "tonnetz":
                result["tonnetz"] = librosa.feature.tonnetz(
                    y=audio, sr=sr
                )
            else:
                logger.warning(f"Feature não reconhecida: {feature}")
                
        return result
        
    except Exception as e:
        logger.error(f"Erro ao extrair features de áudio: {str(e)}")
        raise
