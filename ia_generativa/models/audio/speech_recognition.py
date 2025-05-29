"""
Speech Recognition Manager - Responsável por carregar e utilizar modelos de reconhecimento de fala.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import logging
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

logger = logging.getLogger(__name__)

class SpeechRecognitionManager:
    """
    Gerencia o carregamento e execução de modelos de reconhecimento de fala (ASR).
    """
    
    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-medium",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        use_flash_attention: bool = True
    ):
        """
        Inicializa o gerenciador de reconhecimento de fala.
        
        Args:
            model_name_or_path: ID do modelo no HuggingFace ou caminho local
            device: Dispositivo para execução (cuda, cpu, etc.)
            dtype: Tipo de dados para o modelo (float16, float32)
            cache_dir: Diretório para cache de modelos
            use_flash_attention: Se True, usa flash attention para maior eficiência
        """
        self.model_name = model_name_or_path
        
        # Determinar o dispositivo automaticamente se não for especificado
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Determinar o tipo de dados com base no dispositivo e preferências
        if dtype is None:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = dtype
            
        self.use_flash_attention = use_flash_attention
        
        # Configurar cache para os modelos
        self.cache_dir = cache_dir or os.path.join("data", "models_cache")
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        
        logger.info(f"Inicializando SpeechRecognitionManager com modelo: {self.model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Inicialmente não carregamos os modelos, apenas quando necessário
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Carrega o modelo e processador na memória."""
        if self.model is not None:
            logger.info("Modelo de reconhecimento de fala já carregado, pulando.")
            return
        
        logger.info(f"Carregando modelo de reconhecimento de fala: {self.model_name}")
        
        try:
            # Carregar o processador
            self.processor = WhisperProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Configurar atributos para carregamento do modelo
            kwargs = {
                "torch_dtype": self.dtype,
                "cache_dir": self.cache_dir
            }
            
            # Adicionar configuração de flash attention se disponível e solicitada
            if self.use_flash_attention and self.device == "cuda":
                try:
                    kwargs["use_flash_attention_2"] = True
                    logger.info("Flash Attention 2 habilitado.")
                except Exception as e:
                    logger.warning(f"Não foi possível habilitar Flash Attention: {str(e)}")
            
            # Carregar o modelo
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                **kwargs
            ).to(self.device)
            
            logger.info("Modelo de reconhecimento de fala carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de reconhecimento de fala: {str(e)}")
            raise
            
    def unload_model(self):
        """Libera o modelo da memória."""
        if self.model is not None:
            del self.processor
            del self.model
            self.processor = None
            self.model = None
            
            # Limpar cache CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            logger.info("Modelo de reconhecimento de fala descarregado da memória.")
            
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        sampling_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
        return_timestamps: bool = False,
        chunk_length_s: Optional[float] = None,
        batch_size: int = 16,
        return_all_segments: bool = True
    ) -> Dict[str, Any]:
        """
        Transcreve áudio para texto.
        
        Args:
            audio: Caminho para arquivo de áudio ou array NumPy com dados de áudio
            sampling_rate: Taxa de amostragem do áudio
            language: Código do idioma para transcrição (opcional)
            task: Tarefa a ser realizada ("transcribe" ou "translate")
            return_timestamps: Se True, inclui timestamps na transcrição
            chunk_length_s: Duração dos chunks em segundos para processamento em lotes
            batch_size: Tamanho do lote para processamento em chunks
            return_all_segments: Se True, retorna todos os segmentos transcritos
            
        Returns:
            Dicionário com resultado da transcrição
        """
        # Carregar o modelo se ainda não foi carregado
        if self.model is None:
            self.load_model()
            
        start_time = time.time()
        
        try:
            # Carregar áudio se for um caminho de arquivo
            if isinstance(audio, str):
                logger.info(f"Carregando áudio do arquivo: {audio}")
                audio_array, orig_sampling_rate = librosa.load(audio, sr=None)
                
                # Resample se necessário
                if orig_sampling_rate != sampling_rate:
                    logger.info(f"Resample de {orig_sampling_rate}Hz para {sampling_rate}Hz")
                    audio_array = librosa.resample(
                        audio_array, 
                        orig_sr=orig_sampling_rate, 
                        target_sr=sampling_rate
                    )
            else:
                # Assumir que é um array NumPy
                audio_array = audio
                
            logger.info(f"Transcrevendo áudio de {len(audio_array) / sampling_rate:.2f} segundos")
            
            # Processar áudio
            input_features = self.processor(
                audio_array, 
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Configurar parâmetros de geração
            generation_config = {}
            
            # Adicionar configuração de idioma se especificado
            if language:
                generation_config["language"] = language
                
            # Configurar tarefa (tradução ou transcrição)
            generation_config["task"] = task
            
            # Configurar timestamps
            generation_config["return_timestamps"] = return_timestamps
            
            # Configurar processamento em chunks se necessário
            is_chunked = chunk_length_s is not None
            
            if is_chunked:
                # Calcular tamanho do chunk em samples
                chunk_len = int(chunk_length_s * sampling_rate)
                
                # Processar em chunks
                all_segments = []
                
                for i in range(0, len(audio_array), chunk_len):
                    chunk = audio_array[i:i + chunk_len]
                    chunk_features = self.processor(
                        chunk, 
                        sampling_rate=sampling_rate,
                        return_tensors="pt"
                    ).input_features.to(self.device)
                    
                    with torch.no_grad():
                        result = self.model.generate(
                            chunk_features,
                            **generation_config
                        )
                        
                    # Decodificar resultado
                    decoded = self.processor.batch_decode(
                        result, 
                        skip_special_tokens=True
                    )
                    
                    # Ajustar timestamps se necessário
                    if return_timestamps and len(decoded) > 0:
                        # TODO: Ajustar timestamps para considerar a posição do chunk
                        pass
                        
                    all_segments.extend(decoded)
            else:
                # Processar o áudio completo
                with torch.no_grad():
                    result = self.model.generate(
                        input_features,
                        **generation_config
                    )
                    
                # Decodificar resultado
                transcription = self.processor.batch_decode(
                    result, 
                    skip_special_tokens=True
                )
                
                # Detectar segmentos se o modelo os gerar
                if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "to_dict_with_timestamps"):
                    all_segments = self.processor.tokenizer.to_dict_with_timestamps(result[0])
                else:
                    # Usar resultado simples se não houver segmentação
                    all_segments = [{"text": t} for t in transcription]
                    
            # Juntar todos os segmentos em um texto completo
            full_text = " ".join([segment["text"] for segment in all_segments])
                    
            elapsed_time = time.time() - start_time
            logger.info(f"Transcrição concluída em {elapsed_time:.2f} segundos")
            
            # Construir resultado
            result = {
                "text": full_text.strip(),
                "language": language or "auto-detected",
                "duration": len(audio_array) / sampling_rate,
                "processing_time": elapsed_time
            }
            
            if return_all_segments:
                result["segments"] = all_segments
                
            return result
            
        except Exception as e:
            logger.error(f"Erro ao transcrever áudio: {str(e)}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo carregado."""
        loaded_status = self.model is not None
        
        memory_used = None
        if self.device == "cuda" and torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
        return {
            "model_name": self.model_name,
            "status": "loaded" if loaded_status else "not_loaded",
            "device": self.device,
            "dtype": str(self.dtype),
            "flash_attention": self.use_flash_attention,
            "memory_used_gb": memory_used if loaded_status else None
        }
