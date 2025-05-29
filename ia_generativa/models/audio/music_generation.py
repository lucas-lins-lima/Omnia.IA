"""
Music Generation Manager - Responsável por carregar e utilizar modelos de geração de música.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class MusicGenerationManager:
    """
    Gerencia o carregamento e execução de modelos de geração de música.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "facebook/musicgen-small",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Inicializa o gerenciador de geração de música.
        
        Args:
            model_name_or_path: ID do modelo no HuggingFace ou caminho local
            device: Dispositivo para execução (cuda, cpu, etc.)
            dtype: Tipo de dados para o modelo (float16, float32)
            cache_dir: Diretório para cache de modelos
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
            
        # Configurar cache para os modelos
        self.cache_dir = cache_dir or os.path.join("data", "models_cache")
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        
        logger.info(f"Inicializando MusicGenerationManager com modelo: {self.model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Inicialmente não carregamos os modelos, apenas quando necessário
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Carrega o modelo de geração de música na memória."""
        if self.model is not None:
            logger.info("Modelo de geração de música já carregado, pulando.")
            return
        
        logger.info(f"Carregando modelo de geração de música: {self.model_name}")
        
        try:
            # Usar MusicGen via Transformers
            from transformers import MusicgenProcessor, MusicgenForConditionalGeneration
            
            # Carregar o processador
            self.processor = MusicgenProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Carregar o modelo
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            logger.info("Modelo de geração de música carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de geração de música: {str(e)}")
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
                
            logger.info("Modelo de geração de música descarregado da memória.")
            
    def generate_music(
        self,
        prompt: str,
        duration: float = 10.0,
        sampling_rate: int = 32000,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        classifier_free_guidance: float = 3.0,
        output_path: Optional[str] = None,
        return_waveform: bool = True,
        audio_conditioned: Optional[np.ndarray] = None,
        audio_conditioned_sr: Optional[int] = None
    ) -> Union[np.ndarray, str]:
        """
        Gera música a partir de uma descrição textual.
        
        Args:
            prompt: Descrição textual da música a ser gerada
            duration: Duração da música em segundos
            sampling_rate: Taxa de amostragem do áudio gerado
            top_k: Parâmetro de filtragem top-k
            top_p: Parâmetro de filtragem nucleus
            temperature: Temperatura para geração
            classifier_free_guidance: Escala de orientação para prompt
            output_path: Caminho para salvar o áudio (opcional)
            return_waveform: Se True, retorna o array de formas de onda
            audio_conditioned: Áudio de referência para condicionamento (opcional)
            audio_conditioned_sr: Taxa de amostragem do áudio de referência
            
        Returns:
            Array NumPy com formas de onda ou caminho do arquivo salvo
        """
        # Carregar o modelo se ainda não foi carregado
        if self.model is None:
            self.load_model()
            
        start_time = time.time()
        
        try:
            logger.info(f"Gerando música para prompt: {prompt}")
            
            # Configurar parâmetros de geração
            max_length_seconds = min(duration, 30.0)  # Limitar duração para evitar OOM
            
            # Preparar condicionamento de áudio (se fornecido)
            audio_conditioning = None
            if audio_conditioned is not None and audio_conditioned_sr is not None:
                # Resample se necessário
                if audio_conditioned_sr != sampling_rate:
                    import librosa
                    audio_conditioned = librosa.resample(
                        audio_conditioned, 
                        orig_sr=audio_conditioned_sr, 
                        target_sr=sampling_rate
                    )
                    
                # Converter para tensor
                audio_conditioning = torch.tensor(audio_conditioned).unsqueeze(0).to(self.device)
                
            # Processar o prompt
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Adicionar condicionamento de áudio (se fornecido)
            if audio_conditioning is not None:
                inputs["audio_conditioning"] = audio_conditioning
                
            # Gerar música
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_length_seconds * self.model.config.audio_encoder.frame_rate),
                    do_sample=True,
                    guidance_scale=classifier_free_guidance,
                    top_k=top_k,
                    top_p=top_p if top_p > 0 else None,
                    temperature=temperature
                )
                
            # Converter para numpy
            audio_array = audio_values[0].cpu().numpy()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Música gerada em {elapsed_time:.2f} segundos, duração: {len(audio_array) / sampling_rate:.2f}s")
            
            # Salvar áudio se solicitado
            if output_path:
                self._save_audio(audio_array, output_path, sampling_rate)
                
                # Se não for necessário retornar o array, retorna apenas o caminho
                if not return_waveform:
                    return output_path
                    
            return audio_array
            
        except Exception as e:
            logger.error(f"Erro ao gerar música: {str(e)}")
            raise
            
    def _save_audio(self, audio_array: np.ndarray, output_path: str, sampling_rate: int):
        """
        Salva array de áudio em um arquivo.
        
        Args:
            audio_array: Array com formas de onda
            output_path: Caminho para salvar o arquivo
            sampling_rate: Taxa de amostragem
        """
        try:
            import soundfile as sf
            
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Salvar arquivo
            sf.write(output_path, audio_array, sampling_rate)
            logger.info(f"Áudio salvo em: {output_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar áudio: {str(e)}")
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
            "memory_used_gb": memory_used if loaded_status else None
        }
