"""
Speech Synthesis Manager - Responsável por carregar e utilizar modelos de síntese de fala.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import logging
import time
import tempfile
import re

logger = logging.getLogger(__name__)

class SpeechSynthesisManager:
    """
    Gerencia o carregamento e execução de modelos de síntese de fala (TTS).
    """
    
    def __init__(
        self,
        model_name_or_path: str = "suno/bark-small",
        model_type: str = "bark",  # "bark", "vits", "tts-1"
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Inicializa o gerenciador de síntese de fala.
        
        Args:
            model_name_or_path: ID do modelo no HuggingFace ou caminho local
            model_type: Tipo de modelo de TTS
            device: Dispositivo para execução (cuda, cpu, etc.)
            dtype: Tipo de dados para o modelo (float16, float32)
            cache_dir: Diretório para cache de modelos
        """
        self.model_name = model_name_or_path
        self.model_type = model_type.lower()
        
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
        
        logger.info(f"Inicializando SpeechSynthesisManager com modelo {self.model_type}: {self.model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Inicialmente não carregamos os modelos, apenas quando necessário
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Carrega o modelo de síntese de fala na memória."""
        if self.model is not None:
            logger.info("Modelo de síntese de fala já carregado, pulando.")
            return
        
        logger.info(f"Carregando modelo de síntese de fala {self.model_type}: {self.model_name}")
        
        try:
            # Carregar modelo de acordo com o tipo
            if self.model_type == "bark":
                self._load_bark_model()
            elif self.model_type == "vits":
                self._load_vits_model()
            else:
                raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
                
            logger.info("Modelo de síntese de fala carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de síntese de fala: {str(e)}")
            raise
            
    def _load_bark_model(self):
        """Carrega modelo Bark para síntese de fala."""
        try:
            from transformers import BarkProcessor, BarkModel
            
            # Carregar o processador
            self.processor = BarkProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Carregar o modelo
            self.model = BarkModel.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            logger.info("Modelo Bark carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo Bark: {str(e)}")
            raise
            
    def _load_vits_model(self):
        """Carrega modelo VITS para síntese de fala."""
        try:
            # Note: A implementação pode variar dependendo da versão específica do VITS
            # Exemplo para uma versão específica do VITS
            import sys
            import importlib
            
            # Configurar caminho para o modelo VITS (ajustar conforme necessário)
            vits_path = os.path.join(os.path.dirname(__file__), "../../external/vits")
            if vits_path not in sys.path:
                sys.path.append(vits_path)
                
            # Importar módulos VITS
            vits_utils = importlib.import_module("utils")
            vits_commons = importlib.import_module("commons")
            vits_models = importlib.import_module("models")
            
            # Carregar configuração do modelo
            hps = vits_utils.get_hparams_from_file(f"{self.model_name}/config.json")
            
            # Carregar modelo
            self.model = vits_models.SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model
            )
            
            # Carregar pesos do modelo
            state_dict = torch.load(f"{self.model_name}/model.pth", map_location=self.device)
            self.model.load_state_dict(state_dict['net_g'])
            self.model.eval().to(self.device)
            
            # Armazenar hiperparâmetros para uso posterior
            self.hps = hps
            
            logger.info("Modelo VITS carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo VITS: {str(e)}")
            raise
            
    def unload_model(self):
        """Libera o modelo da memória."""
        if self.model is not None:
            del self.model
            if self.processor is not None:
                del self.processor
            self.model = None
            self.processor = None
            
            # Limpar cache CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            logger.info("Modelo de síntese de fala descarregado da memória.")
            
    def synthesize(
        self,
        text: str,
        voice_preset: Optional[str] = None,
        language: Optional[str] = None,
        speaker_id: Optional[int] = None,
        sampling_rate: int = 24000,
        output_path: Optional[str] = None,
        return_waveform: bool = True
    ) -> Union[np.ndarray, str]:
        """
        Sintetiza fala a partir de texto.
        
        Args:
            text: Texto para sintetizar
            voice_preset: Preset de voz (específico para Bark)
            language: Código do idioma (específico para Bark/VITS)
            speaker_id: ID do falante (específico para VITS)
            sampling_rate: Taxa de amostragem desejada para o áudio
            output_path: Caminho para salvar o áudio (opcional)
            return_waveform: Se True, retorna o array de formas de onda
            
        Returns:
            Array NumPy com formas de onda ou caminho do arquivo salvo
        """
        # Carregar o modelo se ainda não foi carregado
        if self.model is None:
            self.load_model()
            
        start_time = time.time()
        
        try:
            # Processar texto para remover problemas comuns
            text = self._preprocess_text(text)
            
            logger.info(f"Sintetizando fala para texto de {len(text)} caracteres")
            
            # Sintetizar de acordo com o tipo de modelo
            if self.model_type == "bark":
                audio_array = self._synthesize_bark(
                    text,
                    voice_preset=voice_preset,
                    language=language,
                    sampling_rate=sampling_rate
                )
            elif self.model_type == "vits":
                audio_array = self._synthesize_vits(
                    text,
                    speaker_id=speaker_id,
                    language=language,
                    sampling_rate=sampling_rate
                )
            else:
                raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
                
            elapsed_time = time.time() - start_time
            logger.info(f"Síntese concluída em {elapsed_time:.2f} segundos")
            
            # Salvar áudio se solicitado
            if output_path:
                self._save_audio(audio_array, output_path, sampling_rate)
                
                # Se não for necessário retornar o array, retorna apenas o caminho
                if not return_waveform:
                    return output_path
                    
            return audio_array
            
        except Exception as e:
            logger.error(f"Erro ao sintetizar fala: {str(e)}")
            raise
            
    def _synthesize_bark(
        self,
        text: str,
        voice_preset: Optional[str] = None,
        language: Optional[str] = None,
        sampling_rate: int = 24000
    ) -> np.ndarray:
        """
        Sintetiza fala usando o modelo Bark.
        
        Args:
            text: Texto para sintetizar
            voice_preset: Preset de voz
            language: Código do idioma
            sampling_rate: Taxa de amostragem desejada
            
        Returns:
            Array NumPy com formas de onda
        """
        # Configurar inputs para o processador
        inputs = {}
        
        # Adicionar texto
        inputs["text"] = text
        
        # Adicionar preset de voz se fornecido
        if voice_preset:
            inputs["voice_preset"] = voice_preset
            
        # Processar inputs
        inputs = self.processor(**inputs)
        
        # Gerar áudio
        with torch.no_grad():
            output = self.model.generate(**inputs.to(self.device))
            
        # Obter formas de onda
        audio_array = output.cpu().numpy().squeeze()
        
        # Resample se necessário
        if sampling_rate != self.model.generation_config.sample_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=self.model.generation_config.sample_rate, 
                target_sr=sampling_rate
            )
            
        return audio_array
        
    def _synthesize_vits(
        self,
        text: str,
        speaker_id: Optional[int] = 0,
        language: Optional[str] = "en",
        sampling_rate: int = 22050
    ) -> np.ndarray:
        """
        Sintetiza fala usando o modelo VITS.
        
        Args:
            text: Texto para sintetizar
            speaker_id: ID do falante
            language: Código do idioma
            sampling_rate: Taxa de amostragem desejada
            
        Returns:
            Array NumPy com formas de onda
        """
        # Esta implementação pode variar significativamente dependendo da versão específica do VITS
        try:
            # Importar módulos VITS necessários
            import sys
            import importlib
            
            vits_path = os.path.join(os.path.dirname(__file__), "../../external/vits")
            if vits_path not in sys.path:
                sys.path.append(vits_path)
                
            vits_text = importlib.import_module("text")
            
            # Processar texto para entrada do modelo
            if hasattr(vits_text, "text_to_sequence"):
                # Converter texto para sequência de IDs
                if language == "zh":
                    # Chinês
                    from vits_text.mandarin import text_to_sequence
                elif language == "ja":
                    # Japonês
                    from vits_text.japanese import text_to_sequence
                else:
                    # Inglês (padrão)
                    from vits_text.english import text_to_sequence
                    
                sequence = text_to_sequence(text, self.hps.data.text_cleaners)
                sequence = torch.LongTensor(sequence).to(self.device)
                
                # Adicionar dimensão de lote
                sequence = sequence.unsqueeze(0)
                
                # Inferência
                with torch.no_grad():
                    # Para VITS multi-speaker
                    if hasattr(self.model, "speaker_embed"):
                        sid = torch.LongTensor([speaker_id]).to(self.device)
                        audio = self.model.infer(sequence, sid=sid)[0, 0].data.cpu().numpy()
                    else:
                        # Para VITS single-speaker
                        audio = self.model.infer(sequence)[0, 0].data.cpu().numpy()
                        
                # Resample se necessário
                if sampling_rate != self.hps.data.sampling_rate:
                    import librosa
                    audio = librosa.resample(
                        audio, 
                        orig_sr=self.hps.data.sampling_rate, 
                        target_sr=sampling_rate
                    )
                    
                return audio
                
            else:
                logger.error("Função text_to_sequence não encontrada no módulo text")
                raise ValueError("Função text_to_sequence não encontrada")
                
        except Exception as e:
            logger.error(f"Erro na síntese VITS: {str(e)}")
            raise
            
    def _preprocess_text(self, text: str) -> str:
        """
        Pré-processa o texto para síntese de fala.
        
        Args:
            text: Texto original
            
        Returns:
            Texto pré-processado
        """
        # Remover múltiplas quebras de linha
        text = re.sub(r'\n+', '\n', text)
        
        # Remover múltiplos espaços
        text = re.sub(r'\s+', ' ', text)
        
        # Remover URLs se não for necessário pronunciá-las
        # text = re.sub(r'https?://\S+', '', text)
        
        # Expandir abreviações comuns
        abbreviations = {
            "Dr.": "Doutor",
            "Mr.": "Mister",
            "Mrs.": "Missus",
            "Ms.": "Miss",
            "etc.": "etcetera",
            "e.g.": "for example",
            "i.e.": "that is",
            # Adicionar mais conforme necessário
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
            
        return text.strip()
        
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
            
    def get_available_voices(self) -> List[str]:
        """
        Retorna lista de vozes disponíveis para o modelo atual.
        
        Returns:
            Lista de identificadores de vozes
        """
        # Implementação depende do tipo de modelo
        if self.model_type == "bark":
            try:
                # Para Bark, retornar presets de voz disponíveis
                if self.processor is None:
                    self.load_model()
                    
                # Obter presets de vozes disponíveis
                from transformers.models.bark.voice_presets import VOICE_PRESETS
                return list(VOICE_PRESETS.keys())
                
            except Exception as e:
                logger.error(f"Erro ao obter vozes disponíveis: {str(e)}")
                return ["v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3"]
                
        elif self.model_type == "vits":
            # Para VITS, depende da implementação específica
            try:
                if hasattr(self, "hps") and hasattr(self.hps, "speakers"):
                    return list(self.hps.speakers.keys())
                else:
                    # Retornar IDs genéricos
                    return [f"speaker_{i}" for i in range(4)]
            except Exception:
                return ["speaker_0", "speaker_1", "speaker_2", "speaker_3"]
                
        return []
        
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo carregado."""
        loaded_status = self.model is not None
        
        memory_used = None
        if self.device == "cuda" and torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "status": "loaded" if loaded_status else "not_loaded",
            "device": self.device,
            "dtype": str(self.dtype),
            "voices_available": len(self.get_available_voices()) if loaded_status else 0,
            "memory_used_gb": memory_used if loaded_status else None
        }
