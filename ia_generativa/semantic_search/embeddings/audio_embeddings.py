"""
Audio Embeddings - Geração de embeddings para conteúdo de áudio.
"""

import os
import logging
import time
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
import torch
import io
import base64
import tempfile

from transformers import AutoFeatureExtractor, AutoModel
import torchaudio

logger = logging.getLogger(__name__)

class AudioEmbeddingGenerator:
    """Gerador de embeddings para áudio."""
    
    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-base-960h",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        target_sr: int = 16000
    ):
        """
        Inicializa o gerador de embeddings para áudio.
        
        Args:
            model_name: Nome do modelo de embeddings
            device: Dispositivo para execução (cuda, cpu)
            cache_dir: Diretório para cache de modelos
            target_sr: Taxa de amostragem alvo
        """
        self.model_name = model_name
        self.target_sr = target_sr
        
        # Determinar o dispositivo automaticamente se não for especificado
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Configurar cache para os modelos
        self.cache_dir = cache_dir or os.path.join("data", "models_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.embedding_dim = None
        
        logger.info(f"Inicializando AudioEmbeddingGenerator com modelo {model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Carregar modelo
        self._load_model()
        
    def _load_model(self):
        """Carrega o modelo de embeddings para áudio."""
        try:
            start_time = time.time()
            
            # Carregar processador e modelo
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            # Determinar dimensão dos embeddings com uma amostra vazia
            dummy_input = torch.zeros(1, 16000).to(self.device)
            with torch.no_grad():
                outputs = self.model(dummy_input)
                
            # A dimensão pode variar dependendo do modelo
            if hasattr(outputs, "last_hidden_state"):
                self.embedding_dim = outputs.last_hidden_state.shape[-1]
            else:
                # Fallback para alguma outra saída
                output_keys = outputs.keys()
                if len(output_keys) > 0:
                    first_key = list(output_keys)[0]
                    self.embedding_dim = outputs[first_key].shape[-1]
                else:
                    # Sem opções claras, usar um valor fixo
                    self.embedding_dim = 768
                    logger.warning(f"Não foi possível determinar automaticamente a dimensão dos embeddings. Usando {self.embedding_dim}.")
                    
            logger.info(f"Modelo carregado em {time.time() - start_time:.2f} segundos")
            logger.info(f"Dimensão dos embeddings: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de embeddings para áudio: {str(e)}")
            raise
            
    def generate_embeddings(
        self, 
        audio_data: Union[torch.Tensor, np.ndarray, str, List[Union[torch.Tensor, np.ndarray, str]]],
        max_duration: float = 30.0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Gera embeddings para áudio.
        
        Args:
            audio_data: Dados de áudio (tensor, array, base64 ou lista desses)
            max_duration: Duração máxima em segundos
            normalize: Se True, normaliza os embeddings para norma unitária
            
        Returns:
            Array NumPy com embeddings (shape: [n_audios, embedding_dim])
        """
        # Processar entrada
        processed_audio, sample_rates = self._process_input(audio_data)
        
        try:
            start_time = time.time()
            embeddings = []
            
            # Processar cada áudio
            for i, (audio, sr) in enumerate(zip(processed_audio, sample_rates)):
                # Resample se necessário
                if sr != self.target_sr:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                    audio = resampler(audio)
                    
                # Garantir que o áudio é mono
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                    
                # Limitar duração
                max_samples = int(max_duration * self.target_sr)
                if audio.shape[1] > max_samples:
                    audio = audio[:, :max_samples]
                    
                # Mover para o dispositivo
                audio = audio.to(self.device)
                
                # Extrair embeddings
                with torch.no_grad():
                    outputs = self.model(audio)
                    
                # Determinar qual saída usar e como agregá-la
                if hasattr(outputs, "last_hidden_state"):
                    # Calcular média ao longo da dimensão temporal
                    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                else:
                    # Fallback para alguma outra saída
                    output_keys = outputs.keys()
                    if len(output_keys) > 0:
                        first_key = list(output_keys)[0]
                        # Reduzir para uma representação vetorial única
                        if outputs[first_key].dim() > 2:
                            emb = outputs[first_key].mean(dim=1).cpu().numpy()
                        else:
                            emb = outputs[first_key].cpu().numpy()
                    else:
                        raise ValueError("O modelo não produziu saídas utilizáveis")
                        
                embeddings.append(emb)
                
            # Concatenar resultados
            all_embeddings = np.vstack(embeddings)
            
            # Normalizar se solicitado
            if normalize:
                norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
                all_embeddings = all_embeddings / np.maximum(norms, 1e-12)
                
            logger.info(f"Gerados embeddings para {len(processed_audio)} áudios em {time.time() - start_time:.2f} segundos")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings de áudio: {str(e)}")
            raise
            
    def _process_input(
        self, 
        audio_data: Union[torch.Tensor, np.ndarray, str, List[Union[torch.Tensor, np.ndarray, str]]]
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Processa a entrada, convertendo para lista de tensores PyTorch.
        
        Args:
            audio_data: Dados de áudio (tensor, array, base64 ou lista desses)
            
        Returns:
            Tupla (lista de tensores de áudio, lista de taxas de amostragem)
        """
        # Converter para lista se for um único item
        if not isinstance(audio_data, list):
            audio_data = [audio_data]
            
        processed_audio = []
        sample_rates = []
        
        for audio in audio_data:
            if isinstance(audio, torch.Tensor):
                # Já é um tensor PyTorch
                processed_audio.append(audio)
                # Assumir taxa de amostragem padrão
                sample_rates.append(self.target_sr)
                
            elif isinstance(audio, np.ndarray):
                # Converter array NumPy para tensor PyTorch
                audio_tensor = torch.from_numpy(audio).float()
                
                # Garantir formato correto (canais, amostras)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                    
                processed_audio.append(audio_tensor)
                # Assumir taxa de amostragem padrão
                sample_rates.append(self.target_sr)
                
            elif isinstance(audio, str):
                # String base64 ou caminho de arquivo
                if audio.startswith("data:audio"):
                    # Base64 com prefixo data URI
                    _, audio_str = audio.split(';base64,')
                    audio_bytes = base64.b64decode(audio_str)
                    
                    # Salvar em arquivo temporário
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file.write(audio_bytes)
                        temp_path = temp_file.name
                    
                    try:
                        # Carregar com torchaudio
                        waveform, sr = torchaudio.load(temp_path)
                        processed_audio.append(waveform)
                        sample_rates.append(sr)
                    finally:
                        # Limpar arquivo temporário
                        os.unlink(temp_path)
                        
                elif audio.startswith(("http://", "https://")):
                    # URL de áudio - precisaria de um pacote como requests
                    raise ValueError("URLs de áudio não são suportadas diretamente. Baixe o áudio primeiro.")
                else:
                    # Assumir que é um caminho de arquivo ou base64 sem prefixo
                    try:
                        # Tentar carregar como caminho de arquivo
                        waveform, sr = torchaudio.load(audio)
                        processed_audio.append(waveform)
                        sample_rates.append(sr)
                    except:
                        try:
                            # Tentar decodificar como base64
                            audio_bytes = base64.b64decode(audio)
                            
                            # Salvar em arquivo temporário
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                temp_file.write(audio_bytes)
                                temp_path = temp_file.name
                            
                            try:
                                # Carregar com torchaudio
                                waveform, sr = torchaudio.load(temp_path)
                                processed_audio.append(waveform)
                                sample_rates.append(sr)
                            finally:
                                # Limpar arquivo temporário
                                os.unlink(temp_path)
                                
                        except:
                            logger.warning(f"Não foi possível processar o áudio: {audio[:50]}...")
                            continue
            else:
                logger.warning(f"Tipo de entrada não suportado: {type(audio)}")
                continue
                
        return processed_audio, sample_rates
