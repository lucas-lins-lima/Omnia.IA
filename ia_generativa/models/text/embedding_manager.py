"""
Embedding Manager - Responsável por carregar e utilizar modelos de embedding de texto.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Any
import numpy as np
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Gerencia o carregamento e execução de modelos de embedding de texto.
    """
    
    def __init__(
        self, 
        model_name_or_path: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Inicializa o gerenciador de embeddings.
        
        Args:
            model_name_or_path: ID do modelo no HuggingFace ou caminho local
            device: Dispositivo para execução (cuda, cpu, etc.)
            normalize_embeddings: Se True, normaliza os embeddings para norma unitária
        """
        self.model_name = model_name_or_path
        
        # Determinar o dispositivo automaticamente se não for especificado
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"Inicializando EmbeddingManager com modelo: {self.model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Configurar cache para os modelos
        os.environ["TRANSFORMERS_CACHE"] = os.path.join("data", "models_cache")
        
        # Inicialmente não carregamos o modelo, apenas quando necessário
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Carrega o modelo e tokenizador na memória."""
        if self.model is not None:
            logger.info("Modelo de embedding já carregado, pulando.")
            return
        
        logger.info(f"Carregando modelo de embedding: {self.model_name}")
            
        # Carregar o tokenizador
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )
            logger.info("Tokenizador carregado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar tokenizador para embeddings: {str(e)}")
            raise
            
        # Carregar o modelo
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name
            ).to(self.device)
            
            # Colocar em modo de avaliação (desativa dropout, etc.)
            self.model.eval()
            logger.info("Modelo de embedding carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de embedding: {str(e)}")
            raise
            
    def unload_model(self):
        """Libera o modelo da memória."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("Modelo de embedding descarregado da memória.")
            
    def get_embeddings(
        self, 
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Gera embeddings para textos.
        
        Args:
            texts: Um texto ou lista de textos para gerar embeddings
            batch_size: Tamanho do lote para processamento em batch
            
        Returns:
            Array NumPy contendo os embeddings dos textos
        """
        # Carregar o modelo se ainda não foi carregado
        if self.model is None:
            self.load_model()
            
        # Converter texto único para lista
        if isinstance(texts, str):
            texts = [texts]
            
        logger.info(f"Gerando embeddings para {len(texts)} textos")
        
        # Processar em batches para evitar OOM em grandes volumes
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenizar
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512  # Ajustar conforme necessário
            ).to(self.device)
            
            # Gerar embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Obter o embedding da saída (usando a média dos tokens ou [CLS])
            # Estratégia pode variar dependendo do modelo
            attention_mask = inputs["attention_mask"]
            
            # Média dos embeddings dos tokens (excluindo padding)
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            
            # Normalização opcional (geralmente recomendada para similaridade de cosseno)
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            # Converter para NumPy e adicionar ao resultado
            all_embeddings.append(embeddings.cpu().numpy())
            
        # Concatenar todos os batches
        final_embeddings = np.vstack(all_embeddings)
        
        logger.debug(f"Embeddings gerados com formato: {final_embeddings.shape}")
        return final_embeddings
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Calcula a média dos embeddings dos tokens, considerando a máscara de atenção.
        """
        # Expandir a máscara para a dimensão de embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Somar os embeddings dos tokens, multiplicados pela máscara
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Calcular a soma da máscara para obter o número real de tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Dividir para obter a média
        return sum_embeddings / sum_mask
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo de embedding carregado."""
        if self.model is None:
            return {"status": "not_loaded", "model_name": self.model_name}
        
        memory_used = None
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "memory_used_gb": memory_used
        }
