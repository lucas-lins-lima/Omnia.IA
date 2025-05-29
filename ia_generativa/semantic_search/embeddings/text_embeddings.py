"""
Text Embeddings - Geração de embeddings para conteúdo textual.
"""

import os
import logging
import time
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class TextEmbeddingGenerator:
    """Gerador de embeddings para texto."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_sentence_transformers: bool = True
    ):
        """
        Inicializa o gerador de embeddings.
        
        Args:
            model_name: Nome do modelo de embeddings
            device: Dispositivo para execução (cuda, cpu)
            cache_dir: Diretório para cache de modelos
            use_sentence_transformers: Se True, usa a biblioteca SentenceTransformers
        """
        self.model_name = model_name
        
        # Determinar o dispositivo automaticamente se não for especificado
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Configurar cache para os modelos
        self.cache_dir = cache_dir or os.path.join("data", "models_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.use_sentence_transformers = use_sentence_transformers
        self.embedding_dim = None
        
        logger.info(f"Inicializando TextEmbeddingGenerator com modelo {model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Carregar modelo
        self._load_model()
        
    def _load_model(self):
        """Carrega o modelo de embeddings."""
        try:
            start_time = time.time()
            
            if self.use_sentence_transformers:
                # Usar SentenceTransformers (mais fácil de usar)
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=self.device
                )
                # Obter dimensão dos embeddings
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                # Usar HuggingFace diretamente (mais flexível)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                ).to(self.device)
                
                # Executar inferência com um texto vazio para obter a dimensão
                with torch.no_grad():
                    inputs = self.tokenizer("", return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    self.embedding_dim = outputs.last_hidden_state.shape[-1]
                    
            logger.info(f"Modelo carregado em {time.time() - start_time:.2f} segundos")
            logger.info(f"Dimensão dos embeddings: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de embeddings: {str(e)}")
            raise
            
    def generate_embeddings(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Gera embeddings para textos.
        
        Args:
            texts: Texto ou lista de textos
            batch_size: Tamanho do lote para processamento
            normalize: Se True, normaliza os embeddings para norma unitária
            
        Returns:
            Array NumPy com embeddings (shape: [n_texts, embedding_dim])
        """
        # Garantir que a entrada seja uma lista
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            start_time = time.time()
            
            if self.use_sentence_transformers:
                # Usar SentenceTransformers
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=len(texts) > 100,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize
                )
            else:
                # Usar HuggingFace diretamente
                embeddings = []
                
                # Processar em lotes
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    
                    # Tokenizar
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Gerar embeddings
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        
                    # Usar mean pooling (média dos embeddings dos tokens)
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    
                    # Mascarar tokens de padding
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    
                    # Calcular a média de tokens não mascarados
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Normalizar se solicitado
                    if normalize:
                        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                        
                    # Adicionar ao resultado
                    embeddings.append(batch_embeddings.cpu().numpy())
                    
                # Concatenar resultados dos lotes
                embeddings = np.vstack(embeddings)
                
            logger.info(f"Gerados embeddings para {len(texts)} textos em {time.time() - start_time:.2f} segundos")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings: {str(e)}")
            raise
            
    def compute_similarity(
        self, 
        embeddings1: np.ndarray, 
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Calcula a similaridade coseno entre embeddings.
        
        Args:
            embeddings1: Primeiro conjunto de embeddings
            embeddings2: Segundo conjunto de embeddings
            
        Returns:
            Matriz de similaridade (shape: [len(embeddings1), len(embeddings2)])
        """
        # Normalizar embeddings (se ainda não estiverem normalizados)
        norm_emb1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm_emb2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Calcular similaridade (produto escalar de vetores normalizados)
        return np.dot(norm_emb1, norm_emb2.T)

class MultilingualEmbeddingGenerator(TextEmbeddingGenerator):
    """Gerador de embeddings multilingue para texto."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Inicializa o gerador de embeddings multilingue.
        
        Args:
            model_name: Nome do modelo de embeddings multilingue
            device: Dispositivo para execução (cuda, cpu)
            cache_dir: Diretório para cache de modelos
        """
        super().__init__(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            use_sentence_transformers=True  # SentenceTransformers é mais fácil para multilíngue
        )
        
        logger.info(f"Inicializado gerador de embeddings multilingue")
