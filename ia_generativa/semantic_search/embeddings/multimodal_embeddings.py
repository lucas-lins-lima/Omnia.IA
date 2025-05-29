"""
Multimodal Embeddings - Geração de embeddings para conteúdo multimodal.
"""

import os
import logging
import time
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
import torch
from PIL import Image
import io
import base64

from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

class MultimodalEmbeddingGenerator:
    """Gerador de embeddings para conteúdo multimodal (texto e imagem)."""
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Inicializa o gerador de embeddings multimodal.
        
        Args:
            model_name: Nome do modelo CLIP
            device: Dispositivo para execução (cuda, cpu)
            cache_dir: Diretório para cache de modelos
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
        
        self.text_embedding_dim = None
        self.image_embedding_dim = None
        
        logger.info(f"Inicializando MultimodalEmbeddingGenerator com modelo {model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Carregar modelo
        self._load_model()
        
    def _load_model(self):
        """Carrega o modelo CLIP."""
        try:
            start_time = time.time()
            
            # Carregar processador e modelo CLIP
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            # Obter dimensões dos embeddings
            self.text_embedding_dim = self.model.config.text_config.hidden_size
            self.image_embedding_dim = self.model.config.vision_config.hidden_size
            
            logger.info(f"Modelo carregado em {time.time() - start_time:.2f} segundos")
            logger.info(f"Dimensão dos embeddings de texto: {self.text_embedding_dim}")
            logger.info(f"Dimensão dos embeddings de imagem: {self.image_embedding_dim}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo CLIP: {str(e)}")
            raise
            
    def generate_text_embeddings(
        self, 
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Gera embeddings para textos usando CLIP.
        
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
            embeddings = []
            
            # Processar em lotes
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Processar textos
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Gerar embeddings
                with torch.no_grad():
                    outputs = self.model.get_text_features(**inputs)
                    batch_embeddings = outputs.cpu().numpy()
                    
                # Normalizar se solicitado
                if normalize:
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / np.maximum(norms, 1e-12)
                    
                embeddings.append(batch_embeddings)
                
            # Concatenar resultados dos lotes
            all_embeddings = np.vstack(embeddings)
            
            logger.info(f"Gerados embeddings para {len(texts)} textos em {time.time() - start_time:.2f} segundos")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings de texto com CLIP: {str(e)}")
            raise
            
    def generate_image_embeddings(
        self, 
        images: Union[Image.Image, List[Image.Image], str, List[str]],
        batch_size: int = 16,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Gera embeddings para imagens usando CLIP.
        
        Args:
            images: Imagem(ns) como objetos PIL.Image ou strings base64
            batch_size: Tamanho do lote para processamento
            normalize: Se True, normaliza os embeddings para norma unitária
            
        Returns:
            Array NumPy com embeddings (shape: [n_images, embedding_dim])
        """
        # Processar entrada
        processed_images = self._process_images(images)
        
        try:
            start_time = time.time()
            embeddings = []
            
            # Processar em lotes
            for i in range(0, len(processed_images), batch_size):
                batch_images = processed_images[i:i+batch_size]
                
                # Processar imagens
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Gerar embeddings
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    batch_embeddings = outputs.cpu().numpy()
                    
                # Normalizar se solicitado
                if normalize:
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / np.maximum(norms, 1e-12)
                    
                embeddings.append(batch_embeddings)
                
            # Concatenar resultados dos lotes
            all_embeddings = np.vstack(embeddings)
            
            logger.info(f"Gerados embeddings para {len(processed_images)} imagens em {time.time() - start_time:.2f} segundos")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings de imagem com CLIP: {str(e)}")
            raise
            
    def compute_similarity(
        self, 
        text_embeddings: np.ndarray, 
        image_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calcula a similaridade entre embeddings de texto e imagem.
        
        Args:
            text_embeddings: Embeddings de texto
            image_embeddings: Embeddings de imagem
            
        Returns:
            Matriz de similaridade (shape: [len(text_embeddings), len(image_embeddings)])
        """
        # Normalizar embeddings (se ainda não estiverem normalizados)
        text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        image_norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        
        normalized_text_embeddings = text_embeddings / np.maximum(text_norms, 1e-12)
        normalized_image_embeddings = image_embeddings / np.maximum(image_norms, 1e-12)
        
        # Calcular similaridade (produto escalar)
        return np.dot(normalized_text_embeddings, normalized_image_embeddings.T)
        
    def search_images_with_text(
        self, 
        query_texts: Union[str, List[str]], 
        image_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Busca imagens usando consultas de texto.
        
        Args:
            query_texts: Texto(s) de consulta
            image_embeddings: Embeddings de imagens pré-computados
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de dicionários com índices e scores de similaridade
        """
        # Gerar embeddings para as consultas de texto
        text_embeddings = self.generate_text_embeddings(query_texts)
        
        # Calcular similaridade
        similarity_scores = self.compute_similarity(text_embeddings, image_embeddings)
        
        # Processar resultados
        results = []
        
        for i, scores in enumerate(similarity_scores):
            # Obter os top_k índices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Criar resultado para esta consulta
            query_result = {
                "query_index": i,
                "query": query_texts[i] if isinstance(query_texts, list) else query_texts,
                "matches": [
                    {
                        "index": int(idx),
                        "score": float(scores[idx])
                    }
                    for idx in top_indices
                ]
            }
            
            results.append(query_result)
            
        return results
        
    def _process_images(self, images: Union[Image.Image, List[Image.Image], str, List[str]]) -> List[Image.Image]:
        """
        Processa a entrada, convertendo para lista de objetos PIL.Image.
        
        Args:
            images: Imagem(ns) como objetos PIL.Image ou strings base64
            
        Returns:
            Lista de objetos PIL.Image
        """
        # Converter para lista se for um único item
        if isinstance(images, (Image.Image, str)):
            images = [images]
            
        processed_images = []
        
        for img in images:
            if isinstance(img, Image.Image):
                # Já é um objeto PIL.Image
                processed_images.append(img)
            elif isinstance(img, str):
                # String base64 ou caminho de arquivo
                if img.startswith("data:image"):
                    # Base64 com prefixo data URI
                    format, imgstr = img.split(';base64,')
                    img_data = base64.b64decode(imgstr)
                    img_pil = Image.open(io.BytesIO(img_data))
                    processed_images.append(img_pil)
                elif img.startswith(("http://", "https://")):
                    # URL de imagem - precisaria de um pacote como requests
                    raise ValueError("URLs de imagem não são suportadas diretamente. Baixe a imagem primeiro.")
                else:
                    # Assumir que é um caminho de arquivo ou base64 sem prefixo
                    try:
                        # Tentar carregar como caminho de arquivo
                        img_pil = Image.open(img)
                        processed_images.append(img_pil)
                    except:
                        try:
                            # Tentar decodificar como base64
                            img_data = base64.b64decode(img)
                            img_pil = Image.open(io.BytesIO(img_data))
                            processed_images.append(img_pil)
                        except:
                            logger.warning(f"Não foi possível processar a imagem: {img[:50]}...")
                            continue
            else:
                logger.warning(f"Tipo de entrada não suportado: {type(img)}")
                continue
                
        return processed_images
