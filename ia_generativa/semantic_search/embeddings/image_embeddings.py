"""
Image Embeddings - Geração de embeddings para conteúdo visual.
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

from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)

class ImageEmbeddingGenerator:
    """Gerador de embeddings para imagens."""
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_type: str = "clip"
    ):
        """
        Inicializa o gerador de embeddings para imagens.
        
        Args:
            model_name: Nome do modelo de embeddings
            device: Dispositivo para execução (cuda, cpu)
            cache_dir: Diretório para cache de modelos
            model_type: Tipo de modelo ('clip' ou 'vision')
        """
        self.model_name = model_name
        self.model_type = model_type
        
        # Determinar o dispositivo automaticamente se não for especificado
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Configurar cache para os modelos
        self.cache_dir = cache_dir or os.path.join("data", "models_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.embedding_dim = None
        
        logger.info(f"Inicializando ImageEmbeddingGenerator com modelo {model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Carregar modelo
        self._load_model()
        
    def _load_model(self):
        """Carrega o modelo de embeddings para imagens."""
        try:
            start_time = time.time()
            
            if self.model_type == "clip":
                # Carregar modelo CLIP
                self.processor = CLIPProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = CLIPModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                ).to(self.device)
                
                # Obter dimensão dos embeddings
                self.embedding_dim = self.model.config.vision_config.hidden_size
                
            elif self.model_type == "vision":
                # Carregar modelo de visão específico
                self.processor = AutoImageProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                ).to(self.device)
                
                # Estimar dimensão dos embeddings
                dummy_image = Image.new('RGB', (224, 224), color='white')
                inputs = self.processor(images=dummy_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Dependendo do modelo, o embedding pode estar em diferentes locais
                if hasattr(outputs, "pooler_output"):
                    self.embedding_dim = outputs.pooler_output.shape[-1]
                else:
                    self.embedding_dim = outputs.last_hidden_state.shape[-1]
                    
            else:
                raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
                
            logger.info(f"Modelo carregado em {time.time() - start_time:.2f} segundos")
            logger.info(f"Dimensão dos embeddings: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de embeddings para imagens: {str(e)}")
            raise
            
    def generate_embeddings(
        self, 
        images: Union[Image.Image, List[Image.Image], str, List[str]],
        batch_size: int = 16,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Gera embeddings para imagens.
        
        Args:
            images: Imagem(ns) como objetos PIL.Image ou strings base64
            batch_size: Tamanho do lote para processamento
            normalize: Se True, normaliza os embeddings para norma unitária
            
        Returns:
            Array NumPy com embeddings (shape: [n_images, embedding_dim])
        """
        # Processar entrada
        processed_images = self._process_input(images)
        
        try:
            start_time = time.time()
            embeddings = []
            
            # Processar em lotes
            for i in range(0, len(processed_images), batch_size):
                batch_images = processed_images[i:i+batch_size]
                
                if self.model_type == "clip":
                    # Processar com CLIP
                    inputs = self.processor(
                        images=batch_images,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    # Extrair features visuais
                    with torch.no_grad():
                        outputs = self.model.get_image_features(**inputs)
                        batch_embeddings = outputs.cpu().numpy()
                        
                elif self.model_type == "vision":
                    # Processar com modelo de visão
                    inputs = self.processor(
                        images=batch_images,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    # Extrair embeddings
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        
                        # Dependendo do modelo, o embedding pode estar em diferentes locais
                        if hasattr(outputs, "pooler_output"):
                            batch_embeddings = outputs.pooler_output.cpu().numpy()
                        else:
                            # Fazer média do último estado oculto (cls token ou média global)
                            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                            
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
            logger.error(f"Erro ao gerar embeddings de imagem: {str(e)}")
            raise
            
    def _process_input(self, images: Union[Image.Image, List[Image.Image], str, List[str]]) -> List[Image.Image]:
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
