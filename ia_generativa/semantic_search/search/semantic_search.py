"""
Semantic Search - Funções para busca semântica.
"""

import os
import logging
import time
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple, Callable
import json
import uuid
from datetime import datetime
import threading

from semantic_search.embeddings.text_embeddings import TextEmbeddingGenerator
from semantic_search.embeddings.image_embeddings import ImageEmbeddingGenerator
from semantic_search.embeddings.multimodal_embeddings import MultimodalEmbeddingGenerator
from semantic_search.indexing.index_manager import IndexManager

from storage.repositories import ResultRepository

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Motor de busca semântica que coordena embeddings e índices."""
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        use_gpu: bool = None
    ):
        """
        Inicializa o motor de busca semântica.
        
        Args:
            storage_dir: Diretório para armazenar índices e modelos
            use_gpu: Se True, usa GPU quando disponível
        """
        # Determinar uso de GPU automaticamente se não especificado
        if use_gpu is None:
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
                
        self.use_gpu = use_gpu
        self.storage_dir = storage_dir or os.path.join("data", "semantic_search")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Diretórios específicos
        self.index_dir = os.path.join(self.storage_dir, "indices")
        self.model_cache_dir = os.path.join(self.storage_dir, "model_cache")
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Instanciar gerenciador de índices
        self.index_manager = IndexManager(
            storage_dir=self.index_dir,
            auto_save_interval=1800,  # 30 minutos
            auto_save_min_updates=50
        )
        
        # Geradores de embeddings (inicializados sob demanda)
        self.text_embedding_generator = None
        self.image_embedding_generator = None
        self.multimodal_embedding_generator = None
        
        # Mapeamento de tipos de conteúdo para índices
        self.content_type_indices = {}
        
        # Lock para operações thread-safe
        self.lock = threading.RLock()
        
        logger.info(f"Motor de busca semântica inicializado (GPU: {use_gpu})")
        
    def _get_text_embedding_generator(self) -> TextEmbeddingGenerator:
        """
        Obtém o gerador de embeddings de texto, inicializando se necessário.
        
        Returns:
            Gerador de embeddings de texto
        """
        if self.text_embedding_generator is None:
            logger.info("Inicializando gerador de embeddings de texto")
            self.text_embedding_generator = TextEmbeddingGenerator(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cuda" if self.use_gpu else "cpu",
                cache_dir=self.model_cache_dir
            )
        return self.text_embedding_generator
        
    def _get_image_embedding_generator(self) -> ImageEmbeddingGenerator:
        """
        Obtém o gerador de embeddings de imagem, inicializando se necessário.
        
        Returns:
            Gerador de embeddings de imagem
        """
        if self.image_embedding_generator is None:
            logger.info("Inicializando gerador de embeddings de imagem")
            self.image_embedding_generator = ImageEmbeddingGenerator(
                model_name="openai/clip-vit-base-patch32",
                device="cuda" if self.use_gpu else "cpu",
                cache_dir=self.model_cache_dir
            )
        return self.image_embedding_generator
        
    def _get_multimodal_embedding_generator(self) -> MultimodalEmbeddingGenerator:
        """
        Obtém o gerador de embeddings multimodal, inicializando se necessário.
        
        Returns:
            Gerador de embeddings multimodal
        """
        if self.multimodal_embedding_generator is None:
            logger.info("Inicializando gerador de embeddings multimodal")
            self.multimodal_embedding_generator = MultimodalEmbeddingGenerator(
                model_name="openai/clip-vit-base-patch32",
                device="cuda" if self.use_gpu else "cpu",
                cache_dir=self.model_cache_dir
            )
        return self.multimodal_embedding_generator
        
    def create_index_for_content_type(
        self,
        content_type: str,
        dimension: Optional[int] = None,
        index_type: str = "flat",
        metric: str = "cosine"
    ) -> str:
        """
        Cria um índice para um tipo específico de conteúdo.
        
        Args:
            content_type: Tipo de conteúdo ('text', 'image', etc.)
            dimension: Dimensão dos embeddings (opcional, determinada automaticamente)
            index_type: Tipo de índice ('flat', 'ivf', 'hnsw')
            metric: Métrica de distância ('cosine', 'l2')
            
        Returns:
            ID do índice criado
        """
        with self.lock:
            # Determinar dimensão automaticamente se não especificada
            if dimension is None:
                if content_type == "text":
                    generator = self._get_text_embedding_generator()
                    dimension = generator.embedding_dim
                elif content_type == "image":
                    generator = self._get_image_embedding_generator()
                    dimension = generator.embedding_dim
                elif content_type == "multimodal":
                    generator = self._get_multimodal_embedding_generator()
                    dimension = generator.text_embedding_dim  # Usar dimensão de texto como padrão
                else:
                    # Usar dimensão padrão para outros tipos
                    dimension = 768
                    
            # Criar índice
            index_id = self.index_manager.create_index(
                name=f"{content_type}_index",
                dimension=dimension,
                index_type=index_type,
                metric=metric,
                use_gpu=self.use_gpu,
                metadata={"content_type": content_type}
            )
            
            # Associar tipo de conteúdo ao índice
            self.content_type_indices[content_type] = index_id
            
            logger.info(f"Índice criado para tipo de conteúdo '{content_type}': {index_id} (dimensão: {dimension})")
            
            return index_id
            
    def get_index_for_content_type(self, content_type: str) -> Optional[str]:
        """
        Obtém o ID do índice associado a um tipo de conteúdo.
        
        Args:
            content_type: Tipo de conteúdo
            
        Returns:
            ID do índice ou None se não encontrado
        """
        return self.content_type_indices.get(content_type)
        
    def index_text(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[List[str]] = None,
        index_id: Optional[str] = None
    ) -> List[str]:
        """
        Indexa textos para busca semântica.
        
        Args:
            texts: Texto ou lista de textos
            metadata: Metadados (único ou lista)
            ids: IDs opcionais para os textos
            index_id: ID do índice (opcional, usa o índice padrão de texto se None)
            
        Returns:
            Lista de IDs dos itens indexados
        """
        # Normalizar entrada para lista
        if isinstance(texts, str):
            texts = [texts]
            
        # Normalizar metadados
        if metadata is None:
            metadata = [{} for _ in texts]
        elif isinstance(metadata, dict):
            metadata = [metadata.copy() for _ in texts]
            
        if len(metadata) != len(texts):
            raise ValueError(f"Número de metadados ({len(metadata)}) diferente do número de textos ({len(texts)})")
            
        # Determinar índice a usar
        if index_id is None:
            index_id = self.get_index_for_content_type("text")
            if index_id is None:
                # Criar índice se não existir
                index_id = self.create_index_for_content_type("text")
                
        # Gerar embeddings
        generator = self._get_text_embedding_generator()
        embeddings = generator.generate_embeddings(texts)
        
        # Indexar embeddings
        return self.index_manager.add_items(index_id, embeddings, ids, metadata)
        
    def index_image(
        self,
        images: Union[Any, List[Any]],  # PIL.Image, np.ndarray, base64, etc.
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[List[str]] = None,
        index_id: Optional[str] = None
    ) -> List[str]:
        """
        Indexa imagens para busca semântica.
        
        Args:
            images: Imagem ou lista de imagens
            metadata: Metadados (único ou lista)
            ids: IDs opcionais para as imagens
            index_id: ID do índice (opcional, usa o índice padrão de imagem se None)
            
        Returns:
            Lista de IDs dos itens indexados
        """
        # Normalizar entrada para lista
        if not isinstance(images, list):
            images = [images]
            
        # Normalizar metadados
        if metadata is None:
            metadata = [{} for _ in images]
        elif isinstance(metadata, dict):
            metadata = [metadata.copy() for _ in images]
            
        if len(metadata) != len(images):
            raise ValueError(f"Número de metadados ({len(metadata)}) diferente do número de imagens ({len(images)})")
            
        # Determinar índice a usar
        if index_id is None:
            index_id = self.get_index_for_content_type("image")
            if index_id is None:
                # Criar índice se não existir
                index_id = self.create_index_for_content_type("image")
                
        # Gerar embeddings
        generator = self._get_image_embedding_generator()
        embeddings = generator.generate_embeddings(images)
        
        # Indexar embeddings
        return self.index_manager.add_items(index_id, embeddings, ids, metadata)
        
    def index_content_from_repository(
        self,
        user_id: Optional[int] = None,
        result_types: Optional[List[str]] = None,
        limit: int = 1000,
        reindex: bool = False
    ) -> Dict[str, int]:
        """
        Indexa conteúdo do repositório de resultados.
        
        Args:
            user_id: Filtrar por usuário (opcional)
            result_types: Tipos de resultado a indexar (opcional)
            limit: Limite de itens por tipo
            reindex: Se True, reindexar itens já indexados
            
        Returns:
            Dicionário com contagem de itens indexados por tipo
        """
        # Tipos padrão se não especificados
        if result_types is None:
            result_types = ["text", "image"]
            
        counts = {}
        
        # Processar cada tipo
        for result_type in result_types:
            count = 0
            
            # Obter ou criar índice
            index_id = self.get_index_for_content_type(result_type)
            if index_id is None:
                index_id = self.create_index_for_content_type(result_type)
                
            # Buscar resultados
            results, total = ResultRepository.search_results(
                user_id=user_id,
                result_type=result_type,
                limit=limit
            )
            
            # Processar cada resultado
            for result in results:
                result_id = result["result_id"]
                
                # Verificar se já está indexado (se não estiver reindexando)
                if not reindex:
                    # Verificar nos metadados
                    index = self.index_manager.get_index(index_id)
                    exists = False
                    
                    for id_ in index.metadata:
                        if index.metadata[id_].get("result_id") == result_id:
                            exists = True
                            break
                            
                    if exists:
                        continue
                        
                # Obter conteúdo completo
                full_result = ResultRepository.get_result(result_id)
                
                if not full_result or "content" not in full_result:
                    continue
                    
                content = full_result["content"]
                
                # Preparar metadados
                item_metadata = {
                    "result_id": result_id,
                    "user_id": full_result.get("user_id"),
                    "created_at": full_result.get("created_at"),
                    "original_metadata": full_result.get("metadata", {})
                }
                
                # Indexar de acordo com o tipo
                try:
                    if result_type == "text":
                        self.index_text(content, item_metadata, index_id=index_id)
                        count += 1
                    elif result_type == "image":
                        self.index_image(content, item_metadata, index_id=index_id)
                        count += 1
                except Exception as e:
                    logger.error(f"Erro ao indexar resultado {result_id}: {str(e)}")
                    
            counts[result_type] = count
            
        return counts
        
    def search_text(
        self,
        query: str,
        index_id: Optional[str] = None,
        k: int = 10,
        include_content: bool = False,
        filter_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Busca textos semanticamente semelhantes.
        
        Args:
            query: Texto de consulta
            index_id: ID do índice (opcional)
            k: Número de resultados
            include_content: Se True, inclui conteúdo nos resultados
            filter_func: Função para filtrar resultados
            
        Returns:
            Resultados da busca
        """
        # Determinar índice a usar
        if index_id is None:
            index_id = self.get_index_for_content_type("text")
            if index_id is None:
                raise ValueError("Nenhum índice de texto disponível")
                
        # Gerar embedding da consulta
        generator = self._get_text_embedding_generator()
        query_embedding = generator.generate_embeddings(query)
        
        # Buscar no índice
        search_results = self.index_manager.search(
            index_id=index_id,
            query_embedding=query_embedding,
            k=k,
            include_distances=True,
            filter_func=filter_func
        )
        
        # Formatar resultados
        formatted_results = []
        
        for i, result_id in enumerate(search_results["ids"]):
            item = {
                "id": result_id,
                "score": float(search_results["distances"][i]),
                "metadata": search_results["metadata"][i]
            }
            
            # Obter conteúdo completo se solicitado
            if include_content and "result_id" in item["metadata"]:
                try:
                    full_result = ResultRepository.get_result(item["metadata"]["result_id"])
                    if full_result and "content" in full_result:
                        item["content"] = full_result["content"]
                except Exception as e:
                    logger.warning(f"Erro ao obter conteúdo para {result_id}: {str(e)}")
                    
            formatted_results.append(item)
            
        return {
            "query": query,
            "results": formatted_results,
            "total": len(formatted_results)
        }
        
    def search_image(
        self,
        query_image: Any,  # PIL.Image, np.ndarray, base64, etc.
        index_id: Optional[str] = None,
        k: int = 10,
        include_content: bool = False,
        filter_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Busca imagens visualmente semelhantes.
        
        Args:
            query_image: Imagem de consulta
            index_id: ID do índice (opcional)
            k: Número de resultados
            include_content: Se True, inclui conteúdo nos resultados
            filter_func: Função para filtrar resultados
            
        Returns:
            Resultados da busca
        """
        # Determinar índice a usar
        if index_id is None:
            index_id = self.get_index_for_content_type("image")
            if index_id is None:
                raise ValueError("Nenhum índice de imagem disponível")
                
        # Gerar embedding da consulta
        generator = self._get_image_embedding_generator()
        query_embedding = generator.generate_embeddings(query_image)
        
        # Buscar no índice
        search_results = self.index_manager.search(
            index_id=index_id,
            query_embedding=query_embedding,
            k=k,
            include_distances=True,
            filter_func=filter_func
        )
        
        # Formatar resultados
        formatted_results = []
        
        for i, result_id in enumerate(search_results["ids"]):
            item = {
                "id": result_id,
                "score": float(search_results["distances"][i]),
                "metadata": search_results["metadata"][i]
            }
            
            # Obter conteúdo completo se solicitado
            if include_content and "result_id" in item["metadata"]:
                try:
                    full_result = ResultRepository.get_result(item["metadata"]["result_id"])
                    if full_result and "content" in full_result:
                        item["content"] = full_result["content"]
                except Exception as e:
                    logger.warning(f"Erro ao obter conteúdo para {result_id}: {str(e)}")
                    
            formatted_results.append(item)
            
        return {
            "results": formatted_results,
            "total": len(formatted_results)
        }
        
    def search_images_with_text(
        self,
        query: str,
        index_id: Optional[str] = None,
        k: int = 10,
        include_content: bool = False,
        filter_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Busca imagens usando uma consulta de texto (CLIP).
        
        Args:
            query: Texto de consulta
            index_id: ID do índice (opcional)
            k: Número de resultados
            include_content: Se True, inclui conteúdo nos resultados
            filter_func: Função para filtrar resultados
            
        Returns:
            Resultados da busca
        """
        # Determinar índice a usar
        if index_id is None:
            index_id = self.get_index_for_content_type("image")
            if index_id is None:
                raise ValueError("Nenhum índice de imagem disponível")
                
        # Gerar embedding da consulta de texto usando modelo multimodal
        generator = self._get_multimodal_embedding_generator()
        query_embedding = generator.generate_text_embeddings(query)
        
        # Buscar no índice
        search_results = self.index_manager.search(
            index_id=index_id,
            query_embedding=query_embedding,
            k=k,
            include_distances=True,
            filter_func=filter_func
        )
        
        # Formatar resultados
        formatted_results = []
        
        for i, result_id in enumerate(search_results["ids"]):
            item = {
                "id": result_id,
                "score": float(search_results["distances"][i]),
                "metadata": search_results["metadata"][i]
            }
            
            # Obter conteúdo completo se solicitado
            if include_content and "result_id" in item["metadata"]:
                try:
                    full_result = ResultRepository.get_result(item["metadata"]["result_id"])
                    if full_result and "content" in full_result:
                        item["content"] = full_result["content"]
                except Exception as e:
                    logger.warning(f"Erro ao obter conteúdo para {result_id}: {str(e)}")
                    
            formatted_results.append(item)
            
        return {
            "query": query,
            "results": formatted_results,
            "total": len(formatted_results)
        }
