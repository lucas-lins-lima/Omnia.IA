"""
Index Manager - Gerenciador central de índices vetoriais.
"""

import os
import logging
import time
import json
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
import numpy as np
import threading
from datetime import datetime
import uuid

from semantic_search.indexing.vector_index import VectorIndex

logger = logging.getLogger(__name__)

class IndexManager:
    """Gerenciador central de múltiplos índices vetoriais."""
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        auto_save_interval: Optional[int] = 3600,  # Segundos
        auto_save_min_updates: int = 100
    ):
        """
        Inicializa o gerenciador de índices.
        
        Args:
            storage_dir: Diretório para armazenar índices
            auto_save_interval: Intervalo para salvamento automático em segundos (None para desativar)
            auto_save_min_updates: Número mínimo de atualizações para salvar automaticamente
        """
        self.storage_dir = storage_dir or os.path.join("data", "indices")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.auto_save_interval = auto_save_interval
        self.auto_save_min_updates = auto_save_min_updates
        
        # Dicionário de índices
        self.indices: Dict[str, VectorIndex] = {}
        
        # Contadores de atualizações
        self.update_counts: Dict[str, int] = {}
        
        # Timestamps do último salvamento
        self.last_saved: Dict[str, float] = {}
        
        # Lock para operações thread-safe
        self.lock = threading.RLock()
        
        # Iniciar thread de salvamento automático se habilitado
        if auto_save_interval is not None:
            self._start_auto_save_thread()
            
        logger.info(f"Gerenciador de índices inicializado em {self.storage_dir}")
        
    def _start_auto_save_thread(self):
        """Inicia a thread de salvamento automático."""
        def auto_save_worker():
            while True:
                time.sleep(self.auto_save_interval)
                try:
                    self._auto_save()
                except Exception as e:
                    logger.error(f"Erro no salvamento automático: {str(e)}")
                    
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
        logger.info(f"Thread de salvamento automático iniciada (intervalo: {self.auto_save_interval}s)")
        
    def _auto_save(self):
        """Salva índices automaticamente se necessário."""
        with self.lock:
            current_time = time.time()
            saved_count = 0
            
            for index_id, index in self.indices.items():
                # Verificar se há atualizações suficientes
                update_count = self.update_counts.get(index_id, 0)
                if update_count < self.auto_save_min_updates:
                    continue
                    
                # Verificar intervalo de tempo desde o último salvamento
                last_saved = self.last_saved.get(index_id, 0)
                if current_time - last_saved < self.auto_save_interval:
                    continue
                    
                # Salvar índice
                try:
                    file_path = os.path.join(self.storage_dir, f"index_{index_id}")
                    index.save(file_path)
                    
                    # Atualizar contadores
                    self.last_saved[index_id] = current_time
                    self.update_counts[index_id] = 0
                    
                    saved_count += 1
                    
                except Exception as e:
                    logger.error(f"Erro ao salvar índice {index_id}: {str(e)}")
                    
            if saved_count > 0:
                logger.info(f"Salvamento automático: {saved_count} índices salvos")
                
    def create_index(
        self,
        name: str,
        dimension: int,
        index_type: str = "flat",
        metric: str = "cosine",
        use_gpu: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Cria um novo índice.
        
        Args:
            name: Nome do índice
            dimension: Dimensão dos vetores
            index_type: Tipo de índice ('flat', 'ivf', 'hnsw', etc.)
            metric: Métrica de distância ('cosine', 'l2', 'ip')
            use_gpu: Se True, usa GPU para busca
            metadata: Metadados adicionais
            
        Returns:
            ID do índice criado
        """
        with self.lock:
            # Gerar ID único
            index_id = str(uuid.uuid4())
            
            # Criar índice
            index = VectorIndex(
                dimension=dimension,
                index_type=index_type,
                metric=metric,
                use_gpu=use_gpu,
                temp_dir=self.storage_dir
            )
            
            # Adicionar metadados
            index_metadata = metadata or {}
            index_metadata["name"] = name
            index_metadata["created_at"] = datetime.now().isoformat()
            
            # Adicionar ao dicionário de índices
            self.indices[index_id] = index
            self.update_counts[index_id] = 0
            self.last_saved[index_id] = time.time()
            
            logger.info(f"Índice criado: {index_id} (nome: {name}, dimensão: {dimension})")
            
            return index_id
            
    def get_index(self, index_id: str) -> Optional[VectorIndex]:
        """
        Obtém um índice pelo ID.
        
        Args:
            index_id: ID do índice
            
        Returns:
            Índice ou None se não encontrado
        """
        with self.lock:
            return self.indices.get(index_id)
            
    def list_indices(self) -> List[Dict[str, Any]]:
        """
        Lista todos os índices.
        
        Returns:
            Lista de dicionários com informações de cada índice
        """
        with self.lock:
            return [
                {**index.get_index_info(), "update_count": self.update_counts.get(idx, 0)}
                for idx, index in self.indices.items()
            ]
            
    def add_items(
        self,
        index_id: str,
        embeddings: np.ndarray,
        ids: Optional[List[Union[str, int]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Adiciona itens a um índice.
        
        Args:
            index_id: ID do índice
            embeddings: Matriz de embeddings
            ids: Lista de IDs (opcional)
            metadata: Lista de metadados (opcional)
            
        Returns:
            Lista de IDs dos itens adicionados
        """
        with self.lock:
            index = self.get_index(index_id)
            if index is None:
                raise ValueError(f"Índice não encontrado: {index_id}")
                
            # Adicionar ao índice
            result_ids = index.add(embeddings, ids, metadata)
            
            # Atualizar contador
            self.update_counts[index_id] = self.update_counts.get(index_id, 0) + len(result_ids)
            
            return result_ids
            
    def search(
        self,
        index_id: str,
        query_embedding: np.ndarray,
        k: int = 10,
        include_distances: bool = True,
        filter_func: Optional[Callable[[str, Dict[str, Any]], bool]] = None
    ) -> Dict[str, Any]:
        """
        Busca itens semelhantes a um embedding de consulta.
        
        Args:
            index_id: ID do índice
            query_embedding: Embedding de consulta
            k: Número de resultados
            include_distances: Se True, inclui distâncias
            filter_func: Função para filtrar resultados
            
        Returns:
            Resultados da busca
        """
        with self.lock:
            index = self.get_index(index_id)
            if index is None:
                raise ValueError(f"Índice não encontrado: {index_id}")
                
            return index.search(query_embedding, k, include_distances, filter_func)
            
    def batch_search(
        self,
        index_id: str,
        query_embeddings: np.ndarray,
        k: int = 10,
        include_distances: bool = True,
        filter_func: Optional[Callable[[str, Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca itens semelhantes para múltiplos embeddings de consulta.
        
        Args:
            index_id: ID do índice
            query_embeddings: Embeddings de consulta
            k: Número de resultados por consulta
            include_distances: Se True, inclui distâncias
            filter_func: Função para filtrar resultados
            
        Returns:
            Lista de resultados para cada consulta
        """
        with self.lock:
            index = self.get_index(index_id)
            if index is None:
                raise ValueError(f"Índice não encontrado: {index_id}")
                
            return index.batch_search(query_embeddings, k, include_distances, filter_func)
            
    def remove_items(self, index_id: str, ids: List[Union[str, int]]) -> int:
        """
        Remove itens de um índice.
        
        Args:
            index_id: ID do índice
            ids: IDs dos itens a remover
            
        Returns:
            Número de itens removidos
        """
        with self.lock:
            index = self.get_index(index_id)
            if index is None:
                raise ValueError(f"Índice não encontrado: {index_id}")
                
            removed = index.remove(ids)
            
            # Atualizar contador se foram removidos itens
            if removed > 0:
                self.update_counts[index_id] = self.update_counts.get(index_id, 0) + removed
                
            return removed
            
    def save_index(self, index_id: str, file_path: Optional[str] = None) -> str:
        """
        Salva um índice em disco.
        
        Args:
            index_id: ID do índice
            file_path: Caminho para salvar (opcional)
            
        Returns:
            Caminho onde o índice foi salvo
        """
        with self.lock:
            index = self.get_index(index_id)
            if index is None:
                raise ValueError(f"Índice não encontrado: {index_id}")
                
            # Usar caminho padrão se não especificado
            if file_path is None:
                file_path = os.path.join(self.storage_dir, f"index_{index_id}")
                
            # Salvar índice
            saved_path = index.save(file_path)
            
            # Atualizar timestamps e contadores
            self.last_saved[index_id] = time.time()
            self.update_counts[index_id] = 0
            
            return saved_path
            
    def load_index(
        self,
        file_path: str,
        index_id: Optional[str] = None,
        use_gpu: bool = False
    ) -> str:
        """
        Carrega um índice do disco.
        
        Args:
            file_path: Caminho base do índice
            index_id: ID para o índice carregado (opcional, usa o original se None)
            use_gpu: Se True, carrega o índice na GPU
            
        Returns:
            ID do índice carregado
        """
        with self.lock:
            # Carregar índice
            loaded_index = VectorIndex.load(file_path, use_gpu)
            
            # Usar ID original ou o fornecido
            if index_id is None:
                index_id = loaded_index.index_id
                
            # Adicionar ao dicionário de índices
            self.indices[index_id] = loaded_index
            self.update_counts[index_id] = 0
            self.last_saved[index_id] = time.time()
            
            logger.info(f"Índice carregado: {index_id} (de {file_path})")
            
            return index_id
            
    def delete_index(self, index_id: str) -> bool:
        """
        Remove um índice da memória.
        
        Args:
            index_id: ID do índice
            
        Returns:
            True se removido com sucesso, False se não encontrado
        """
        with self.lock:
            if index_id not in self.indices:
                return False
                
            # Remover do dicionário
            del self.indices[index_id]
            
            # Remover contadores e timestamps
            if index_id in self.update_counts:
                del self.update_counts[index_id]
            if index_id in self.last_saved:
                del self.last_saved[index_id]
                
            logger.info(f"Índice removido: {index_id}")
            
            return True
            
    def get_index_stats(self, index_id: str) -> Dict[str, Any]:
        """
        Obtém estatísticas detalhadas de um índice.
        
        Args:
            index_id: ID do índice
            
        Returns:
            Dicionário com estatísticas
        """
        with self.lock:
            index = self.get_index(index_id)
            if index is None:
                raise ValueError(f"Índice não encontrado: {index_id}")
                
            # Obter informações básicas
            info = index.get_index_info()
            
            # Adicionar estatísticas do gerenciador
            info["update_count"] = self.update_counts.get(index_id, 0)
            
            # Calcular tempo desde o último salvamento
            last_saved = self.last_saved.get(index_id, 0)
            info["last_saved"] = datetime.fromtimestamp(last_saved).isoformat() if last_saved > 0 else None
            info["seconds_since_last_save"] = time.time() - last_saved if last_saved > 0 else None
            
            return info
