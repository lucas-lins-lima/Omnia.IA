"""
Vector Index - Gerenciamento de índices vetoriais para busca semântica.
"""

import os
import logging
import time
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple, Callable
import json
import pickle
import uuid
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS não está disponível. Usando fallback para busca vetorial.")
    FAISS_AVAILABLE = False

class VectorIndex:
    """Índice vetorial para busca semântica rápida."""
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "cosine",
        use_gpu: bool = False,
        temp_dir: Optional[str] = None
    ):
        """
        Inicializa o índice vetorial.
        
        Args:
            dimension: Dimensão dos vetores
            index_type: Tipo de índice ('flat', 'ivf', 'hnsw', etc.)
            metric: Métrica de distância ('cosine', 'l2', 'ip')
            use_gpu: Se True, usa GPU para busca (requer faiss-gpu)
            temp_dir: Diretório para arquivos temporários
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Metadados internos
        self.index = None
        self.metadata = {}
        self.id_to_index = {}  # Mapeamento de ID externo para índice interno
        self.index_to_id = {}  # Mapeamento de índice interno para ID externo
        self.is_trained = False
        self.index_size = 0
        self.index_id = str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
        self.last_updated = None
        
        logger.info(f"Inicializando índice vetorial {self.index_id} com dimensão {dimension}")
        
        # Criar índice
        self._create_index()
        
    def _create_index(self):
        """Cria o índice FAISS com base no tipo e métrica especificados."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS não disponível, usando fallback para NumPy")
            self.embeddings = np.zeros((0, self.dimension), dtype=np.float32)
            return
            
        try:
            # Determinar métrica
            if self.metric == "cosine":
                # Para similaridade de cosseno, usamos produto interno em vetores normalizados
                metric_type = faiss.METRIC_INNER_PRODUCT
            elif self.metric == "l2":
                metric_type = faiss.METRIC_L2
            elif self.metric == "ip":
                metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                logger.warning(f"Métrica desconhecida: {self.metric}, usando L2")
                metric_type = faiss.METRIC_L2
                
            # Criar índice
            if self.index_type == "flat":
                # Índice exato (sem aproximação)
                self.index = faiss.IndexFlatL2(self.dimension)
                if self.metric == "cosine" or self.metric == "ip":
                    self.index = faiss.IndexFlatIP(self.dimension)
                    
            elif self.index_type == "ivf":
                # Índice IVF (Inverted File Index)
                # Requer treinamento com dados
                quantizer = faiss.IndexFlatL2(self.dimension)
                if self.metric == "cosine" or self.metric == "ip":
                    quantizer = faiss.IndexFlatIP(self.dimension)
                    
                # Número de centroides (clusters)
                nlist = 100  # Ajuste este valor de acordo com o tamanho esperado do índice
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric_type)
                
            elif self.index_type == "hnsw":
                # Índice HNSW (Hierarchical Navigable Small World)
                # Rápido e eficiente para pesquisas aproximadas
                self.index = faiss.IndexHNSWFlat(self.dimension, 32, metric_type)
                
            else:
                logger.warning(f"Tipo de índice desconhecido: {self.index_type}, usando Flat")
                self.index = faiss.IndexFlatL2(self.dimension)
                if self.metric == "cosine" or self.metric == "ip":
                    self.index = faiss.IndexFlatIP(self.dimension)
                    
            # Configurar GPU se solicitado
            if self.use_gpu:
                try:
                    gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
                    logger.info("Índice movido para GPU")
                except Exception as e:
                    logger.warning(f"Não foi possível usar GPU: {str(e)}")
                    
            logger.info(f"Índice {self.index_type} criado com métrica {self.metric}")
            
        except Exception as e:
            logger.error(f"Erro ao criar índice FAISS: {str(e)}")
            logger.warning("Usando fallback para NumPy")
            self.embeddings = np.zeros((0, self.dimension), dtype=np.float32)
            
    def add(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[Union[str, int]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Adiciona embeddings ao índice.
        
        Args:
            embeddings: Matriz de embeddings (shape: [n_vectors, dimension])
            ids: Lista de IDs externos (opcional, gera automático se None)
            metadata: Lista de metadados para cada embedding (opcional)
            
        Returns:
            Lista de IDs dos embeddings adicionados
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Dimensão incompatível: {embeddings.shape[1]}, esperado {self.dimension}")
            
        # Converter para float32 (requerido pelo FAISS)
        embeddings = embeddings.astype(np.float32)
        
        # Normalizar vetores para similaridade de cosseno
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)
            
        # Gerar IDs se não fornecidos
        n_vectors = embeddings.shape[0]
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n_vectors)]
        elif len(ids) != n_vectors:
            raise ValueError(f"Número de IDs ({len(ids)}) diferente do número de embeddings ({n_vectors})")
            
        # Padronizar metadados
        if metadata is None:
            metadata = [{} for _ in range(n_vectors)]
        elif len(metadata) != n_vectors:
            raise ValueError(f"Número de metadados ({len(metadata)}) diferente do número de embeddings ({n_vectors})")
            
        # Adicionar ao índice
        if FAISS_AVAILABLE and self.index is not None:
            try:
                # Índices iniciais para os novos vetores
                start_index = self.index_size
                
                # Adicionar ao índice FAISS
                if hasattr(self.index, 'train') and not self.is_trained and self.index_type != "flat":
                    # Treinar o índice com os novos dados
                    self.index.train(embeddings)
                    self.is_trained = True
                    logger.info(f"Índice treinado com {n_vectors} vetores")
                    
                self.index.add(embeddings)
                
                # Atualizar mapeamentos e metadados
                for i, (id_, meta) in enumerate(zip(ids, metadata)):
                    idx = start_index + i
                    self.id_to_index[id_] = idx
                    self.index_to_id[idx] = id_
                    self.metadata[id_] = meta
                    
                # Atualizar tamanho do índice
                self.index_size += n_vectors
                
            except Exception as e:
                logger.error(f"Erro ao adicionar ao índice FAISS: {str(e)}")
                raise
        else:
            # Fallback para NumPy
            start_index = self.embeddings.shape[0]
            
            # Concatenar novos embeddings
            self.embeddings = np.vstack([self.embeddings, embeddings])
            
            # Atualizar mapeamentos e metadados
            for i, (id_, meta) in enumerate(zip(ids, metadata)):
                idx = start_index + i
                self.id_to_index[id_] = idx
                self.index_to_id[idx] = id_
                self.metadata[id_] = meta
                
            # Atualizar tamanho do índice
            self.index_size = self.embeddings.shape[0]
            
        # Atualizar timestamp
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Adicionados {n_vectors} vetores ao índice, novo tamanho: {self.index_size}")
        
        return ids
        
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        include_distances: bool = True,
        filter_func: Optional[Callable[[str, Dict[str, Any]], bool]] = None
    ) -> Dict[str, Any]:
        """
        Busca os vetores mais próximos de um embedding de consulta.
        
        Args:
            query_embedding: Embedding de consulta (shape: [dimension] ou [1, dimension])
            k: Número de resultados a retornar
            include_distances: Se True, inclui distâncias/scores nos resultados
            filter_func: Função opcional para filtrar resultados com base no ID e metadados
            
        Returns:
            Dicionário com IDs, distâncias (opcional) e metadados
        """
        # Garantir formato correto
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Converter para float32
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalizar para similaridade de cosseno
        if self.metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            query_embedding = query_embedding / max(norm, 1e-10)
            
        # Aumentar k se estiver usando filtro
        search_k = k
        if filter_func is not None:
            search_k = min(k * 10, self.index_size)  # Buscar mais para compensar filtragem
            
        if FAISS_AVAILABLE and self.index is not None:
            try:
                # Buscar com FAISS
                distances, indices = self.index.search(query_embedding, search_k)
                
                # Converter para listas Python
                distances = distances[0].tolist()
                indices = indices[0].tolist()
                
            except Exception as e:
                logger.error(f"Erro na busca FAISS: {str(e)}")
                raise
        else:
            # Fallback para NumPy
            if self.embeddings.shape[0] == 0:
                return {"ids": [], "distances": [], "metadata": []}
                
            if self.metric == "cosine" or self.metric == "ip":
                # Similaridade por produto escalar (para vetores normalizados, isto é similaridade de cosseno)
                scores = np.dot(self.embeddings, query_embedding.T).flatten()
                indices = np.argsort(scores)[::-1][:search_k].tolist()
                distances = scores[indices].tolist()
            else:
                # Distância L2
                distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
                indices = np.argsort(distances)[:search_k].tolist()
                distances = distances[indices].tolist()
                
        # Mapear índices para IDs e aplicar filtro
        results = []
        
        for idx, dist in zip(indices, distances):
            # Verificar se o índice é válido
            if idx not in self.index_to_id:
                continue
                
            result_id = self.index_to_id[idx]
            meta = self.metadata.get(result_id, {})
            
            # Aplicar filtro se fornecido
            if filter_func is not None and not filter_func(result_id, meta):
                continue
                
            results.append((result_id, dist, meta))
            
            # Parar quando atingir k resultados após filtragem
            if len(results) >= k:
                break
                
        # Montar resposta
        ids = [r[0] for r in results]
        metadata_list = [r[2] for r in results]
        
        response = {
            "ids": ids,
            "metadata": metadata_list
        }
        
        if include_distances:
            response["distances"] = [r[1] for r in results]
            
        return response
        
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        include_distances: bool = True,
        filter_func: Optional[Callable[[str, Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca os vetores mais próximos para múltiplos embeddings de consulta.
        
        Args:
            query_embeddings: Embeddings de consulta (shape: [n_queries, dimension])
            k: Número de resultados a retornar por consulta
            include_distances: Se True, inclui distâncias/scores nos resultados
            filter_func: Função opcional para filtrar resultados
            
        Returns:
            Lista de dicionários com resultados para cada consulta
        """
        results = []
        
        # Processar cada consulta
        for i in range(query_embeddings.shape[0]):
            query_result = self.search(
                query_embeddings[i].reshape(1, -1),
                k=k,
                include_distances=include_distances,
                filter_func=filter_func
            )
            
            results.append(query_result)
            
        return results
        
        def remove(self, ids: List[Union[str, int]]) -> int:
        """
        Remove embeddings do índice.
        
        Args:
            ids: Lista de IDs a remover
            
        Returns:
            Número de embeddings removidos
        """
        if not ids:
            return 0
            
        # Para FAISS, a remoção é complicada e menos eficiente que reconstruir o índice
        # A estratégia é marcar os vetores como removidos em nossos metadados
        # e, opcionalmente, reconstruir o índice periodicamente
        
        removed_count = 0
        
        for id_ in ids:
            if id_ in self.id_to_index:
                # Marcar como removido nos metadados
                idx = self.id_to_index[id_]
                
                # Remover dos mapeamentos
                del self.id_to_index[id_]
                del self.index_to_id[idx]
                
                # Remover metadados
                if id_ in self.metadata:
                    del self.metadata[id_]
                    
                removed_count += 1
                
        # Se muitos vetores foram removidos, podemos reconstruir o índice
        # para recuperar espaço e melhorar a eficiência
        # Isso é deixado como uma otimização futura
        
        logger.info(f"Removidos {removed_count} vetores do índice")
        self.last_updated = datetime.now().isoformat()
        
        return removed_count
        
    def update(
        self,
        id_: Union[str, int],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Atualiza um embedding existente.
        
        Args:
            id_: ID do embedding
            embedding: Novo embedding
            metadata: Novos metadados (opcional)
            
        Returns:
            True se atualizado com sucesso, False se ID não encontrado
        """
        # Verificar se o ID existe
        if id_ not in self.id_to_index:
            return False
            
        # Remover o antigo e adicionar o novo
        self.remove([id_])
        self.add(embedding.reshape(1, -1), [id_], [metadata or {}])
        
        return True
        
    def get_metadata(self, id_: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Obtém metadados de um embedding.
        
        Args:
            id_: ID do embedding
            
        Returns:
            Dicionário de metadados ou None se ID não encontrado
        """
        return self.metadata.get(id_)
        
    def get_index_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o índice.
        
        Returns:
            Dicionário com informações do índice
        """
        return {
            "index_id": self.index_id,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "size": self.index_size,
            "is_trained": self.is_trained,
            "using_faiss": FAISS_AVAILABLE and self.index is not None,
            "using_gpu": self.use_gpu,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
        
    def save(self, file_path: Optional[str] = None) -> str:
        """
        Salva o índice em disco.
        
        Args:
            file_path: Caminho para salvar (opcional)
            
        Returns:
            Caminho onde o índice foi salvo
        """
        if file_path is None:
            # Gerar caminho padrão
            file_path = os.path.join(self.temp_dir, f"index_{self.index_id}.bin")
            
        try:
            # Preparar estado para salvar
            state = {
                "index_id": self.index_id,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
                "metadata": self.metadata,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
                "is_trained": self.is_trained,
                "index_size": self.index_size,
                "created_at": self.created_at,
                "last_updated": self.last_updated
            }
            
            # Salvar metadados e mapeamentos
            with open(file_path + ".meta", "wb") as f:
                pickle.dump(state, f)
                
            # Salvar o índice
            if FAISS_AVAILABLE and self.index is not None:
                # Mover para CPU se estiver na GPU
                if self.use_gpu:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, file_path + ".faiss")
                else:
                    faiss.write_index(self.index, file_path + ".faiss")
            else:
                # Salvar embeddings NumPy
                np.save(file_path + ".npy", self.embeddings)
                
            logger.info(f"Índice salvo em {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar índice: {str(e)}")
            raise
            
    @classmethod
    def load(cls, file_path: str, use_gpu: bool = False) -> "VectorIndex":
        """
        Carrega um índice do disco.
        
        Args:
            file_path: Caminho base do índice
            use_gpu: Se True, carrega o índice na GPU
            
        Returns:
            Instância de VectorIndex
        """
        try:
            # Carregar metadados e mapeamentos
            with open(file_path + ".meta", "rb") as f:
                state = pickle.load(f)
                
            # Criar instância vazia
            index = cls(
                dimension=state["dimension"],
                index_type=state["index_type"],
                metric=state["metric"],
                use_gpu=use_gpu
            )
            
            # Restaurar estado
            index.index_id = state["index_id"]
            index.metadata = state["metadata"]
            index.id_to_index = state["id_to_index"]
            index.index_to_id = state["index_to_id"]
            index.is_trained = state["is_trained"]
            index.index_size = state["index_size"]
            index.created_at = state["created_at"]
            index.last_updated = state["last_updated"]
            
            # Carregar o índice
            if FAISS_AVAILABLE and os.path.exists(file_path + ".faiss"):
                index.index = faiss.read_index(file_path + ".faiss")
                
                # Mover para GPU se solicitado
                if use_gpu:
                    try:
                        gpu_resources = faiss.StandardGpuResources()
                        index.index = faiss.index_cpu_to_gpu(gpu_resources, 0, index.index)
                        logger.info("Índice movido para GPU")
                    except Exception as e:
                        logger.warning(f"Não foi possível usar GPU: {str(e)}")
            elif os.path.exists(file_path + ".npy"):
                # Carregar embeddings NumPy
                index.embeddings = np.load(file_path + ".npy")
                
            logger.info(f"Índice carregado de {file_path}")
            
            return index
            
        except Exception as e:
            logger.error(f"Erro ao carregar índice: {str(e)}")
            raise
