"""
Schemas para validação de dados de busca semântica na API.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

class ContentTypeEnum(str, Enum):
    """Tipos de conteúdo para busca semântica."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    
class IndexTypeEnum(str, Enum):
    """Tipos de índice para busca vetorial."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    
class MetricTypeEnum(str, Enum):
    """Tipos de métrica para busca vetorial."""
    COSINE = "cosine"
    L2 = "l2"
    IP = "ip"  # Inner Product
    
class CreateIndexRequest(BaseModel):
    """Requisição para criar um índice vetorial."""
    content_type: ContentTypeEnum = Field(..., description="Tipo de conteúdo")
    dimension: Optional[int] = Field(None, description="Dimensão dos embeddings")
    index_type: IndexTypeEnum = Field(IndexTypeEnum.FLAT, description="Tipo de índice")
    metric: MetricTypeEnum = Field(MetricTypeEnum.COSINE, description="Métrica de distância")
    
class IndexTextRequest(BaseModel):
    """Requisição para indexar textos."""
    texts: Union[str, List[str]] = Field(..., description="Texto ou lista de textos")
    metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(None, description="Metadados")
    index_id: Optional[str] = Field(None, description="ID do índice (opcional)")
    
    @validator('texts')
    def validate_texts(cls, v):
        """Valida que há pelo menos um texto."""
        if isinstance(v, list) and not v:
            raise ValueError("A lista de textos não pode estar vazia")
        return v
    
class IndexImageRequest(BaseModel):
    """Requisição para indexar imagens."""
    images: Union[str, List[str]] = Field(..., description="Imagem ou lista de imagens em formato base64")
    metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(None, description="Metadados")
    index_id: Optional[str] = Field(None, description="ID do índice (opcional)")
    
    @validator('images')
    def validate_images(cls, v):
        """Valida que há pelo menos uma imagem."""
        if isinstance(v, list) and not v:
            raise ValueError("A lista de imagens não pode estar vazia")
        return v
    
class IndexRepositoryRequest(BaseModel):
    """Requisição para indexar conteúdo do repositório."""
    user_id: Optional[int] = Field(None, description="Filtrar por usuário")
    result_types: Optional[List[ContentTypeEnum]] = Field(None, description="Tipos de resultado a indexar")
    limit: int = Field(1000, description="Limite de itens por tipo")
    reindex: bool = Field(False, description="Se True, reindexar itens já indexados")
    
class SearchTextRequest(BaseModel):
    """Requisição para busca semântica de texto."""
    query: str = Field(..., description="Texto de consulta")
    index_id: Optional[str] = Field(None, description="ID do índice (opcional)")
    k: int = Field(10, description="Número de resultados")
    include_content: bool = Field(False, description="Se True, inclui conteúdo nos resultados")
    user_id: Optional[int] = Field(None, description="Filtrar por usuário")
    
    @validator('query')
    def validate_query(cls, v):
        """Valida que a consulta não está vazia."""
        if not v.strip():
            raise ValueError("A consulta não pode estar vazia")
        return v
    
class SearchImageRequest(BaseModel):
    """Requisição para busca semântica de imagem."""
    image: str = Field(..., description="Imagem de consulta em formato base64")
    index_id: Optional[str] = Field(None, description="ID do índice (opcional)")
    k: int = Field(10, description="Número de resultados")
    include_content: bool = Field(False, description="Se True, inclui conteúdo nos resultados")
    user_id: Optional[int] = Field(None, description="Filtrar por usuário")
    
class SearchImagesWithTextRequest(BaseModel):
    """Requisição para busca de imagens com texto."""
    query: str = Field(..., description="Texto de consulta")
    index_id: Optional[str] = Field(None, description="ID do índice (opcional)")
    k: int = Field(10, description="Número de resultados")
    include_content: bool = Field(False, description="Se True, inclui conteúdo nos resultados")
    user_id: Optional[int] = Field(None, description="Filtrar por usuário")
    
    @validator('query')
    def validate_query(cls, v):
        """Valida que a consulta não está vazia."""
        if not v.strip():
            raise ValueError("A consulta não pode estar vazia")
        return v
    
class IndexListResponse(BaseModel):
    """Resposta com lista de índices."""
    indices: List[Dict[str, Any]] = Field(..., description="Lista de índices")
    
class IndexResponse(BaseModel):
    """Resposta com informações de um índice."""
    index_id: str = Field(..., description="ID do índice")
    dimension: int = Field(..., description="Dimensão dos embeddings")
    index_type: str = Field(..., description="Tipo de índice")
    metric: str = Field(..., description="Métrica de distância")
    size: int = Field(..., description="Número de itens no índice")
    content_type: Optional[str] = Field(None, description="Tipo de conteúdo")
    
class IndexOperationResponse(BaseModel):
    """Resposta para operações de indexação."""
    success: bool = Field(..., description="Indicador de sucesso")
    index_id: str = Field(..., description="ID do índice")
    item_ids: Optional[List[str]] = Field(None, description="IDs dos itens indexados")
    count: Optional[int] = Field(None, description="Número de itens afetados")
    message: Optional[str] = Field(None, description="Mensagem adicional")
    
class IndexRepositoryResponse(BaseModel):
    """Resposta para indexação de repositório."""
    success: bool = Field(..., description="Indicador de sucesso")
    counts: Dict[str, int] = Field(..., description="Contagem por tipo")
    total: int = Field(..., description="Total de itens indexados")
    
class SearchResult(BaseModel):
    """Item de resultado de busca."""
    id: str = Field(..., description="ID do item")
    score: float = Field(..., description="Score de similaridade")
    metadata: Dict[str, Any] = Field(..., description="Metadados do item")
    content: Optional[Any] = Field(None, description="Conteúdo do item (opcional)")
    
class SearchResponse(BaseModel):
    """Resposta para busca semântica."""
    query: Optional[str] = Field(None, description="Consulta original")
    results: List[SearchResult] = Field(..., description="Resultados da busca")
    total: int = Field(..., description="Número total de resultados")
